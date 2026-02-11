"""
Iterative cost-per-solve evaluation framework.

Extends the pass@1 framework with retry loops: both control and treatment
arms get multiple attempts to solve each problem. When an attempt fails,
the error feedback is appended to the prompt for the next attempt. The
treatment arm optionally transforms the growing prompt each iteration.

The key metric is **cost-per-solve (CPS)**: total cost to get working code,
including retry loops with error feedback.

Inherits from Pass1Runner (reuses _call_llm, _execute_code, _apply_transform).

Statistical methodology:
- Wilson score intervals for final solve rates
- Bootstrap CI on cost-per-solve savings (likely non-normal)
- Paired difference test on per-problem costs
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

from .pass1_runner import (
    Pass1Runner,
    CONTROL_SYSTEM_PROMPT,
    TREATMENT_SYSTEM_PROMPT,
    sanitize_code,
)
from .base import BenchmarkProblem
from .humaneval import HumanEvalLoader
from .mbpp import MBPPLoader

from itereval.utils import estimate_tokens, get_model_pricing, calculate_cost, NumpyEncoder

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Error feedback template
# ---------------------------------------------------------------------------

ERROR_FEEDBACK_TEMPLATE = (
    "\n\n--- Previous attempt failed ---\n"
    "Error: {error}\n"
    "Your previous code:\n```python\n{code}\n```\n"
    "Fix the error and try again."
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class AttemptResult:
    """Single attempt within an iterative solve."""
    attempt_number: int          # 1-indexed
    code: str
    passed: bool
    error: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    execution_ms: float
    cumulative_input_tokens: int
    cumulative_output_tokens: int
    prompt_sent: str = ""        # Only populated when verbose=True


@dataclass
class IterativeProblemResult:
    """Per-problem result across all iterations."""
    task_id: str
    benchmark: str
    entry_point: str

    # Control arm
    control_solved: bool
    control_attempts: list[AttemptResult]
    control_solved_at: Optional[int]       # Which attempt solved it (1-indexed)
    control_total_input_tokens: int
    control_total_output_tokens: int
    control_total_cost: float

    # Treatment arm
    treatment_solved: bool
    treatment_attempts: list[AttemptResult]
    treatment_solved_at: Optional[int]
    treatment_total_input_tokens: int
    treatment_total_output_tokens: int
    treatment_total_cost: float

    # Transform metadata
    compression_ratio: float
    transform_name: str

    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        d = asdict(self)
        _strip_keys = {"code", "prompt_sent"}
        d["control_attempts"] = [
            {k: v for k, v in a.items() if k not in _strip_keys}
            for a in d["control_attempts"]
        ]
        d["treatment_attempts"] = [
            {k: v for k, v in a.items() if k not in _strip_keys}
            for a in d["treatment_attempts"]
        ]
        return d

    def to_jsonl_line(self) -> str:
        return json.dumps(self.to_dict())


@dataclass
class IterativeSummary:
    """Aggregated iterative results with cost-per-solve metrics."""
    # Config
    benchmark: str
    transform_name: str
    model: str
    max_attempts: int

    # Solve rates
    total_problems: int
    control_solved: int
    treatment_solved: int

    # THE KEY METRIC
    control_cost_per_solve: float
    treatment_cost_per_solve: float
    cost_per_solve_savings_pct: float

    # Per-iteration convergence (accuracy at attempt N)
    control_cumulative_solve_rate: list[float]
    treatment_cumulative_solve_rate: list[float]

    # Token/cost totals
    control_total_cost: float
    treatment_total_cost: float
    total_cost_savings_pct: float

    # Avg attempts to solve (for solved problems only)
    control_avg_attempts: float
    treatment_avg_attempts: float

    # Prompt configuration
    prompt_config: str = "split"

    # Wilson CIs on final solve rates
    control_solve_rate: float = 0.0
    control_ci_lower: float = 0.0
    control_ci_upper: float = 0.0
    treatment_solve_rate: float = 0.0
    treatment_ci_lower: float = 0.0
    treatment_ci_upper: float = 0.0

    # McNemar's test on final solve/not-solve
    mcnemar_both_passed: int = 0
    mcnemar_control_only: int = 0
    mcnemar_treatment_only: int = 0
    mcnemar_both_failed: int = 0
    mcnemar_p_value: float = 1.0
    mcnemar_method: str = ""
    mcnemar_significant: bool = False

    # Bootstrap CI on cost-per-solve savings (resampled over ALL problems)
    cps_bootstrap_ci_lower: float = 0.0
    cps_bootstrap_ci_upper: float = 0.0
    cps_bootstrap_n_problems: int = 0  # N used for bootstrap (all problems)
    cps_bootstrap_dropped_frac: float = 0.0  # fraction of iterations with 0 solves in an arm

    # Paired permutation test on per-problem costs
    cost_permutation_p_value: float = 1.0

    # Token breakdown
    control_total_input_tokens: int = 0
    control_total_output_tokens: int = 0
    treatment_total_input_tokens: int = 0
    treatment_total_output_tokens: int = 0
    input_token_savings_pct: float = 0.0
    output_token_savings_pct: float = 0.0

    # Solve-at-attempt histogram
    control_solve_at_histogram: dict[int, int] = field(default_factory=dict)
    treatment_solve_at_histogram: dict[int, int] = field(default_factory=dict)

    # First-attempt vs retry token breakdown
    control_first_attempt_input_tokens: int = 0
    control_retry_input_tokens: int = 0
    treatment_first_attempt_input_tokens: int = 0
    treatment_retry_input_tokens: int = 0

    # Error feedback token growth (avg input token increase per retry)
    control_avg_error_feedback_tokens: float = 0.0
    treatment_avg_error_feedback_tokens: float = 0.0

    # Improvement rate: % of retries that fix a previously-failing problem
    control_improvement_rate: float = 0.0
    treatment_improvement_rate: float = 0.0

    # Cost per attempt
    control_cost_per_attempt: float = 0.0
    treatment_cost_per_attempt: float = 0.0

    # Latency (API call time, not execution)
    control_avg_latency_ms: float = 0.0
    treatment_avg_latency_ms: float = 0.0
    control_total_latency_ms: float = 0.0
    treatment_total_latency_ms: float = 0.0

    # Timing
    total_runtime_seconds: float = 0.0

    # Per-problem results
    results: list[IterativeProblemResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = {k: v for k, v in asdict(self).items() if k != "results"}
        d["results_count"] = len(self.results)
        return d

    def format_report(self) -> str:
        """Format a human-readable console report."""
        lines = [
            "",
            "=" * 65,
            f"  Iterative Cost-Per-Solve: {self.transform_name}",
            f"  Model: {self.model}  |  Max Attempts: {self.max_attempts}",
            f"  Prompt config: {self.prompt_config}",
            "=" * 65,
            "",
            "  Solve Rates:",
            f"    Control:   {self.control_solve_rate:.1%}"
            f"  (Wilson 95% CI: [{self.control_ci_lower:.1%}, {self.control_ci_upper:.1%}])",
            f"    Treatment: {self.treatment_solve_rate:.1%}"
            f"  (Wilson 95% CI: [{self.treatment_ci_lower:.1%}, {self.treatment_ci_upper:.1%}])",
            "",
            "  Cost Per Solve:",
            f"    Control:   ${self.control_cost_per_solve:.4f}",
            f"    Treatment: ${self.treatment_cost_per_solve:.4f}"
            f"  ({self.cost_per_solve_savings_pct:.1f}% savings)",
            f"    Bootstrap 95% CI on savings: "
            f"[{self.cps_bootstrap_ci_lower:.1f}%, {self.cps_bootstrap_ci_upper:.1f}%]"
            f"  (N={self.cps_bootstrap_n_problems}, all problems"
            f"{f', {self.cps_bootstrap_dropped_frac:.1%} iterations dropped' if self.cps_bootstrap_dropped_frac > 0.01 else ''})",
            f"    Paired permutation test (per-problem total cost): p = {self.cost_permutation_p_value:.4f}",
            "",
            f"  McNemar's test ({self.mcnemar_method}):",
            f"    p = {self.mcnemar_p_value:.4f}  "
            f"({'SIGNIFICANT' if self.mcnemar_significant else 'not significant'})",
            f"    Contingency: both={self.mcnemar_both_passed}, "
            f"ctrl_only={self.mcnemar_control_only}, "
            f"treat_only={self.mcnemar_treatment_only}, "
            f"neither={self.mcnemar_both_failed}",
            "",
            "  Convergence (cumulative solve rate by attempt):",
            f"    {'Attempt':<12}{'Control':<12}{'Treatment':<12}",
        ]

        for i in range(self.max_attempts):
            ctrl_rate = self.control_cumulative_solve_rate[i] if i < len(self.control_cumulative_solve_rate) else 0.0
            treat_rate = self.treatment_cumulative_solve_rate[i] if i < len(self.treatment_cumulative_solve_rate) else 0.0
            lines.append(f"    {i+1:<12}{ctrl_rate:<12.1%}{treat_rate:<12.1%}")

        lines.extend([
            "",
            "  Average attempts to solve:",
            f"    Control:   {self.control_avg_attempts:.2f}",
            f"    Treatment: {self.treatment_avg_attempts:.2f}",
        ])

        if self.control_solve_at_histogram or self.treatment_solve_at_histogram:
            all_attempts = sorted(
                set(list(self.control_solve_at_histogram.keys())
                    + list(self.treatment_solve_at_histogram.keys()))
            )
            lines.extend([
                "",
                "  Solve-at-Attempt Distribution:",
                f"    {'Attempt':<12}{'Control':<12}{'Treatment':<12}",
            ])
            for a in all_attempts:
                ctrl_cnt = self.control_solve_at_histogram.get(a, 0)
                treat_cnt = self.treatment_solve_at_histogram.get(a, 0)
                lines.append(f"    {a:<12}{ctrl_cnt:<12}{treat_cnt:<12}")

        ctrl_retry_pct = (
            self.control_retry_input_tokens
            / max(1, self.control_first_attempt_input_tokens + self.control_retry_input_tokens)
            * 100
        )
        treat_retry_pct = (
            self.treatment_retry_input_tokens
            / max(1, self.treatment_first_attempt_input_tokens + self.treatment_retry_input_tokens)
            * 100
        )
        lines.extend([
            "",
            "  First-Attempt vs Retry Token Split:",
            f"    {'':20s}{'Control':>12s}{'Treatment':>12s}",
            f"    {'First-attempt in':20s}"
            f"{self.control_first_attempt_input_tokens:>12,}"
            f"{self.treatment_first_attempt_input_tokens:>12,}",
            f"    {'Retry input':20s}"
            f"{self.control_retry_input_tokens:>12,}"
            f"{self.treatment_retry_input_tokens:>12,}",
            f"    {'Retry overhead':20s}"
            f"{ctrl_retry_pct:>11.1f}%"
            f"{treat_retry_pct:>11.1f}%",
            "",
            "  Error Feedback Growth (avg tokens added per retry):",
            f"    Control:   {self.control_avg_error_feedback_tokens:+.0f} tokens",
            f"    Treatment: {self.treatment_avg_error_feedback_tokens:+.0f} tokens",
            "",
            "  Improvement Rate (% of retries that fix):",
            f"    Control:   {self.control_improvement_rate:.1f}%",
            f"    Treatment: {self.treatment_improvement_rate:.1f}%",
            "",
            "  Cost Per Attempt:",
            f"    Control:   ${self.control_cost_per_attempt:.6f}",
            f"    Treatment: ${self.treatment_cost_per_attempt:.6f}",
            "",
            "  Token Breakdown:",
            f"    {'':20s}{'Control':>12s}{'Treatment':>12s}{'Savings':>10s}",
            f"    {'Input tokens':20s}"
            f"{self.control_total_input_tokens:>12,}"
            f"{self.treatment_total_input_tokens:>12,}"
            f"{self.input_token_savings_pct:>9.1f}%",
            f"    {'Output tokens':20s}"
            f"{self.control_total_output_tokens:>12,}"
            f"{self.treatment_total_output_tokens:>12,}"
            f"{self.output_token_savings_pct:>9.1f}%",
            f"    {'Total cost':20s}"
            f"{'$'+f'{self.control_total_cost:.4f}':>12s}"
            f"{'$'+f'{self.treatment_total_cost:.4f}':>12s}"
            f"{self.total_cost_savings_pct:>9.1f}%",
            "",
            "  API Latency:",
            f"    {'':20s}{'Control':>12s}{'Treatment':>12s}",
            f"    {'Avg per call':20s}"
            f"{self.control_avg_latency_ms:>11.0f}ms"
            f"{self.treatment_avg_latency_ms:>11.0f}ms",
            f"    {'Total':20s}"
            f"{self.control_total_latency_ms:>10.0f}ms"
            f"{self.treatment_total_latency_ms:>10.0f}ms",
            "",
            f"  Runtime: {self.total_runtime_seconds:.1f}s",
            "=" * 65,
        ])
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

class IterativeRunner(Pass1Runner):
    """
    Iterative cost-per-solve evaluation runner.

    Extends Pass1Runner with retry loops: both arms get multiple attempts
    per problem. Error feedback is appended to the prompt between attempts.
    The treatment arm optionally re-transforms the growing prompt each iteration.
    """

    def __init__(
        self,
        *args,
        max_attempts: int = 5,
        budget_per_problem: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if max_attempts < 1:
            raise ValueError(f"max_attempts must be >= 1, got {max_attempts}")
        if budget_per_problem is not None and budget_per_problem <= 0:
            raise ValueError(f"budget_per_problem must be > 0, got {budget_per_problem}")
        self.max_attempts = max_attempts
        self.budget_per_problem = budget_per_problem

    async def run_problem_iterative(
        self,
        problem: BenchmarkProblem,
    ) -> IterativeProblemResult:
        """Run a single problem with iterative retry loops for both arms."""
        original_prompt = problem.prompt

        # --- Transform prompt for treatment (initial) ---
        transformed_prompt, _, _ = self._apply_transform(original_prompt)
        compression_ratio = (
            estimate_tokens(transformed_prompt, self.model)
            / max(1, estimate_tokens(original_prompt, self.model))
        )

        # === Control arm ===
        control_attempts: list[AttemptResult] = []
        ctrl_solved = False
        ctrl_solved_at: Optional[int] = None
        ctrl_cumulative_in = 0
        ctrl_cumulative_out = 0
        ctrl_prompt = original_prompt

        for attempt_num in range(1, self.max_attempts + 1):
            if self.budget_per_problem is not None:
                cost_so_far = calculate_cost(ctrl_cumulative_in, ctrl_cumulative_out, self.model)
                if cost_so_far >= self.budget_per_problem:
                    break

            try:
                resp_text, latency, in_tok, out_tok = await self._call_llm(
                    ctrl_prompt, self.control_system_prompt
                )
                code = sanitize_code(resp_text, problem.entry_point)
            except Exception as e:
                logger.error(f"Control LLM call failed for {problem.task_id} attempt {attempt_num}: {e}")
                code = ""
                latency, in_tok, out_tok = 0.0, 0, 0

            passed, error, exec_ms = self._execute_code(code, problem)

            ctrl_cumulative_in += in_tok
            ctrl_cumulative_out += out_tok

            control_attempts.append(AttemptResult(
                attempt_number=attempt_num,
                code=code,
                passed=passed,
                error=error,
                input_tokens=in_tok,
                output_tokens=out_tok,
                latency_ms=latency,
                execution_ms=exec_ms,
                cumulative_input_tokens=ctrl_cumulative_in,
                cumulative_output_tokens=ctrl_cumulative_out,
                prompt_sent=ctrl_prompt if self.verbose else "",
            ))

            if passed:
                ctrl_solved = True
                ctrl_solved_at = attempt_num
                break

            ctrl_prompt = original_prompt + ERROR_FEEDBACK_TEMPLATE.format(
                error=error, code=code
            )

        ctrl_total_cost = calculate_cost(ctrl_cumulative_in, ctrl_cumulative_out, self.model)

        # === Treatment arm ===
        treatment_attempts: list[AttemptResult] = []
        treat_solved = False
        treat_solved_at: Optional[int] = None
        treat_cumulative_in = 0
        treat_cumulative_out = 0
        treat_full_prompt = original_prompt  # Uncompressed accumulator

        for attempt_num in range(1, self.max_attempts + 1):
            if self.budget_per_problem is not None:
                cost_so_far = calculate_cost(treat_cumulative_in, treat_cumulative_out, self.model)
                if cost_so_far >= self.budget_per_problem:
                    break

            # Transform the full prompt (original + any error feedback)
            if attempt_num == 1:
                prompt_to_send = transformed_prompt
            else:
                prompt_to_send, _, _ = self._apply_transform(treat_full_prompt)

            try:
                resp_text, latency, in_tok, out_tok = await self._call_llm(
                    prompt_to_send, self.treatment_system_prompt
                )
                code = sanitize_code(resp_text, problem.entry_point)
            except Exception as e:
                logger.error(f"Treatment LLM call failed for {problem.task_id} attempt {attempt_num}: {e}")
                code = ""
                latency, in_tok, out_tok = 0.0, 0, 0

            passed, error, exec_ms = self._execute_code(code, problem)

            treat_cumulative_in += in_tok
            treat_cumulative_out += out_tok

            treatment_attempts.append(AttemptResult(
                attempt_number=attempt_num,
                code=code,
                passed=passed,
                error=error,
                input_tokens=in_tok,
                output_tokens=out_tok,
                latency_ms=latency,
                execution_ms=exec_ms,
                cumulative_input_tokens=treat_cumulative_in,
                cumulative_output_tokens=treat_cumulative_out,
                prompt_sent=prompt_to_send if self.verbose else "",
            ))

            if passed:
                treat_solved = True
                treat_solved_at = attempt_num
                break

            treat_full_prompt = original_prompt + ERROR_FEEDBACK_TEMPLATE.format(
                error=error, code=code
            )

        treat_total_cost = calculate_cost(treat_cumulative_in, treat_cumulative_out, self.model)

        return IterativeProblemResult(
            task_id=problem.task_id,
            benchmark=problem.benchmark,
            entry_point=problem.entry_point,
            control_solved=ctrl_solved,
            control_attempts=control_attempts,
            control_solved_at=ctrl_solved_at,
            control_total_input_tokens=ctrl_cumulative_in,
            control_total_output_tokens=ctrl_cumulative_out,
            control_total_cost=ctrl_total_cost,
            treatment_solved=treat_solved,
            treatment_attempts=treatment_attempts,
            treatment_solved_at=treat_solved_at,
            treatment_total_input_tokens=treat_cumulative_in,
            treatment_total_output_tokens=treat_cumulative_out,
            treatment_total_cost=treat_total_cost,
            compression_ratio=compression_ratio,
            transform_name=self.transform_name,
        )

    async def run(
        self,
        n_problems: Optional[int] = None,
        max_concurrent: Optional[int] = None,
        benchmark: str = "humaneval",
        progress_callback=None,
    ) -> IterativeSummary:
        """Run the full iterative cost-per-solve evaluation."""
        start_time = time.perf_counter()

        loader = HumanEvalLoader() if benchmark == "humaneval" else MBPPLoader()
        problems = loader.load(n=n_problems)
        if not problems:
            logger.warning("No problems loaded")
            return self._empty_iterative_summary(benchmark)

        logger.info(
            f"Running iterative evaluation: {len(problems)} {benchmark} problems, "
            f"transform={self.transform_name}, max_attempts={self.max_attempts}"
        )

        self.output_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        jsonl_path = self.output_dir / (
            f"iterative_{self.transform_name}_{self.prompt_tag}_{ts}.jsonl"
        )

        verbose_path = None
        if self.verbose:
            verbose_path = jsonl_path.with_suffix(".verbose.log")
            self._verbose_file = open(verbose_path, "w", encoding="utf-8")
            self._verbose_header_written = False
            self._verbose_file.write(
                f"WARNING: This log may contain prompt content and generated code.\n"
                f"Do not commit to public repositories without review.\n\n"
                f"Iterative Cost-Per-Solve Verbose Log\n"
                f"Transform: {self.transform_name}\n"
                f"Model: {self.model}  |  Max Attempts: {self.max_attempts}\n"
                f"Benchmark: {benchmark}  |  Problems: {len(problems)}\n"
                f"{'=' * 65}\n"
            )

        semaphore = asyncio.Semaphore(max_concurrent or self.max_concurrent)
        jsonl_lock = asyncio.Lock()
        completed = 0
        results: list[IterativeProblemResult] = []

        async def run_one(problem: BenchmarkProblem) -> IterativeProblemResult:
            nonlocal completed
            async with semaphore:
                result = await self.run_problem_iterative(problem)
                completed += 1
                if progress_callback:
                    progress_callback(completed, len(problems))
                try:
                    async with jsonl_lock:
                        with open(jsonl_path, "a", encoding="utf-8") as f:
                            f.write(result.to_jsonl_line() + "\n")
                except Exception as e:
                    logger.warning(f"Failed to write JSONL: {e}")

                if self.verbose and self._verbose_file:
                    self._write_iterative_verbose(result)

                return result

        try:
            tasks = [run_one(p) for p in problems]
            results = list(await asyncio.gather(*tasks))

            runtime = time.perf_counter() - start_time
            summary = self._build_iterative_summary(results, benchmark, runtime)

            if self._verbose_file:
                self._verbose_file.write(f"\n{'=' * 65}\n")
                self._verbose_file.write(summary.format_report())
                self._verbose_file.write("\n")

            summary_path = jsonl_path.with_suffix(".json")
            try:
                with open(summary_path, "w", encoding="utf-8") as f:
                    json.dump(summary.to_dict(), f, indent=2, cls=NumpyEncoder)
                print(f"Results: {jsonl_path}")
                print(f"Summary: {summary_path}")
                if verbose_path:
                    print(f"Verbose: {verbose_path}")
            except Exception as e:
                logger.warning(f"Failed to write summary: {e}")

            return summary
        finally:
            if self._verbose_file:
                self._verbose_file.close()
                self._verbose_file = None

    def run_sync(
        self,
        n_problems: Optional[int] = None,
        benchmark: str = "humaneval",
        progress_callback=None,
    ) -> IterativeSummary:
        """Synchronous wrapper for run()."""
        return asyncio.run(
            self.run(
                n_problems=n_problems,
                benchmark=benchmark,
                progress_callback=progress_callback,
            )
        )

    def _build_iterative_summary(
        self,
        results: list[IterativeProblemResult],
        benchmark: str,
        runtime: float,
    ) -> IterativeSummary:
        """Build IterativeSummary with cost-per-solve and convergence metrics."""
        import random
        from itereval.statistics import (
            wilson_score_interval,
            mcnemar_test,
            paired_permutation_test,
        )

        n = len(results)
        if n == 0:
            return self._empty_iterative_summary(benchmark)

        ctrl_solved_count = sum(1 for r in results if r.control_solved)
        treat_solved_count = sum(1 for r in results if r.treatment_solved)

        ctrl_total_cost = sum(r.control_total_cost for r in results)
        treat_total_cost = sum(r.treatment_total_cost for r in results)

        ctrl_cost_per_solve = (
            ctrl_total_cost / ctrl_solved_count
            if ctrl_solved_count > 0 else 0.0
        )
        treat_cost_per_solve = (
            treat_total_cost / treat_solved_count
            if treat_solved_count > 0 else 0.0
        )

        cps_savings_pct = (
            (1 - treat_cost_per_solve / ctrl_cost_per_solve) * 100
            if ctrl_cost_per_solve > 0 else 0.0
        )

        # Cumulative solve rate by attempt
        ctrl_cumulative = []
        treat_cumulative = []
        for attempt_i in range(1, self.max_attempts + 1):
            ctrl_solved_by_i = sum(
                1 for r in results
                if r.control_solved and r.control_solved_at is not None
                and r.control_solved_at <= attempt_i
            )
            treat_solved_by_i = sum(
                1 for r in results
                if r.treatment_solved and r.treatment_solved_at is not None
                and r.treatment_solved_at <= attempt_i
            )
            ctrl_cumulative.append(ctrl_solved_by_i / n)
            treat_cumulative.append(treat_solved_by_i / n)

        # Avg attempts to solve
        ctrl_solved_ats = [
            r.control_solved_at for r in results
            if r.control_solved and r.control_solved_at is not None
        ]
        treat_solved_ats = [
            r.treatment_solved_at for r in results
            if r.treatment_solved and r.treatment_solved_at is not None
        ]
        ctrl_avg_attempts = sum(ctrl_solved_ats) / len(ctrl_solved_ats) if ctrl_solved_ats else 0.0
        treat_avg_attempts = sum(treat_solved_ats) / len(treat_solved_ats) if treat_solved_ats else 0.0

        # Wilson CIs
        ctrl_wilson = wilson_score_interval(ctrl_solved_count, n)
        treat_wilson = wilson_score_interval(treat_solved_count, n)

        # McNemar's test on final solve/not-solve outcomes
        both_passed = sum(1 for r in results if r.control_solved and r.treatment_solved)
        control_only = sum(1 for r in results if r.control_solved and not r.treatment_solved)
        treatment_only = sum(1 for r in results if not r.control_solved and r.treatment_solved)
        both_failed = sum(1 for r in results if not r.control_solved and not r.treatment_solved)
        mcnemar = mcnemar_test(both_passed, control_only, treatment_only, both_failed)

        # Bootstrap CI on CPS savings — resampled over ALL N problems.
        #
        # Each bootstrap iteration:
        #   1. Sample N problems with replacement from the full result set
        #   2. Compute control CPS = sum(ctrl costs) / count(ctrl solved)
        #   3. Compute treatment CPS = sum(treat costs) / count(treat solved)
        #   4. Record savings = (1 - treat_CPS / ctrl_CPS) * 100
        #
        # This captures both cost variation AND solve-rate variation in the CI,
        # unlike conditioning on both-solved which only measures per-problem
        # cost differences.
        #
        # NOTE: Iterations where either arm has zero solves are dropped because
        # CPS is undefined (division by zero). When solve rates are low this
        # can bias the CI optimistically — the fraction of dropped iterations
        # is tracked so callers can assess the impact.
        cps_bootstrap_lower = 0.0
        cps_bootstrap_upper = 0.0
        cps_bootstrap_dropped_frac = 0.0
        n_bootstrap = 1000
        if ctrl_solved_count > 0 and treat_solved_count > 0 and n >= 2:
            rng = random.Random(42)
            bootstrap_savings = []
            n_dropped = 0
            for _ in range(n_bootstrap):
                sample = rng.choices(results, k=n)
                bs_ctrl_cost = sum(r.control_total_cost for r in sample)
                bs_treat_cost = sum(r.treatment_total_cost for r in sample)
                bs_ctrl_solved = sum(1 for r in sample if r.control_solved)
                bs_treat_solved = sum(1 for r in sample if r.treatment_solved)
                if bs_ctrl_solved > 0 and bs_treat_solved > 0:
                    bs_ctrl_cps = bs_ctrl_cost / bs_ctrl_solved
                    bs_treat_cps = bs_treat_cost / bs_treat_solved
                    if bs_ctrl_cps > 0:
                        bootstrap_savings.append(
                            (1 - bs_treat_cps / bs_ctrl_cps) * 100
                        )
                    else:
                        n_dropped += 1
                else:
                    n_dropped += 1
            cps_bootstrap_dropped_frac = n_dropped / n_bootstrap
            if len(bootstrap_savings) >= 2:
                bootstrap_savings.sort()
                alpha = 0.05
                lower_idx = max(0, round(len(bootstrap_savings) * alpha / 2) - 1)
                upper_idx = min(
                    len(bootstrap_savings) - 1,
                    round(len(bootstrap_savings) * (1 - alpha / 2)) - 1,
                )
                cps_bootstrap_lower = bootstrap_savings[lower_idx]
                cps_bootstrap_upper = bootstrap_savings[upper_idx]

        # Paired permutation test on per-problem costs
        cost_perm_p = 1.0
        if n >= 2:
            ctrl_costs = [r.control_total_cost for r in results]
            treat_costs = [r.treatment_total_cost for r in results]
            cost_perm_p = paired_permutation_test(
                ctrl_costs, treat_costs,
                n_resamples=9999, alternative="greater", seed=42,
            )

        # Token breakdown
        ctrl_in = sum(r.control_total_input_tokens for r in results)
        ctrl_out = sum(r.control_total_output_tokens for r in results)
        treat_in = sum(r.treatment_total_input_tokens for r in results)
        treat_out = sum(r.treatment_total_output_tokens for r in results)

        def _pct_savings(control: float, treatment: float) -> float:
            return (1 - treatment / control) * 100 if control > 0 else 0.0

        # Solve-at-attempt histogram
        ctrl_hist: dict[int, int] = {}
        treat_hist: dict[int, int] = {}
        for r in results:
            if r.control_solved and r.control_solved_at is not None:
                ctrl_hist[r.control_solved_at] = ctrl_hist.get(r.control_solved_at, 0) + 1
            if r.treatment_solved and r.treatment_solved_at is not None:
                treat_hist[r.treatment_solved_at] = treat_hist.get(r.treatment_solved_at, 0) + 1

        # First-attempt vs retry token breakdown
        ctrl_first_in = sum(
            r.control_attempts[0].input_tokens for r in results if r.control_attempts
        )
        treat_first_in = sum(
            r.treatment_attempts[0].input_tokens for r in results if r.treatment_attempts
        )
        ctrl_retry_in = ctrl_in - ctrl_first_in
        treat_retry_in = treat_in - treat_first_in

        # Error feedback token growth
        ctrl_deltas: list[int] = []
        treat_deltas: list[int] = []
        for r in results:
            for i in range(1, len(r.control_attempts)):
                ctrl_deltas.append(
                    r.control_attempts[i].input_tokens - r.control_attempts[i - 1].input_tokens
                )
            for i in range(1, len(r.treatment_attempts)):
                treat_deltas.append(
                    r.treatment_attempts[i].input_tokens - r.treatment_attempts[i - 1].input_tokens
                )
        ctrl_avg_feedback = sum(ctrl_deltas) / len(ctrl_deltas) if ctrl_deltas else 0.0
        treat_avg_feedback = sum(treat_deltas) / len(treat_deltas) if treat_deltas else 0.0

        # Improvement rate
        ctrl_fixes = 0
        ctrl_retries_after_fail = 0
        treat_fixes = 0
        treat_retries_after_fail = 0
        for r in results:
            for i in range(1, len(r.control_attempts)):
                if not r.control_attempts[i - 1].passed:
                    ctrl_retries_after_fail += 1
                    if r.control_attempts[i].passed:
                        ctrl_fixes += 1
            for i in range(1, len(r.treatment_attempts)):
                if not r.treatment_attempts[i - 1].passed:
                    treat_retries_after_fail += 1
                    if r.treatment_attempts[i].passed:
                        treat_fixes += 1
        ctrl_improvement = ctrl_fixes / ctrl_retries_after_fail * 100 if ctrl_retries_after_fail > 0 else 0.0
        treat_improvement = treat_fixes / treat_retries_after_fail * 100 if treat_retries_after_fail > 0 else 0.0

        # Cost per attempt
        ctrl_total_attempts = sum(len(r.control_attempts) for r in results)
        treat_total_attempts = sum(len(r.treatment_attempts) for r in results)
        ctrl_cost_per_attempt = ctrl_total_cost / ctrl_total_attempts if ctrl_total_attempts > 0 else 0.0
        treat_cost_per_attempt = treat_total_cost / treat_total_attempts if treat_total_attempts > 0 else 0.0

        # Latency
        ctrl_latencies = [
            a.latency_ms for r in results for a in r.control_attempts
        ]
        treat_latencies = [
            a.latency_ms for r in results for a in r.treatment_attempts
        ]
        ctrl_total_latency = sum(ctrl_latencies)
        treat_total_latency = sum(treat_latencies)
        ctrl_avg_latency = ctrl_total_latency / len(ctrl_latencies) if ctrl_latencies else 0.0
        treat_avg_latency = treat_total_latency / len(treat_latencies) if treat_latencies else 0.0

        return IterativeSummary(
            benchmark=benchmark,
            transform_name=self.transform_name,
            model=self.model,
            max_attempts=self.max_attempts,
            prompt_config=self.prompt_tag,
            total_problems=n,
            control_solved=ctrl_solved_count,
            treatment_solved=treat_solved_count,
            control_cost_per_solve=ctrl_cost_per_solve,
            treatment_cost_per_solve=treat_cost_per_solve,
            cost_per_solve_savings_pct=cps_savings_pct,
            control_cumulative_solve_rate=ctrl_cumulative,
            treatment_cumulative_solve_rate=treat_cumulative,
            control_total_cost=ctrl_total_cost,
            treatment_total_cost=treat_total_cost,
            total_cost_savings_pct=_pct_savings(ctrl_total_cost, treat_total_cost),
            control_avg_attempts=ctrl_avg_attempts,
            treatment_avg_attempts=treat_avg_attempts,
            control_solve_rate=ctrl_wilson.proportion,
            control_ci_lower=ctrl_wilson.ci_lower,
            control_ci_upper=ctrl_wilson.ci_upper,
            treatment_solve_rate=treat_wilson.proportion,
            treatment_ci_lower=treat_wilson.ci_lower,
            treatment_ci_upper=treat_wilson.ci_upper,
            mcnemar_both_passed=both_passed,
            mcnemar_control_only=control_only,
            mcnemar_treatment_only=treatment_only,
            mcnemar_both_failed=both_failed,
            mcnemar_p_value=mcnemar.p_value,
            mcnemar_method=mcnemar.method,
            mcnemar_significant=mcnemar.is_significant,
            cps_bootstrap_ci_lower=cps_bootstrap_lower,
            cps_bootstrap_ci_upper=cps_bootstrap_upper,
            cps_bootstrap_n_problems=n,
            cps_bootstrap_dropped_frac=cps_bootstrap_dropped_frac,
            cost_permutation_p_value=cost_perm_p,
            control_total_input_tokens=ctrl_in,
            control_total_output_tokens=ctrl_out,
            treatment_total_input_tokens=treat_in,
            treatment_total_output_tokens=treat_out,
            input_token_savings_pct=_pct_savings(ctrl_in, treat_in),
            output_token_savings_pct=_pct_savings(ctrl_out, treat_out),
            control_solve_at_histogram=ctrl_hist,
            treatment_solve_at_histogram=treat_hist,
            control_first_attempt_input_tokens=ctrl_first_in,
            control_retry_input_tokens=ctrl_retry_in,
            treatment_first_attempt_input_tokens=treat_first_in,
            treatment_retry_input_tokens=treat_retry_in,
            control_avg_error_feedback_tokens=ctrl_avg_feedback,
            treatment_avg_error_feedback_tokens=treat_avg_feedback,
            control_improvement_rate=ctrl_improvement,
            treatment_improvement_rate=treat_improvement,
            control_cost_per_attempt=ctrl_cost_per_attempt,
            treatment_cost_per_attempt=treat_cost_per_attempt,
            control_avg_latency_ms=ctrl_avg_latency,
            treatment_avg_latency_ms=treat_avg_latency,
            control_total_latency_ms=ctrl_total_latency,
            treatment_total_latency_ms=treat_total_latency,
            total_runtime_seconds=runtime,
            results=results,
        )

    def _empty_iterative_summary(self, benchmark: str) -> IterativeSummary:
        return IterativeSummary(
            benchmark=benchmark,
            transform_name=self.transform_name,
            model=self.model,
            max_attempts=self.max_attempts,
            prompt_config=self.prompt_tag,
            total_problems=0,
            control_solved=0,
            treatment_solved=0,
            control_cost_per_solve=0.0,
            treatment_cost_per_solve=0.0,
            cost_per_solve_savings_pct=0.0,
            control_cumulative_solve_rate=[],
            treatment_cumulative_solve_rate=[],
            control_total_cost=0.0,
            treatment_total_cost=0.0,
            total_cost_savings_pct=0.0,
            control_avg_attempts=0.0,
            treatment_avg_attempts=0.0,
        )

    def _write_iterative_verbose(self, result: IterativeProblemResult) -> None:
        """Write per-problem verbose output for iterative results."""
        f = self._verbose_file
        if f is None:
            return

        sep = "-" * 65

        def w(line: str = "") -> None:
            f.write(line + "\n")

        if not self._verbose_header_written:
            self._verbose_header_written = True
            w(f"\n  [Control System Prompt]")
            w(f"    {self.control_system_prompt}")
            w(f"\n  [Treatment System Prompt]")
            w(f"    {self.treatment_system_prompt}")

        w(f"\n{sep}")
        w(f"  {result.task_id} ({result.entry_point})")
        w(sep)

        for arm_name, attempts, solved, solved_at in [
            ("Control", result.control_attempts, result.control_solved, result.control_solved_at),
            ("Treatment", result.treatment_attempts, result.treatment_solved, result.treatment_solved_at),
        ]:
            status = f"SOLVED at attempt {solved_at}" if solved else "UNSOLVED"
            w(f"\n  [{arm_name}] {status}")

            for a in attempts:
                a_status = "PASS" if a.passed else "FAIL"
                w(f"\n    --- Attempt {a.attempt_number} [{a_status}] ---")
                w(f"    Input tokens: {a.input_tokens}  |  Output tokens: {a.output_tokens}")
                w(f"    Cumulative: in={a.cumulative_input_tokens}, out={a.cumulative_output_tokens}")

                if a.prompt_sent:
                    prompt_tok = estimate_tokens(a.prompt_sent, self.model)
                    w(f"\n    [Prompt Sent] ({prompt_tok} est. tokens)")
                    for line in a.prompt_sent.strip().splitlines():
                        w(f"      {line}")

                if a.code:
                    w(f"\n    [Generated Code]")
                    for line in a.code.strip().splitlines():
                        w(f"      {line}")

                if a.error:
                    w(f"\n    [Error Output]")
                    for line in a.error.strip().splitlines():
                        w(f"      {line}")

        w()
        f.flush()

    def get_config(self) -> dict:
        """Return current configuration as a dict."""
        config = super().get_config()
        config["max_attempts"] = self.max_attempts
        config["budget_per_problem"] = self.budget_per_problem
        return config
