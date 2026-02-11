"""
Pass@1 evaluation framework for LLM code generation.

Tests whether a treatment condition (e.g., different system prompt, input
transformation) degrades code generation quality using deterministic pass@1
(temperature=0, greedy decoding) with Wilson score CIs and McNemar's test
for paired binary comparison.

The default treatment uses a system prompt instructing the LLM to write
compact/minified code, which reduces output tokens and thus API cost.

Statistical methodology:
- Wilson score intervals for pass@1 CIs (not CLT/Wald)
- McNemar's test for paired binary outcomes (not paired t-test)
- Per ICML 2025 "Don't Use CLT in LLM Evals With Fewer Than a Few Hundred Datapoints"
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Callable

from .base import BenchmarkProblem
from .humaneval import HumanEvalLoader
from .mbpp import MBPPLoader
from .sanitize import sanitize_code

logger = logging.getLogger(__name__)

from itereval.utils import estimate_tokens, get_model_pricing, calculate_cost, NumpyEncoder


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

CONTROL_SYSTEM_PROMPT = (
    "You are a Python coding assistant. Write correct, clean Python code. "
    "Return ONLY the requested function implementation. "
    "Do not include explanations, tests, or example usage."
)

TREATMENT_SYSTEM_PROMPT = (
    "You are a Python coding assistant. Write correct Python code in a compact, "
    "minified style: use short variable names (single letters where unambiguous), "
    "collapse simple logic onto fewer lines, omit unnecessary whitespace and blank "
    "lines. Return ONLY the requested function implementation. "
    "Do not include explanations, tests, or example usage."
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Pass1ProblemResult:
    """Per-problem result for pass@1 evaluation."""
    task_id: str
    benchmark: str
    entry_point: str
    control_passed: bool
    treatment_passed: bool

    # Control (uncompressed prompt, control system prompt)
    control_code: str = ""
    control_error: str = ""
    control_input_tokens: int = 0
    control_output_tokens: int = 0
    control_latency_ms: float = 0.0

    # Treatment (optionally transformed prompt, treatment system prompt)
    treatment_code: str = ""
    treatment_error: str = ""
    treatment_input_tokens: int = 0
    treatment_output_tokens: int = 0
    treatment_latency_ms: float = 0.0

    # Transformation metadata
    original_prompt: str = ""
    transformed_prompt: str = ""
    compression_ratio: float = 1.0
    compression_latency_ms: float = 0.0
    transform_name: str = ""

    # Execution timing
    control_execution_ms: float = 0.0
    treatment_execution_ms: float = 0.0

    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        d = asdict(self)
        # Drop large text fields from JSONL for compactness
        d.pop("original_prompt", None)
        d.pop("transformed_prompt", None)
        d.pop("control_code", None)
        d.pop("treatment_code", None)
        return d

    def to_jsonl_line(self) -> str:
        return json.dumps(self.to_dict())


@dataclass
class Pass1Summary:
    """Aggregated pass@1 results with Wilson CIs and McNemar test."""
    benchmark: str
    transform_name: str
    model: str

    total_problems: int
    control_passed: int
    treatment_passed: int
    both_passed: int
    control_only: int  # regressions: control passed, treatment failed
    treatment_only: int  # improvements: treatment passed, control failed
    both_failed: int

    # Prompt configuration
    prompt_config: str = "split"

    # Wilson score CIs
    control_pass_rate: float = 0.0
    control_ci_lower: float = 0.0
    control_ci_upper: float = 0.0
    treatment_pass_rate: float = 0.0
    treatment_ci_lower: float = 0.0
    treatment_ci_upper: float = 0.0

    # McNemar's test
    mcnemar_statistic: float = 0.0
    mcnemar_p_value: float = 1.0
    mcnemar_method: str = ""
    mcnemar_significant: bool = False

    # Token breakdown (from API response.usage)
    control_input_tokens: int = 0
    control_output_tokens: int = 0
    treatment_input_tokens: int = 0
    treatment_output_tokens: int = 0

    # Token savings (derived)
    avg_compression_ratio: float = 1.0
    total_control_tokens: int = 0
    total_treatment_tokens: int = 0
    input_token_savings_pct: float = 0.0
    output_token_savings_pct: float = 0.0
    total_token_savings_pct: float = 0.0

    # Cost breakdown (USD)
    control_input_cost: float = 0.0
    control_output_cost: float = 0.0
    control_total_cost: float = 0.0
    treatment_input_cost: float = 0.0
    treatment_output_cost: float = 0.0
    treatment_total_cost: float = 0.0
    total_cost_savings_pct: float = 0.0

    # Timing
    total_runtime_seconds: float = 0.0

    # Per-problem results
    results: list[Pass1ProblemResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = {k: v for k, v in asdict(self).items() if k != "results"}
        d["results_count"] = len(self.results)
        return d

    def format_report(self) -> str:
        """Format a human-readable console report."""
        lines = [
            "",
            "=" * 65,
            f"  Pass@1 Results: {self.transform_name}",
            f"  Model: {self.model}  |  Prompt config: {self.prompt_config}",
            "=" * 65,
            "",
            f"  Problems: {self.total_problems}",
            f"  Control pass@1:   {self.control_pass_rate:.1%} "
            f"  (Wilson 95% CI: [{self.control_ci_lower:.1%}, {self.control_ci_upper:.1%}])",
            f"  Treatment pass@1: {self.treatment_pass_rate:.1%} "
            f"  (Wilson 95% CI: [{self.treatment_ci_lower:.1%}, {self.treatment_ci_upper:.1%}])",
            "",
            "  Contingency Table:",
            f"    Both passed:      {self.both_passed}",
            f"    Control only:     {self.control_only}  (regressions)",
            f"    Treatment only:   {self.treatment_only}  (improvements)",
            f"    Both failed:      {self.both_failed}",
            "",
            f"  McNemar's test ({self.mcnemar_method}):",
            f"    statistic = {self.mcnemar_statistic:.4f},  "
            f"p = {self.mcnemar_p_value:.4f}  "
            f"({'SIGNIFICANT' if self.mcnemar_significant else 'not significant'})",
            "",
            "  Token Breakdown:",
            f"    {'':30s} {'Control':>10s} {'Treatment':>10s} {'Savings':>10s}",
            f"    {'Input tokens':30s} {self.control_input_tokens:>10,} {self.treatment_input_tokens:>10,} {self.input_token_savings_pct:>9.1f}%",
            f"    {'Output tokens':30s} {self.control_output_tokens:>10,} {self.treatment_output_tokens:>10,} {self.output_token_savings_pct:>9.1f}%",
            f"    {'Total tokens':30s} {self.total_control_tokens:>10,} {self.total_treatment_tokens:>10,} {self.total_token_savings_pct:>9.1f}%",
            "",
            "  Cost Breakdown:",
            f"    {'':30s} {'Control':>10s} {'Treatment':>10s} {'Savings':>10s}",
            f"    {'Input cost':30s} {'$'+f'{self.control_input_cost:.4f}':>10s} {'$'+f'{self.treatment_input_cost:.4f}':>10s}",
            f"    {'Output cost':30s} {'$'+f'{self.control_output_cost:.4f}':>10s} {'$'+f'{self.treatment_output_cost:.4f}':>10s}",
            f"    {'Total cost':30s} {'$'+f'{self.control_total_cost:.4f}':>10s} {'$'+f'{self.treatment_total_cost:.4f}':>10s} {self.total_cost_savings_pct:>9.1f}%",
            "",
            f"  Avg prompt compression ratio: {self.avg_compression_ratio:.2f}",
            f"  Runtime: {self.total_runtime_seconds:.1f}s",
            "=" * 65,
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def _load_env_file():
    """Load environment variables from .env file."""
    search_paths = [
        Path.cwd() / ".env",
        Path(__file__).parent.parent.parent / ".env",
    ]
    for env_path in search_paths:
        if env_path.exists():
            try:
                with open(env_path) as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#") and "=" in line:
                            key, _, value = line.partition("=")
                            key = key.strip()
                            value = value.strip().strip('"').strip("'")
                            if key and value and key not in os.environ:
                                os.environ[key] = value
                return
            except (FileNotFoundError, OSError, ValueError):
                pass


class Pass1Runner:
    """
    Deterministic pass@1 evaluation runner.

    Runs HumanEval/MBPP problems at temperature=0 with control and treatment
    conditions, then compares using Wilson score CIs and McNemar's test.

    The treatment condition can differ from control in two ways:
    1. Different system prompt (default: conciseness instruction)
    2. Optional prompt_transform function applied to the input

    Args:
        prompt_transform: Optional callable that transforms the user prompt
            for the treatment arm. Signature: (text: str) -> str.
            When None, the treatment arm receives the same prompt as control.
        transform_name: Human-readable name for the transform (for reports).
        system_prompt: Treatment system prompt override.
        control_system_prompt: Control system prompt override.
        model: LLM model identifier.
        max_tokens: Max output tokens per API call.
        timeout_seconds: Code execution timeout.
        output_dir: Directory for result files.
        max_concurrent: Max concurrent API calls.
    """

    def __init__(
        self,
        prompt_transform: Optional[Callable[[str], str]] = None,
        transform_name: str = "prompt_only",
        system_prompt: Optional[str] = None,
        control_system_prompt: Optional[str] = None,
        model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 2048,
        timeout_seconds: float = 30.0,
        output_dir: Optional[Path] = None,
        max_concurrent: int = 5,
    ):
        _load_env_file()
        self.prompt_transform = prompt_transform
        self._transform_name = transform_name
        self.treatment_system_prompt = system_prompt or TREATMENT_SYSTEM_PROMPT
        self.control_system_prompt = control_system_prompt or CONTROL_SYSTEM_PROMPT
        self.model = model
        self.max_tokens = max_tokens
        self.timeout_seconds = timeout_seconds
        self.output_dir = output_dir or (
            Path(__file__).parent.parent / "results"
        )
        self.max_concurrent = max_concurrent
        self.verbose = False
        self._verbose_file = None
        self._verbose_header_written = False

    @property
    def transform_name(self) -> str:
        """Human-readable name for the current configuration."""
        return self._transform_name

    @property
    def prompt_tag(self) -> str:
        """Tag reflecting which prompt configuration is active."""
        if self.control_system_prompt == self.treatment_system_prompt:
            if self.treatment_system_prompt == TREATMENT_SYSTEM_PROMPT:
                return "treat"
            return "ctrl"
        return "split"

    def _apply_transform(self, text: str) -> tuple[str, float, dict]:
        """
        Apply the prompt transform (if any) to the input text.

        Returns:
            (transformed_text, latency_ms, metadata)
        """
        if self.prompt_transform is None:
            return text, 0.0, {"transform": "none"}

        start = time.perf_counter()
        original_tokens = estimate_tokens(text)
        transformed = self.prompt_transform(text)
        latency_ms = (time.perf_counter() - start) * 1000

        metadata = {
            "transform": self._transform_name,
            "original_tokens": original_tokens,
            "transformed_tokens": estimate_tokens(transformed),
        }
        return transformed, latency_ms, metadata

    async def _call_llm(
        self,
        prompt: str,
        system: str,
    ) -> tuple[str, float, int, int]:
        """
        Call Claude API with temperature=0 for deterministic generation.

        Returns:
            (response_text, latency_ms, input_tokens, output_tokens)
        """
        import anthropic

        client = anthropic.AsyncAnthropic()
        start = time.perf_counter()

        response = await client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=0,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )

        latency_ms = (time.perf_counter() - start) * 1000
        text = response.content[0].text
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens

        return text, latency_ms, input_tokens, output_tokens

    def _execute_code(
        self,
        generated_code: str,
        problem: BenchmarkProblem,
    ) -> tuple[bool, str, float]:
        """
        Execute generated code against benchmark tests.

        Returns:
            (passed, error_message, execution_time_ms)
        """
        if problem.benchmark == "humaneval":
            test_script = HumanEvalLoader.format_test_harness(generated_code, problem)
        else:
            test_script = MBPPLoader.format_test_harness(generated_code, problem)

        start = time.perf_counter()
        temp_path = None

        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".py", delete=False, encoding="utf-8"
            ) as f:
                f.write(test_script)
                temp_path = f.name

            result = subprocess.run(
                [sys.executable, temp_path],
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                cwd=tempfile.gettempdir(),
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            )

            exec_ms = (time.perf_counter() - start) * 1000

            if result.returncode == 0:
                return True, "", exec_ms
            else:
                error = result.stderr.strip() or result.stdout.strip() or "Unknown error"
                return False, error[:500], exec_ms

        except subprocess.TimeoutExpired:
            return False, f"Timeout after {self.timeout_seconds}s", self.timeout_seconds * 1000
        except Exception as e:
            return False, str(e)[:500], (time.perf_counter() - start) * 1000
        finally:
            try:
                if temp_path is not None:
                    os.unlink(temp_path)
            except (FileNotFoundError, OSError):
                pass

    def _write_verbose(
        self,
        problem: BenchmarkProblem,
        original_prompt: str,
        transformed_prompt: str,
        transform_meta: dict,
        ctrl_code: str,
        treat_code: str,
        ctrl_passed: bool,
        treat_passed: bool,
        ctrl_error: str,
        treat_error: str,
        ctrl_in: int,
        ctrl_out: int,
        treat_in: int,
        treat_out: int,
    ) -> None:
        """Write detailed per-problem verbose output to log file."""
        f = self._verbose_file
        if f is None:
            return

        sep = "-" * 65

        def w(line: str = "") -> None:
            f.write(line + "\n")

        w(f"\n{sep}")
        w(f"  {problem.task_id} ({problem.entry_point})")
        w(sep)

        # System prompts (once, for first problem)
        if not self._verbose_header_written:
            self._verbose_header_written = True
            w(f"\n  [Control System Prompt]")
            w(f"    {self.control_system_prompt}")
            w(f"\n  [Treatment System Prompt]")
            w(f"    {self.treatment_system_prompt}")

        # Prompt section
        w(f"\n  [Original Prompt] ({estimate_tokens(original_prompt, self.model)} est. tokens)")
        for line in original_prompt.strip().splitlines():
            w(f"    {line}")

        w(f"\n  [Transformed Prompt] ({estimate_tokens(transformed_prompt, self.model)} est. tokens)")
        for line in transformed_prompt.strip().splitlines():
            w(f"    {line}")

        # Control output
        ctrl_status = "PASS" if ctrl_passed else "FAIL"
        w(f"\n  [Control Output] {ctrl_status} "
          f"(in={ctrl_in}, out={ctrl_out})")
        for line in ctrl_code.strip().splitlines():
            w(f"    {line}")
        if ctrl_error:
            w(f"  [Control Error] {ctrl_error}")

        # Treatment output
        treat_status = "PASS" if treat_passed else "FAIL"
        w(f"\n  [Treatment Output] {treat_status} "
          f"(in={treat_in}, out={treat_out})")
        for line in treat_code.strip().splitlines():
            w(f"    {line}")
        if treat_error:
            w(f"  [Treatment Error] {treat_error}")

        w()
        f.flush()

    async def run_problem(self, problem: BenchmarkProblem) -> Pass1ProblemResult:
        """Run a single problem with control and treatment conditions."""
        original_prompt = problem.prompt

        # --- Transform prompt for treatment ---
        transformed_prompt, transform_latency, transform_meta = self._apply_transform(original_prompt)
        compression_ratio = (
            estimate_tokens(transformed_prompt, self.model)
            / max(1, estimate_tokens(original_prompt, self.model))
        )

        # --- Control: original prompt, control system prompt ---
        try:
            ctrl_response, ctrl_latency, ctrl_in, ctrl_out = await self._call_llm(
                original_prompt, self.control_system_prompt
            )
            ctrl_code = sanitize_code(ctrl_response, problem.entry_point)
        except Exception as e:
            logger.error(f"Control LLM call failed for {problem.task_id}: {e}")
            ctrl_code = ""
            ctrl_latency, ctrl_in, ctrl_out = 0.0, 0, 0

        # --- Treatment: transformed prompt, treatment system prompt ---
        try:
            treat_response, treat_latency, treat_in, treat_out = await self._call_llm(
                transformed_prompt, self.treatment_system_prompt
            )
            treat_code = sanitize_code(treat_response, problem.entry_point)
        except Exception as e:
            logger.error(f"Treatment LLM call failed for {problem.task_id}: {e}")
            treat_code = ""
            treat_latency, treat_in, treat_out = 0.0, 0, 0

        # --- Execute both ---
        ctrl_passed, ctrl_error, ctrl_exec = self._execute_code(ctrl_code, problem)
        treat_passed, treat_error, treat_exec = self._execute_code(treat_code, problem)

        # --- Verbose output ---
        if self.verbose and self._verbose_file:
            self._write_verbose(
                problem, original_prompt, transformed_prompt, transform_meta,
                ctrl_code, treat_code, ctrl_passed, treat_passed,
                ctrl_error, treat_error, ctrl_in, ctrl_out, treat_in, treat_out,
            )

        return Pass1ProblemResult(
            task_id=problem.task_id,
            benchmark=problem.benchmark,
            entry_point=problem.entry_point,
            control_passed=ctrl_passed,
            control_code=ctrl_code,
            control_error=ctrl_error,
            control_input_tokens=ctrl_in,
            control_output_tokens=ctrl_out,
            control_latency_ms=ctrl_latency,
            treatment_passed=treat_passed,
            treatment_code=treat_code,
            treatment_error=treat_error,
            treatment_input_tokens=treat_in,
            treatment_output_tokens=treat_out,
            treatment_latency_ms=treat_latency,
            original_prompt=original_prompt,
            transformed_prompt=transformed_prompt,
            compression_ratio=compression_ratio,
            compression_latency_ms=transform_latency,
            transform_name=self.transform_name,
            control_execution_ms=ctrl_exec,
            treatment_execution_ms=treat_exec,
        )

    async def run(
        self,
        n_problems: Optional[int] = None,
        max_concurrent: Optional[int] = None,
        benchmark: str = "humaneval",
        progress_callback=None,
    ) -> Pass1Summary:
        """
        Run the full pass@1 evaluation.

        Args:
            n_problems: Number of problems (None = all available)
            max_concurrent: Max concurrent API calls
            benchmark: "humaneval" or "mbpp"
            progress_callback: Optional callback(current, total)

        Returns:
            Pass1Summary with Wilson CIs and McNemar test
        """
        start_time = time.perf_counter()

        # Load problems
        loader = HumanEvalLoader() if benchmark == "humaneval" else MBPPLoader()
        problems = loader.load(n=n_problems)
        if not problems:
            logger.warning("No problems loaded")
            return self._empty_summary(benchmark)

        logger.info(
            f"Running pass@1 evaluation: {len(problems)} {benchmark} problems, "
            f"transform={self.transform_name}"
        )

        # Prepare JSONL output
        self.output_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        jsonl_path = self.output_dir / f"pass1_{self.transform_name}_{self.prompt_tag}_{ts}.jsonl"

        # Open verbose log file if enabled
        verbose_path = None
        if self.verbose:
            verbose_path = jsonl_path.with_suffix(".verbose.log")
            self._verbose_file = open(verbose_path, "w", encoding="utf-8")
            self._verbose_header_written = False
            self._verbose_file.write(
                f"WARNING: This log may contain prompt content and generated code.\n"
                f"Do not commit to public repositories without review.\n\n"
                f"Pass@1 Verbose Log\n"
                f"Transform: {self.transform_name}\n"
                f"Model: {self.model}\n"
                f"Benchmark: {benchmark}  |  Problems: {len(problems)}\n"
                f"{'=' * 65}\n"
            )

        # Run with concurrency control
        semaphore = asyncio.Semaphore(max_concurrent or self.max_concurrent)
        jsonl_lock = asyncio.Lock()
        completed = 0
        results: list[Pass1ProblemResult] = []

        async def run_one(problem: BenchmarkProblem) -> Pass1ProblemResult:
            nonlocal completed
            async with semaphore:
                result = await self.run_problem(problem)
                completed += 1
                if progress_callback:
                    progress_callback(completed, len(problems))
                try:
                    async with jsonl_lock:
                        with open(jsonl_path, "a", encoding="utf-8") as f:
                            f.write(result.to_jsonl_line() + "\n")
                except Exception as e:
                    logger.warning(f"Failed to write JSONL: {e}")
                return result

        try:
            tasks = [run_one(p) for p in problems]
            results = list(await asyncio.gather(*tasks))

            runtime = time.perf_counter() - start_time

            # Build summary with statistics
            summary = self._build_summary(results, benchmark, runtime)

            # Write verbose log footer
            if self._verbose_file:
                self._verbose_file.write(f"\n{'=' * 65}\n")
                self._verbose_file.write(summary.format_report())
                self._verbose_file.write("\n")

            # Write summary JSON alongside JSONL
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
    ) -> Pass1Summary:
        """Synchronous wrapper for run()."""
        return asyncio.run(
            self.run(
                n_problems=n_problems,
                benchmark=benchmark,
                progress_callback=progress_callback,
            )
        )

    def _build_summary(
        self,
        results: list[Pass1ProblemResult],
        benchmark: str,
        runtime: float,
    ) -> Pass1Summary:
        """Build Pass1Summary with Wilson CIs and McNemar test."""
        from itereval.statistics import wilson_score_interval, mcnemar_test

        n = len(results)
        if n == 0:
            return self._empty_summary(benchmark)

        # Contingency table
        both_passed = sum(1 for r in results if r.control_passed and r.treatment_passed)
        control_only = sum(1 for r in results if r.control_passed and not r.treatment_passed)
        treatment_only = sum(1 for r in results if not r.control_passed and r.treatment_passed)
        both_failed = sum(1 for r in results if not r.control_passed and not r.treatment_passed)

        ctrl_passed = both_passed + control_only
        treat_passed = both_passed + treatment_only

        # Wilson score CIs
        ctrl_wilson = wilson_score_interval(ctrl_passed, n)
        treat_wilson = wilson_score_interval(treat_passed, n)

        # McNemar's test
        mcnemar = mcnemar_test(both_passed, control_only, treatment_only, both_failed)

        # Token breakdown (4 buckets)
        ctrl_in = sum(r.control_input_tokens for r in results)
        ctrl_out = sum(r.control_output_tokens for r in results)
        treat_in = sum(r.treatment_input_tokens for r in results)
        treat_out = sum(r.treatment_output_tokens for r in results)

        total_ctrl = ctrl_in + ctrl_out
        total_treat = treat_in + treat_out

        def _pct_savings(control: int, treatment: int) -> float:
            return (1 - treatment / max(1, control)) * 100 if control > 0 else 0.0

        # Cost breakdown
        pricing = get_model_pricing(self.model)
        ctrl_in_cost = ctrl_in * pricing["input"] / 1_000_000
        ctrl_out_cost = ctrl_out * pricing["output"] / 1_000_000
        treat_in_cost = treat_in * pricing["input"] / 1_000_000
        treat_out_cost = treat_out * pricing["output"] / 1_000_000
        ctrl_total_cost = ctrl_in_cost + ctrl_out_cost
        treat_total_cost = treat_in_cost + treat_out_cost

        avg_comp_ratio = sum(r.compression_ratio for r in results) / n

        return Pass1Summary(
            benchmark=benchmark,
            transform_name=self.transform_name,
            model=self.model,
            prompt_config=self.prompt_tag,
            total_problems=n,
            control_passed=ctrl_passed,
            treatment_passed=treat_passed,
            both_passed=both_passed,
            control_only=control_only,
            treatment_only=treatment_only,
            both_failed=both_failed,
            control_pass_rate=ctrl_wilson.proportion,
            control_ci_lower=ctrl_wilson.ci_lower,
            control_ci_upper=ctrl_wilson.ci_upper,
            treatment_pass_rate=treat_wilson.proportion,
            treatment_ci_lower=treat_wilson.ci_lower,
            treatment_ci_upper=treat_wilson.ci_upper,
            mcnemar_statistic=mcnemar.statistic,
            mcnemar_p_value=mcnemar.p_value,
            mcnemar_method=mcnemar.method,
            mcnemar_significant=mcnemar.is_significant,
            control_input_tokens=ctrl_in,
            control_output_tokens=ctrl_out,
            treatment_input_tokens=treat_in,
            treatment_output_tokens=treat_out,
            avg_compression_ratio=avg_comp_ratio,
            total_control_tokens=total_ctrl,
            total_treatment_tokens=total_treat,
            input_token_savings_pct=_pct_savings(ctrl_in, treat_in),
            output_token_savings_pct=_pct_savings(ctrl_out, treat_out),
            total_token_savings_pct=_pct_savings(total_ctrl, total_treat),
            control_input_cost=ctrl_in_cost,
            control_output_cost=ctrl_out_cost,
            control_total_cost=ctrl_total_cost,
            treatment_input_cost=treat_in_cost,
            treatment_output_cost=treat_out_cost,
            treatment_total_cost=treat_total_cost,
            total_cost_savings_pct=_pct_savings(
                round(ctrl_total_cost * 1_000_000),
                round(treat_total_cost * 1_000_000),
            ),
            total_runtime_seconds=runtime,
            results=results,
        )

    def _empty_summary(self, benchmark: str) -> Pass1Summary:
        return Pass1Summary(
            benchmark=benchmark,
            transform_name=self.transform_name,
            model=self.model,
            prompt_config=self.prompt_tag,
            total_problems=0,
            control_passed=0,
            treatment_passed=0,
            both_passed=0,
            control_only=0,
            treatment_only=0,
            both_failed=0,
        )

    def get_config(self) -> dict:
        """Return current configuration as a dict (useful for --dry-run)."""
        return {
            "transform_name": self.transform_name,
            "has_prompt_transform": self.prompt_transform is not None,
            "model": self.model,
            "max_tokens": self.max_tokens,
            "timeout_seconds": self.timeout_seconds,
            "output_dir": str(self.output_dir),
            "control_system_prompt": self.control_system_prompt,
            "treatment_system_prompt": self.treatment_system_prompt,
            "prompt_config": self.prompt_tag,
        }
