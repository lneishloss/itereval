#!/usr/bin/env python3
"""
CLI for shared-control A/B sweep experiments.

Runs multiple experiments that share a single control arm, eliminating
redundant API calls. For example, isolating 2 IVs (prompt, minification)
against 164 HumanEval problems would normally require 2 x 164 = 328
control calls. With shared control, only 164 are needed.

Usage:
    # Pass@1 sweep with 20 problems
    python scripts/run_ab_sweep.py --preset isolate-ivs -n 20

    # Full HumanEval
    python scripts/run_ab_sweep.py --preset isolate-ivs --all

    # Iterative sweep
    python scripts/run_ab_sweep.py --preset isolate-ivs -n 20 --iterative --max-attempts 3

    # Dry run (show config, no API calls)
    python scripts/run_ab_sweep.py --preset isolate-ivs --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Callable

# Ensure project root is on path
_script_dir = Path(__file__).parent.resolve()
_project_root = _script_dir.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from itereval.benchmarks import (
    BenchmarkProblem,
    HumanEvalLoader,
    MBPPLoader,
    Pass1Runner,
    IterativeRunner,
)
from itereval.benchmarks.pass1_runner import (
    CONTROL_SYSTEM_PROMPT,
    TREATMENT_SYSTEM_PROMPT,
    Pass1ProblemResult,
    sanitize_code,
)
from itereval.benchmarks.iterative_runner import (
    IterativeProblemResult,
    AttemptResult,
    ERROR_FEEDBACK_TEMPLATE,
)
from itereval.utils import NumpyEncoder, calculate_cost, estimate_tokens
from itereval.transforms import strip_whitespace, minify_prompt, minify_python


# ---------------------------------------------------------------------------
# Experiment dataclass
# ---------------------------------------------------------------------------

@dataclass
class Experiment:
    """One arm of a sweep experiment."""
    name: str
    desc: str
    system_prompt: str
    prompt_transform: Optional[Callable[[str], str]]
    transform_name: str
    code_transform: Optional[Callable[[str], str]] = None


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------

def _get_presets() -> dict:
    """Build experiment presets. Deferred to avoid import-time side effects."""
    return {
        "isolate-ivs": {
            "description": "Isolate each independent variable against a shared control",
            "experiments": [
                Experiment(
                    name="prompt_only",
                    desc="Conciseness instruction effect",
                    system_prompt=TREATMENT_SYSTEM_PROMPT,
                    prompt_transform=None,
                    transform_name="prompt_only",
                ),
                Experiment(
                    name="minification",
                    desc="Prompt minification (preserve docstring) + code minification in error feedback",
                    system_prompt=CONTROL_SYSTEM_PROMPT,
                    prompt_transform=minify_prompt,
                    transform_name="minification",
                    code_transform=minify_python,
                ),
            ],
        },
    }


# ---------------------------------------------------------------------------
# Single-arm helpers
# ---------------------------------------------------------------------------

async def _run_single_arm_pass1(
    runner: Pass1Runner,
    problem: BenchmarkProblem,
    system_prompt: str,
    transform_fn: Optional[Callable[[str], str]] = None,
) -> dict:
    """Run a single pass@1 arm, returning a result dict."""
    prompt = problem.prompt
    transformed_prompt = transform_fn(prompt) if transform_fn else prompt

    try:
        resp_text, latency, in_tok, out_tok = await runner._call_llm(
            transformed_prompt, system_prompt
        )
        code = sanitize_code(resp_text, problem.entry_point)
    except Exception:
        code = ""
        latency, in_tok, out_tok = 0.0, 0, 0

    passed, error, exec_ms = runner._execute_code(code, problem)

    return {
        "passed": passed,
        "code": code,
        "error": error,
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "latency_ms": latency,
        "execution_ms": exec_ms,
        "prompt": prompt,
        "transformed_prompt": transformed_prompt,
    }


async def _run_single_arm_iterative(
    runner: Pass1Runner,
    problem: BenchmarkProblem,
    system_prompt: str,
    max_attempts: int,
    transform_fn: Optional[Callable[[str], str]] = None,
    code_transform_fn: Optional[Callable[[str], str]] = None,
    verbose: bool = False,
) -> dict:
    """Run a single iterative arm with retry loop, returning a result dict.

    Args:
        code_transform_fn: Optional transform applied to generated code before
            embedding in error feedback. Reduces token growth on retries.
        verbose: If True, include per-attempt details in the result for logging.
    """
    original_prompt = problem.prompt
    solved = False
    solved_at: Optional[int] = None
    cumulative_in = 0
    cumulative_out = 0
    current_prompt = original_prompt
    attempts: list[dict] = []

    for attempt_num in range(1, max_attempts + 1):
        # Apply prompt transform to the current prompt (original + any error feedback)
        prompt_to_send = transform_fn(current_prompt) if transform_fn else current_prompt

        try:
            resp_text, latency, in_tok, out_tok = await runner._call_llm(
                prompt_to_send, system_prompt
            )
            code = sanitize_code(resp_text, problem.entry_point)
        except Exception:
            code = ""
            latency, in_tok, out_tok = 0.0, 0, 0

        passed, error, exec_ms = runner._execute_code(code, problem)

        cumulative_in += in_tok
        cumulative_out += out_tok

        attempts.append({
            "attempt": attempt_num,
            "passed": passed,
            "code": code,
            "error": error,
            "input_tokens": in_tok,
            "output_tokens": out_tok,
            "cumulative_in": cumulative_in,
            "cumulative_out": cumulative_out,
            "latency_ms": latency,
            "execution_ms": exec_ms,
            "prompt_sent": prompt_to_send if verbose else "",
        })

        if passed:
            solved = True
            solved_at = attempt_num
            break

        # Optionally minify code before embedding in error feedback
        feedback_code = code_transform_fn(code) if code_transform_fn and code else code

        # Build next prompt: original + error feedback, then transform on next iter
        current_prompt = original_prompt + ERROR_FEEDBACK_TEMPLATE.format(
            error=error, code=feedback_code
        )

    total_cost = calculate_cost(cumulative_in, cumulative_out, runner.model)

    return {
        "solved": solved,
        "solved_at": solved_at,
        "total_input_tokens": cumulative_in,
        "total_output_tokens": cumulative_out,
        "total_cost": total_cost,
        "attempts": attempts,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Shared-Control A/B Sweep",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --preset isolate-ivs -n 20
  %(prog)s --preset isolate-ivs --all
  %(prog)s --preset isolate-ivs -n 20 --iterative --max-attempts 3
  %(prog)s --preset isolate-ivs --dry-run
        """,
    )

    parser.add_argument(
        "--preset", required=True,
        help="Experiment preset name (e.g. isolate-ivs)",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Run all available problems (164 for HumanEval, 427 for MBPP)",
    )
    parser.add_argument(
        "-n", type=int, default=20,
        help="Number of problems to run (default: 20, ignored if --all)",
    )
    parser.add_argument(
        "--benchmark", choices=["humaneval", "mbpp"], default="humaneval",
        help="Benchmark to run (default: humaneval)",
    )
    parser.add_argument(
        "--model", default="claude-sonnet-4-20250514",
        help="Model to use (default: claude-sonnet-4-20250514)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Output directory (default: itereval/results/)",
    )
    parser.add_argument(
        "--max-concurrent", type=int, default=2,
        help="Max concurrent API calls (default: 2)",
    )
    parser.add_argument(
        "--iterative", action="store_true",
        help="Run iterative cost-per-solve mode instead of pass@1",
    )
    parser.add_argument(
        "--max-attempts", type=int, default=5,
        help="Max retry attempts per problem in iterative mode (default: 5)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show configuration without making API calls",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Write detailed per-problem verbose logs",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Attempt dict -> AttemptResult conversion
# ---------------------------------------------------------------------------

def _dicts_to_attempt_results(attempt_dicts: list[dict]) -> list[AttemptResult]:
    """Convert attempt dicts from _run_single_arm_iterative to AttemptResult objects."""
    return [
        AttemptResult(
            attempt_number=a["attempt"],
            code=a.get("code", ""),
            passed=a["passed"],
            error=a.get("error", ""),
            input_tokens=a["input_tokens"],
            output_tokens=a["output_tokens"],
            latency_ms=a.get("latency_ms", 0.0),
            execution_ms=a.get("execution_ms", 0.0),
            cumulative_input_tokens=a["cumulative_in"],
            cumulative_output_tokens=a["cumulative_out"],
            prompt_sent=a.get("prompt_sent", ""),
        )
        for a in attempt_dicts
    ]


# ---------------------------------------------------------------------------
# Verbose logging
# ---------------------------------------------------------------------------

def _open_verbose_log(
    output_dir: Path,
    prefix: str,
    exp: Experiment,
    ts: str,
    model: str,
    mode: str,
    max_attempts: int = 0,
) -> tuple:
    """Open a verbose log file for an experiment. Returns (file, path)."""
    path = output_dir / f"{prefix}_{exp.name}_{ts}.verbose.log"
    f = open(path, "w", encoding="utf-8")
    f.write(
        f"WARNING: This log contains prompt content and generated code.\n"
        f"Do not commit to public repositories without review.\n\n"
        f"A/B Sweep Verbose Log — {exp.name}\n"
        f"Description: {exp.desc}\n"
        f"Model: {model}  |  Mode: {mode}\n"
    )
    if max_attempts:
        f.write(f"Max Attempts: {max_attempts}\n")
    prompt_label = "TREATMENT" if exp.system_prompt == TREATMENT_SYSTEM_PROMPT else "CONTROL"
    f.write(
        f"System Prompt: {prompt_label}\n"
        f"Prompt Transform: {exp.transform_name if exp.prompt_transform else 'none'}\n"
        f"Code Transform: {'minify_python' if exp.code_transform else 'none'}\n"
        f"{'=' * 65}\n"
    )
    return f, path


def _write_verbose_pass1(
    f,
    problem: BenchmarkProblem,
    ctrl_result: dict,
    treat_result: dict,
    model: str,
) -> None:
    """Write a single pass@1 problem to the verbose log."""
    sep = "-" * 65

    def w(line: str = "") -> None:
        f.write(line + "\n")

    w(f"\n{sep}")
    w(f"  {problem.task_id} ({problem.entry_point})")
    w(sep)

    w(f"\n  [Original Prompt] ({estimate_tokens(problem.prompt, model)} est. tokens)")
    for line in problem.prompt.strip().splitlines():
        w(f"    {line}")

    if treat_result["transformed_prompt"] != problem.prompt:
        w(f"\n  [Transformed Prompt] ({estimate_tokens(treat_result['transformed_prompt'], model)} est. tokens)")
        for line in treat_result["transformed_prompt"].strip().splitlines():
            w(f"    {line}")

    for arm_name, result in [("Control", ctrl_result), ("Treatment", treat_result)]:
        status = "PASS" if result["passed"] else "FAIL"
        w(f"\n  [{arm_name}] {status}  (in={result['input_tokens']}, out={result['output_tokens']})")
        if result["code"]:
            for line in result["code"].strip().splitlines():
                w(f"    {line}")
        if result["error"]:
            w(f"  [{arm_name} Error] {result['error'][:300]}")

    w()
    f.flush()


def _write_verbose_iterative(
    f,
    problem: BenchmarkProblem,
    ctrl_result: dict,
    treat_result: dict,
    model: str,
) -> None:
    """Write a single iterative problem to the verbose log."""
    sep = "-" * 65

    def w(line: str = "") -> None:
        f.write(line + "\n")

    w(f"\n{sep}")
    w(f"  {problem.task_id} ({problem.entry_point})")
    w(sep)

    for arm_name, result in [("Control", ctrl_result), ("Treatment", treat_result)]:
        status = f"SOLVED at attempt {result['solved_at']}" if result["solved"] else "UNSOLVED"
        w(f"\n  [{arm_name}] {status}  "
          f"(total in={result['total_input_tokens']}, out={result['total_output_tokens']}, "
          f"cost=${result['total_cost']:.6f})")

        for a in result.get("attempts", []):
            a_status = "PASS" if a["passed"] else "FAIL"
            w(f"\n    --- Attempt {a['attempt']} [{a_status}] ---")
            w(f"    Input tokens: {a['input_tokens']}  |  Output tokens: {a['output_tokens']}")
            w(f"    Cumulative: in={a['cumulative_in']}, out={a['cumulative_out']}")

            if a["prompt_sent"]:
                tok = estimate_tokens(a["prompt_sent"], model)
                w(f"\n    [Prompt Sent] ({tok} est. tokens)")
                for line in a["prompt_sent"].strip().splitlines():
                    w(f"      {line}")

            if a["code"]:
                w(f"\n    [Generated Code]")
                for line in a["code"].strip().splitlines():
                    w(f"      {line}")

            if a["error"]:
                w(f"\n    [Error Output]")
                for line in a["error"][:500].strip().splitlines():
                    w(f"      {line}")

    w()
    f.flush()


# ---------------------------------------------------------------------------
# Pass@1 sweep
# ---------------------------------------------------------------------------

async def run_pass1_sweep(
    experiments: list[Experiment],
    problems: list[BenchmarkProblem],
    model: str,
    max_concurrent: int,
    output_dir: Path,
    verbose: bool = False,
) -> dict:
    """Run shared-control pass@1 sweep. Returns comparison dict."""
    # We need a runner for _call_llm and _execute_code
    runner = Pass1Runner(model=model, max_concurrent=max_concurrent, output_dir=output_dir)

    # --- Run control arm once ---
    print(f"\n  Running shared control arm ({len(problems)} problems)...")
    semaphore = asyncio.Semaphore(max_concurrent)
    completed = 0

    async def run_ctrl(problem):
        nonlocal completed
        async with semaphore:
            result = await _run_single_arm_pass1(
                runner, problem, CONTROL_SYSTEM_PROMPT, transform_fn=None
            )
            completed += 1
            print(f"    Control [{completed}/{len(problems)}]", end="\r", flush=True)
            return result

    control_results = await asyncio.gather(*[run_ctrl(p) for p in problems])
    print(f"    Control [{len(problems)}/{len(problems)}] done.")

    # --- Run each experiment's treatment arm ---
    all_summaries = []
    ts = time.strftime("%Y%m%d_%H%M%S")

    for exp in experiments:
        print(f"\n  Experiment: {exp.name} ({exp.desc})")

        # Build a runner configured for this experiment (for _build_summary)
        treat_runner = Pass1Runner(
            prompt_transform=exp.prompt_transform,
            transform_name=exp.transform_name,
            system_prompt=exp.system_prompt,
            control_system_prompt=CONTROL_SYSTEM_PROMPT,
            model=model,
            max_concurrent=max_concurrent,
            output_dir=output_dir,
        )

        completed = 0

        async def run_treat(problem, ctrl_result, _exp=exp):
            nonlocal completed

            async with semaphore:
                result = await _run_single_arm_pass1(
                    runner, problem, _exp.system_prompt,
                    transform_fn=_exp.prompt_transform,
                )
                completed += 1
                print(
                    f"    Treatment [{completed}/{len(problems)}]",
                    end="\r", flush=True,
                )


            original_tokens = max(1, estimate_tokens(problem.prompt, model))
            transformed_tokens = estimate_tokens(result["transformed_prompt"], model)
            compression_ratio = transformed_tokens / original_tokens

            combined = Pass1ProblemResult(
                task_id=problem.task_id,
                benchmark=problem.benchmark,
                entry_point=problem.entry_point,
                control_passed=ctrl_result["passed"],
                control_code=ctrl_result["code"],
                control_error=ctrl_result["error"],
                control_input_tokens=ctrl_result["input_tokens"],
                control_output_tokens=ctrl_result["output_tokens"],
                control_latency_ms=ctrl_result["latency_ms"],
                treatment_passed=result["passed"],
                treatment_code=result["code"],
                treatment_error=result["error"],
                treatment_input_tokens=result["input_tokens"],
                treatment_output_tokens=result["output_tokens"],
                treatment_latency_ms=result["latency_ms"],
                original_prompt=problem.prompt,
                transformed_prompt=result["transformed_prompt"],
                compression_ratio=compression_ratio,
                compression_latency_ms=0.0,
                transform_name=_exp.transform_name,
                control_execution_ms=ctrl_result["execution_ms"],
                treatment_execution_ms=result["execution_ms"],
            )
            return combined

        combined_results = await asyncio.gather(*[
            run_treat(p, cr) for p, cr in zip(problems, control_results)
        ])
        combined_results = list(combined_results)
        print(f"    Treatment [{len(problems)}/{len(problems)}] done.")

        # Build summary using the runner's existing method
        summary = treat_runner._build_summary(combined_results, problems[0].benchmark, 0.0)
        all_summaries.append((exp, summary))

        # Write JSONL + JSON
        jsonl_path = output_dir / f"sweep_{exp.name}_{ts}.jsonl"
        summary_path = jsonl_path.with_suffix(".json")
        try:
            with open(jsonl_path, "w", encoding="utf-8") as f:
                for r in combined_results:
                    f.write(r.to_jsonl_line() + "\n")
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary.to_dict(), f, indent=2, cls=NumpyEncoder)
        except Exception as e:
            print(f"    Warning: Failed to write results: {e}")

        # Write verbose log
        if verbose:
            vf, vpath = _open_verbose_log(
                output_dir, "sweep", exp, ts, model, "pass@1"
            )
            for p, cr, tr in zip(problems, control_results, combined_results):
                # Reconstruct treatment result dict from Pass1ProblemResult
                treat_dict = {
                    "passed": tr.treatment_passed,
                    "code": tr.treatment_code,
                    "error": tr.treatment_error,
                    "input_tokens": tr.treatment_input_tokens,
                    "output_tokens": tr.treatment_output_tokens,
                    "transformed_prompt": tr.transformed_prompt,
                }
                _write_verbose_pass1(vf, p, cr, treat_dict, model)
            vf.write(f"\n{'=' * 65}\n")
            vf.write(summary.format_report())
            vf.write("\n")
            vf.close()
            print(f"    Verbose: {vpath}")

        print(summary.format_report())

    return _format_comparison(experiments, all_summaries, problems, model, ts, output_dir)


# ---------------------------------------------------------------------------
# Iterative sweep
# ---------------------------------------------------------------------------

async def run_iterative_sweep(
    experiments: list[Experiment],
    problems: list[BenchmarkProblem],
    model: str,
    max_concurrent: int,
    max_attempts: int,
    output_dir: Path,
    verbose: bool = False,
) -> dict:
    """Run shared-control iterative sweep. Returns comparison dict."""
    # Runner for _call_llm and _execute_code
    runner = Pass1Runner(model=model, max_concurrent=max_concurrent, output_dir=output_dir)

    # --- Run control arm once (iterative) ---
    print(f"\n  Running shared control arm (iterative, {len(problems)} problems)...")
    semaphore = asyncio.Semaphore(max_concurrent)
    completed = 0

    async def run_ctrl(problem):
        nonlocal completed
        async with semaphore:
            result = await _run_single_arm_iterative(
                runner, problem, CONTROL_SYSTEM_PROMPT,
                max_attempts=max_attempts, transform_fn=None,
                verbose=verbose,
            )
            completed += 1
            print(f"    Control [{completed}/{len(problems)}]", end="\r", flush=True)
            return result

    control_results = await asyncio.gather(*[run_ctrl(p) for p in problems])
    print(f"    Control [{len(problems)}/{len(problems)}] done.")

    # --- Run each experiment's treatment arm ---
    all_summaries = []
    ts = time.strftime("%Y%m%d_%H%M%S")

    for exp in experiments:
        print(f"\n  Experiment: {exp.name} ({exp.desc})")

        # Build a runner configured for this experiment (for _build_iterative_summary)
        treat_runner = IterativeRunner(
            prompt_transform=exp.prompt_transform,
            transform_name=exp.transform_name,
            system_prompt=exp.system_prompt,
            control_system_prompt=CONTROL_SYSTEM_PROMPT,
            model=model,
            max_concurrent=max_concurrent,
            output_dir=output_dir,
            max_attempts=max_attempts,
        )

        completed = 0

        async def run_treat(problem, ctrl_result, _exp=exp):
            nonlocal completed

            async with semaphore:
                result = await _run_single_arm_iterative(
                    runner, problem, _exp.system_prompt,
                    max_attempts=max_attempts,
                    transform_fn=_exp.prompt_transform,
                    code_transform_fn=_exp.code_transform,
                    verbose=verbose,
                )
                completed += 1
                print(
                    f"    Treatment [{completed}/{len(problems)}]",
                    end="\r", flush=True,
                )


            compression_ratio = 1.0
            if _exp.prompt_transform:
                transformed = _exp.prompt_transform(problem.prompt)
                original_tokens = max(1, estimate_tokens(problem.prompt, model))
                compression_ratio = estimate_tokens(transformed, model) / original_tokens

            combined = IterativeProblemResult(
                task_id=problem.task_id,
                benchmark=problem.benchmark,
                entry_point=problem.entry_point,
                control_solved=ctrl_result["solved"],
                control_attempts=_dicts_to_attempt_results(ctrl_result["attempts"]),
                control_solved_at=ctrl_result["solved_at"],
                control_total_input_tokens=ctrl_result["total_input_tokens"],
                control_total_output_tokens=ctrl_result["total_output_tokens"],
                control_total_cost=ctrl_result["total_cost"],
                treatment_solved=result["solved"],
                treatment_attempts=_dicts_to_attempt_results(result["attempts"]),
                treatment_solved_at=result["solved_at"],
                treatment_total_input_tokens=result["total_input_tokens"],
                treatment_total_output_tokens=result["total_output_tokens"],
                treatment_total_cost=result["total_cost"],
                compression_ratio=compression_ratio,
                transform_name=_exp.transform_name,
            )
            return combined, result

        raw_results = await asyncio.gather(*[
            run_treat(p, cr) for p, cr in zip(problems, control_results)
        ])
        combined_results = [r[0] for r in raw_results]
        treat_raw_results = [r[1] for r in raw_results]
        print(f"    Treatment [{len(problems)}/{len(problems)}] done.")

        summary = treat_runner._build_iterative_summary(
            combined_results, problems[0].benchmark, 0.0
        )
        all_summaries.append((exp, summary))

        # Write JSONL + JSON
        jsonl_path = output_dir / f"sweep_iter_{exp.name}_{ts}.jsonl"
        summary_path = jsonl_path.with_suffix(".json")
        try:
            with open(jsonl_path, "w", encoding="utf-8") as f:
                for r in combined_results:
                    f.write(r.to_jsonl_line() + "\n")
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary.to_dict(), f, indent=2, cls=NumpyEncoder)
        except Exception as e:
            print(f"    Warning: Failed to write results: {e}")

        # Write verbose log
        if verbose:
            vf, vpath = _open_verbose_log(
                output_dir, "sweep_iter", exp, ts, model,
                "iterative", max_attempts=max_attempts,
            )
            for p, cr, tr in zip(problems, control_results, treat_raw_results):
                _write_verbose_iterative(vf, p, cr, tr, model)
            vf.write(f"\n{'=' * 65}\n")
            vf.write(summary.format_report())
            vf.write("\n")
            vf.close()
            print(f"    Verbose: {vpath}")

        print(summary.format_report())

    return _format_comparison(
        experiments, all_summaries, problems, model, ts, output_dir,
        iterative=True,
    )


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------

def _format_comparison(
    experiments: list[Experiment],
    all_summaries: list,
    problems: list[BenchmarkProblem],
    model: str,
    ts: str,
    output_dir: Path,
    iterative: bool = False,
) -> dict:
    """Print and save comparison table."""
    n_problems = len(problems)
    n_experiments = len(experiments)
    n_control = n_problems
    n_treatment = n_problems * n_experiments
    n_total = n_control + n_treatment
    n_saved = n_problems * (n_experiments - 1)  # vs independent control per experiment

    print()
    print("=" * 75)
    print(f"  A/B Sweep | Shared control ({n_problems} problems)")
    print(f"  Model: {model}")
    print("=" * 75)

    if iterative:
        print(f"  {'Experiment':<18} {'Ctrl Solve':>11} {'Treat Solve':>12} "
              f"{'CPS Savings':>12}")
        print("-" * 75)
        for exp, summary in all_summaries:
            print(
                f"  {exp.name:<18} "
                f"{summary.control_solve_rate:>10.1%}  "
                f"{summary.treatment_solve_rate:>10.1%}  "
                f"{summary.cost_per_solve_savings_pct:>10.1f}%"
            )
    else:
        print(f"  {'Experiment':<18} {'Control':>10} {'Treatment':>11} "
              f"{'McNemar p':>11} {'Cost Savings':>13}")
        print("-" * 75)
        for exp, summary in all_summaries:
            print(
                f"  {exp.name:<18} "
                f"{summary.control_pass_rate:>9.1%}  "
                f"{summary.treatment_pass_rate:>9.1%}  "
                f"{summary.mcnemar_p_value:>10.4f}  "
                f"{summary.total_cost_savings_pct:>11.1f}%"
            )

    print("=" * 75)
    print(f"  API calls: {n_control} control + {n_treatment} treatment = {n_total} total")
    if n_saved > 0:
        print(f"  (saved {n_saved} calls vs. independent control per experiment)")
    print("=" * 75)

    # Write comparison JSON
    comparison = {
        "preset_description": all_summaries[0][1].transform_name if all_summaries else "",
        "model": model,
        "n_problems": n_problems,
        "n_experiments": n_experiments,
        "api_calls_total": n_total,
        "api_calls_saved": n_saved,
        "experiments": [],
    }
    for exp, summary in all_summaries:
        d = summary.to_dict()
        d["experiment_name"] = exp.name
        d["experiment_desc"] = exp.desc
        comparison["experiments"].append(d)

    comparison_path = output_dir / f"sweep_comparison_{ts}.json"
    try:
        with open(comparison_path, "w", encoding="utf-8") as f:
            json.dump(comparison, f, indent=2, cls=NumpyEncoder)
        print(f"\n  Comparison: {comparison_path}")
    except Exception as e:
        print(f"  Warning: Failed to write comparison: {e}")

    return comparison


# ---------------------------------------------------------------------------
# Dry run
# ---------------------------------------------------------------------------

def _print_dry_run(args, preset_config, n_problems):
    """Show configuration without making API calls."""
    experiments = preset_config["experiments"]

    print(f"\n{'=' * 65}")
    print(f"  DRY RUN -- A/B Sweep: {args.preset}")
    print(f"{'=' * 65}")
    print(f"\n  Description: {preset_config['description']}")
    print(f"  Model: {args.model}")
    print(f"  Benchmark: {args.benchmark}")
    print(f"  Problems: {'all' if args.all else args.n}")
    print(f"  Mode: {'iterative' if args.iterative else 'pass@1'}")
    if args.iterative:
        print(f"  Max attempts: {args.max_attempts}")

    print(f"\n  Control arm:")
    print(f"    System prompt: CONTROL (standard)")
    print(f"    Transform: none")

    print(f"\n  Experiments ({len(experiments)}):")
    for exp in experiments:
        prompt_label = "TREATMENT" if exp.system_prompt == TREATMENT_SYSTEM_PROMPT else "CONTROL"
        prompt_xform = exp.transform_name if exp.prompt_transform else "none"
        code_xform = "minify_python" if exp.code_transform else "none"
        print(f"    {exp.name}: {exp.desc}")
        print(f"      System prompt: {prompt_label}, Prompt transform: {prompt_xform}, Code transform: {code_xform}")

    n = n_problems or (164 if args.benchmark == "humaneval" else 427)
    n_exp = len(experiments)
    n_ctrl = n
    n_treat = n * n_exp
    n_total = n_ctrl + n_treat
    n_independent = n * 2 * n_exp
    n_saved = n_independent - n_total

    if args.iterative:
        print(f"\n  Note: iterative mode, each problem gets up to {args.max_attempts} attempts")
        n_ctrl_max = n * args.max_attempts
        n_treat_max = n * n_exp * args.max_attempts
        n_total_max = n_ctrl_max + n_treat_max
        n_independent_max = n * 2 * n_exp * args.max_attempts
        n_saved_max = n_independent_max - n_total_max
        print(f"  Max API calls: {n_ctrl_max} control + {n_treat_max} treatment = {n_total_max} total")
        print(f"  Independent runs would need: {n_independent_max} calls (max)")
        print(f"  Max savings: {n_saved_max} calls ({n_saved_max / max(1, n_independent_max) * 100:.0f}%)")
    else:
        print(f"\n  API calls: {n_ctrl} control + {n_treat} treatment = {n_total} total")
        print(f"  Independent runs would need: {n_independent} calls")
        print(f"  Savings: {n_saved} calls ({n_saved / max(1, n_independent) * 100:.0f}%)")

    print(f"{'=' * 65}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    presets = _get_presets()
    if args.preset not in presets:
        print(f"Unknown preset: {args.preset}")
        print(f"Available presets: {', '.join(presets.keys())}")
        sys.exit(1)

    preset_config = presets[args.preset]
    experiments = preset_config["experiments"]
    n_problems = None if args.all else args.n
    output_dir = args.output_dir or (Path(__file__).parent.parent / "itereval" / "results")
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        _print_dry_run(args, preset_config, n_problems)
        return

    # Load problems
    loader = HumanEvalLoader() if args.benchmark == "humaneval" else MBPPLoader()
    problems = loader.load(n=n_problems)
    if not problems:
        print("No problems loaded")
        sys.exit(1)

    print(f"\nLoaded {len(problems)} {args.benchmark} problems")

    if args.iterative:
        asyncio.run(run_iterative_sweep(
            experiments=experiments,
            problems=problems,
            model=args.model,
            max_concurrent=args.max_concurrent,
            max_attempts=args.max_attempts,
            output_dir=output_dir,
            verbose=args.verbose,
        ))
    else:
        asyncio.run(run_pass1_sweep(
            experiments=experiments,
            problems=problems,
            model=args.model,
            max_concurrent=args.max_concurrent,
            output_dir=output_dir,
            verbose=args.verbose,
        ))


if __name__ == "__main__":
    main()
