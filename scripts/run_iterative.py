#!/usr/bin/env python3
"""
CLI entry point for iterative cost-per-solve evaluation.

Usage:
    # Default: system prompt treatment only, 3 attempts
    python scripts/run_iterative.py -n 20

    # Custom max attempts and budget cap
    python scripts/run_iterative.py -n 10 --max-attempts 5 --budget-per-problem 0.05

    # Use same system prompt for both arms (no treatment)
    python scripts/run_iterative.py --same-prompt -n 20

    # Full HumanEval
    python scripts/run_iterative.py --all --max-attempts 3

    # Dry run (show config, no API calls)
    python scripts/run_iterative.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

# Ensure project root is on path
_script_dir = Path(__file__).parent.resolve()
_project_root = _script_dir.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Iterative Cost-Per-Solve Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s -n 20
  %(prog)s --all --max-attempts 3
  %(prog)s --same-prompt -n 20
  %(prog)s --dry-run
        """,
    )

    prompt_group = parser.add_mutually_exclusive_group()
    prompt_group.add_argument(
        "--same-prompt", action="store_true",
        help="Use the same system prompt for both arms (control prompt). "
             "This removes the prompt engineering IV, leaving only the "
             "prompt_transform (if any) as the treatment.",
    )
    prompt_group.add_argument(
        "--treatment-prompt", action="store_true",
        help="Use the treatment (compact-code) system prompt for both arms.",
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
        "--max-attempts", type=int, default=3,
        help="Max retry attempts per problem (default: 3)",
    )
    parser.add_argument(
        "--budget-per-problem", type=float, default=None,
        help="Optional USD cap per problem (default: unlimited)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show configuration without making API calls",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Show full per-problem detail: attempts, errors, generated code",
    )

    return parser.parse_args()


def resolve_prompts(args: argparse.Namespace) -> tuple[Optional[str], Optional[str]]:
    """Returns (treatment_system_prompt, control_system_prompt) overrides."""
    from itereval.benchmarks.pass1_runner import (
        CONTROL_SYSTEM_PROMPT,
        TREATMENT_SYSTEM_PROMPT,
    )

    if getattr(args, "same_prompt", False):
        return CONTROL_SYSTEM_PROMPT, None
    elif getattr(args, "treatment_prompt", False):
        return None, TREATMENT_SYSTEM_PROMPT
    else:
        return None, None


def main():
    args = parse_args()

    from itereval.benchmarks.iterative_runner import IterativeRunner

    treatment_override, control_override = resolve_prompts(args)
    n_problems = None if args.all else args.n

    if args.dry_run:
        print("\n=== DRY RUN -- Configuration ===\n")
        runner = IterativeRunner(
            system_prompt=treatment_override,
            control_system_prompt=control_override,
            model=args.model,
            output_dir=args.output_dir,
            max_attempts=args.max_attempts,
            budget_per_problem=args.budget_per_problem,
        )
        config = runner.get_config()
        print(json.dumps(config, indent=2))
        print()
        print(f"Benchmark: {args.benchmark}")
        print(f"Problems: {'all' if args.all else args.n}")
        print(f"Max attempts per problem: {args.max_attempts}")
        budget_str = f"${args.budget_per_problem:.4f}" if args.budget_per_problem else "unlimited"
        print(f"Budget per problem: {budget_str}")
        max_calls = (n_problems or 164) * 2 * args.max_attempts
        print(f"Max API calls: {max_calls}")
        return

    print(f"\nRunning iterative CPS evaluation (max {args.max_attempts} attempts)")

    runner = IterativeRunner(
        system_prompt=treatment_override,
        control_system_prompt=control_override,
        model=args.model,
        max_concurrent=args.max_concurrent,
        output_dir=args.output_dir,
        max_attempts=args.max_attempts,
        budget_per_problem=args.budget_per_problem,
    )
    runner.verbose = args.verbose
    if args.verbose:
        runner.max_concurrent = 1

    def progress(current, total):
        pct = current / total * 100
        print(f"  [{current}/{total}] {pct:.0f}%", end="\r", flush=True)

    summary = runner.run_sync(
        n_problems=n_problems,
        benchmark=args.benchmark,
        progress_callback=progress,
    )

    print(summary.format_report())


if __name__ == "__main__":
    main()
