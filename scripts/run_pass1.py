#!/usr/bin/env python3
"""
CLI entry point for pass@1 evaluation.

Usage:
    # Default: system prompt treatment only
    python scripts/run_pass1.py -n 20

    # Use same prompt for both arms (no treatment)
    python scripts/run_pass1.py --same-prompt -n 20

    # Full HumanEval
    python scripts/run_pass1.py --all

    # Dry run
    python scripts/run_pass1.py --dry-run
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
        description="Pass@1 Evaluation Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s -n 20
  %(prog)s --all
  %(prog)s --same-prompt -n 20
  %(prog)s --dry-run
        """,
    )

    prompt_group = parser.add_mutually_exclusive_group()
    prompt_group.add_argument(
        "--same-prompt", action="store_true",
        help="Use the same system prompt for both arms (control prompt).",
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
        "--dry-run", action="store_true",
        help="Show configuration without making API calls",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Show full per-problem detail: prompts, generated code, errors",
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

    from itereval.benchmarks.pass1_runner import Pass1Runner

    treatment_override, control_override = resolve_prompts(args)
    n_problems = None if args.all else args.n

    if args.dry_run:
        print("\n=== DRY RUN -- Configuration ===\n")
        runner = Pass1Runner(
            system_prompt=treatment_override,
            control_system_prompt=control_override,
            model=args.model,
            output_dir=args.output_dir,
        )
        config = runner.get_config()
        print(json.dumps(config, indent=2))
        print()
        print(f"Benchmark: {args.benchmark}")
        print(f"Problems: {'all' if args.all else args.n}")
        n_api_calls = (n_problems or 164) * 2
        print(f"API calls: {n_api_calls} (control + treatment)")
        return

    print(f"\nRunning pass@1 evaluation")

    runner = Pass1Runner(
        system_prompt=treatment_override,
        control_system_prompt=control_override,
        model=args.model,
        max_concurrent=args.max_concurrent,
        output_dir=args.output_dir,
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
