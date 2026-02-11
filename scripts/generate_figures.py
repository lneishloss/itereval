#!/usr/bin/env python3
"""
Generate figures from itereval result files.

Usage:
    # From a summary JSON (needs matching JSONL for per-problem data)
    python scripts/generate_figures.py itereval/results/iterative_prompt_only_split_20260211.json

    # Specify output directory
    python scripts/generate_figures.py results.json --output-dir figures/

    # Generate from multiple result files
    python scripts/generate_figures.py results1.json results2.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import fields
from pathlib import Path

# Ensure project root is on path
_script_dir = Path(__file__).parent.resolve()
_project_root = _script_dir.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))


def _load_summary_from_files(json_path: Path):
    """Reconstruct an IterativeSummary from JSON + JSONL files."""
    from itereval.benchmarks.iterative_runner import (
        IterativeSummary,
        IterativeProblemResult,
        AttemptResult,
    )

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Load per-problem results from JSONL if available
    jsonl_path = json_path.with_suffix(".jsonl")
    results = []
    if jsonl_path.exists():
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rd = json.loads(line)
                    # Reconstruct AttemptResult lists
                    ctrl_attempts = [
                        AttemptResult(**a) for a in rd.get("control_attempts", [])
                    ]
                    treat_attempts = [
                        AttemptResult(**a) for a in rd.get("treatment_attempts", [])
                    ]
                    results.append(IterativeProblemResult(
                        task_id=rd["task_id"],
                        benchmark=rd["benchmark"],
                        entry_point=rd["entry_point"],
                        control_solved=rd["control_solved"],
                        control_attempts=ctrl_attempts,
                        control_solved_at=rd.get("control_solved_at"),
                        control_total_input_tokens=rd["control_total_input_tokens"],
                        control_total_output_tokens=rd["control_total_output_tokens"],
                        control_total_cost=rd["control_total_cost"],
                        treatment_solved=rd["treatment_solved"],
                        treatment_attempts=treat_attempts,
                        treatment_solved_at=rd.get("treatment_solved_at"),
                        treatment_total_input_tokens=rd["treatment_total_input_tokens"],
                        treatment_total_output_tokens=rd["treatment_total_output_tokens"],
                        treatment_total_cost=rd["treatment_total_cost"],
                        compression_ratio=rd.get("compression_ratio", 1.0),
                        transform_name=rd.get("transform_name", ""),
                    ))

    # Build summary from saved data, filling in what we can
    summary_fields = {f.name for f in fields(IterativeSummary)}
    kwargs = {}
    for key, value in data.items():
        if key in summary_fields:
            kwargs[key] = value
    kwargs["results"] = results

    # Ensure required fields have defaults
    for f in fields(IterativeSummary):
        if f.name not in kwargs:
            if f.name == "control_cumulative_solve_rate":
                kwargs[f.name] = data.get("control_cumulative_solve_rate", [])
            elif f.name == "treatment_cumulative_solve_rate":
                kwargs[f.name] = data.get("treatment_cumulative_solve_rate", [])

    return IterativeSummary(**kwargs)


def main():
    parser = argparse.ArgumentParser(
        description="Generate figures from itereval results",
    )
    parser.add_argument(
        "json_files", nargs="+", type=Path,
        help="Path(s) to summary JSON files",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Output directory for figures (default: same dir as JSON)",
    )
    args = parser.parse_args()

    from itereval.figures import generate_all_figures

    for json_path in args.json_files:
        if not json_path.exists():
            print(f"Not found: {json_path}")
            continue

        print(f"\nGenerating figures from: {json_path}")
        summary = _load_summary_from_files(json_path)

        output_dir = args.output_dir or json_path.parent / "figures"
        paths = generate_all_figures(summary, output_dir)

        if paths:
            print(f"  Generated {len(paths)} figures in {output_dir}")
        else:
            print("  No figures generated (insufficient data)")


if __name__ == "__main__":
    main()
