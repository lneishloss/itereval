"""Tests for itereval.figures — figure generation smoke tests."""

import pytest
from pathlib import Path
from dataclasses import field

from itereval.benchmarks.iterative_runner import (
    IterativeSummary,
    IterativeProblemResult,
    AttemptResult,
)

try:
    import matplotlib
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False

pytestmark = pytest.mark.skipif(not _HAS_MPL, reason="matplotlib not installed")


def _make_attempt(num, passed, in_tok=200, out_tok=100, latency=500.0):
    return AttemptResult(
        attempt_number=num,
        code="def foo(): return 1\n",
        passed=passed,
        error="" if passed else "AssertionError",
        input_tokens=in_tok,
        output_tokens=out_tok,
        latency_ms=latency,
        execution_ms=50.0,
        cumulative_input_tokens=in_tok * num,
        cumulative_output_tokens=out_tok * num,
    )


def _make_result(task_id, ctrl_solved_at, treat_solved_at, max_attempts=3):
    """Create a mock IterativeProblemResult."""
    ctrl_attempts = []
    treat_attempts = []

    for i in range(1, max_attempts + 1):
        ctrl_passed = ctrl_solved_at is not None and i >= ctrl_solved_at
        treat_passed = treat_solved_at is not None and i >= treat_solved_at
        ctrl_attempts.append(_make_attempt(i, ctrl_passed and i == ctrl_solved_at))
        treat_attempts.append(_make_attempt(i, treat_passed and i == treat_solved_at, in_tok=180, out_tok=80))
        if ctrl_passed and i == ctrl_solved_at:
            ctrl_attempts = ctrl_attempts[:i]
        if treat_passed and i == treat_solved_at:
            treat_attempts = treat_attempts[:i]

    ctrl_in = sum(a.input_tokens for a in ctrl_attempts)
    ctrl_out = sum(a.output_tokens for a in ctrl_attempts)
    treat_in = sum(a.input_tokens for a in treat_attempts)
    treat_out = sum(a.output_tokens for a in treat_attempts)

    from itereval.utils import calculate_cost
    return IterativeProblemResult(
        task_id=task_id,
        benchmark="humaneval",
        entry_point="foo",
        control_solved=ctrl_solved_at is not None,
        control_attempts=ctrl_attempts,
        control_solved_at=ctrl_solved_at,
        control_total_input_tokens=ctrl_in,
        control_total_output_tokens=ctrl_out,
        control_total_cost=calculate_cost(ctrl_in, ctrl_out, "claude-sonnet-4-20250514"),
        treatment_solved=treat_solved_at is not None,
        treatment_attempts=treat_attempts,
        treatment_solved_at=treat_solved_at,
        treatment_total_input_tokens=treat_in,
        treatment_total_output_tokens=treat_out,
        treatment_total_cost=calculate_cost(treat_in, treat_out, "claude-sonnet-4-20250514"),
        compression_ratio=0.9,
        transform_name="test",
    )


def _make_summary():
    """Create a minimal but realistic IterativeSummary for testing."""
    results = [
        _make_result("HumanEval/0", ctrl_solved_at=1, treat_solved_at=1),
        _make_result("HumanEval/1", ctrl_solved_at=1, treat_solved_at=2),
        _make_result("HumanEval/2", ctrl_solved_at=2, treat_solved_at=1),
        _make_result("HumanEval/3", ctrl_solved_at=1, treat_solved_at=1),
        _make_result("HumanEval/4", ctrl_solved_at=3, treat_solved_at=None),
        _make_result("HumanEval/5", ctrl_solved_at=None, treat_solved_at=2),
        _make_result("HumanEval/6", ctrl_solved_at=1, treat_solved_at=1),
        _make_result("HumanEval/7", ctrl_solved_at=2, treat_solved_at=1),
        _make_result("HumanEval/8", ctrl_solved_at=1, treat_solved_at=1),
        _make_result("HumanEval/9", ctrl_solved_at=1, treat_solved_at=1),
    ]

    n = len(results)
    ctrl_solved = sum(1 for r in results if r.control_solved)
    treat_solved = sum(1 for r in results if r.treatment_solved)
    ctrl_cost = sum(r.control_total_cost for r in results)
    treat_cost = sum(r.treatment_total_cost for r in results)

    return IterativeSummary(
        benchmark="humaneval",
        transform_name="test_transform",
        model="claude-sonnet-4-20250514",
        max_attempts=3,
        total_problems=n,
        control_solved=ctrl_solved,
        treatment_solved=treat_solved,
        control_cost_per_solve=ctrl_cost / ctrl_solved if ctrl_solved else 0,
        treatment_cost_per_solve=treat_cost / treat_solved if treat_solved else 0,
        cost_per_solve_savings_pct=15.0,
        control_cumulative_solve_rate=[0.6, 0.8, 0.9],
        treatment_cumulative_solve_rate=[0.7, 0.85, 0.9],
        control_total_cost=ctrl_cost,
        treatment_total_cost=treat_cost,
        total_cost_savings_pct=10.0,
        control_avg_attempts=1.5,
        treatment_avg_attempts=1.3,
        control_total_input_tokens=sum(r.control_total_input_tokens for r in results),
        control_total_output_tokens=sum(r.control_total_output_tokens for r in results),
        treatment_total_input_tokens=sum(r.treatment_total_input_tokens for r in results),
        treatment_total_output_tokens=sum(r.treatment_total_output_tokens for r in results),
        control_solve_at_histogram={1: 6, 2: 2, 3: 1},
        treatment_solve_at_histogram={1: 6, 2: 3},
        results=results,
    )


class TestFigureGeneration:
    """Smoke tests for figure generation — ensures figures are created without errors."""

    def test_convergence_curve(self, tmp_path):
        from itereval.figures import plot_convergence_curve
        summary = _make_summary()
        path = plot_convergence_curve(summary, output_path=tmp_path / "conv.png")
        assert path is not None
        assert path.exists()
        assert path.stat().st_size > 1000

    def test_cost_breakdown(self, tmp_path):
        from itereval.figures import plot_cost_breakdown
        summary = _make_summary()
        path = plot_cost_breakdown(summary, output_path=tmp_path / "cost.png")
        assert path is not None
        assert path.exists()

    def test_cost_scatter(self, tmp_path):
        from itereval.figures import plot_cost_scatter
        summary = _make_summary()
        path = plot_cost_scatter(summary, output_path=tmp_path / "scatter.png")
        assert path is not None
        assert path.exists()

    def test_solve_histogram(self, tmp_path):
        from itereval.figures import plot_solve_histogram
        summary = _make_summary()
        path = plot_solve_histogram(summary, output_path=tmp_path / "hist.png")
        assert path is not None
        assert path.exists()

    def test_bootstrap_distribution(self, tmp_path):
        from itereval.figures import plot_bootstrap_distribution
        summary = _make_summary()
        path = plot_bootstrap_distribution(
            summary, output_path=tmp_path / "boot.png", n_bootstrap=200
        )
        assert path is not None
        assert path.exists()

    def test_generate_all(self, tmp_path):
        from itereval.figures import generate_all_figures
        summary = _make_summary()
        paths = generate_all_figures(summary, tmp_path)
        assert len(paths) == 5
        for p in paths:
            assert p.exists()

    def test_empty_results(self, tmp_path):
        """Scatter/histogram with no results should return None gracefully."""
        from itereval.figures import plot_cost_scatter, plot_solve_histogram
        summary = _make_summary()
        summary.results = []
        assert plot_cost_scatter(summary, tmp_path / "empty.png") is None
        summary.control_solve_at_histogram = {}
        summary.treatment_solve_at_histogram = {}
        assert plot_solve_histogram(summary, tmp_path / "empty2.png") is None
