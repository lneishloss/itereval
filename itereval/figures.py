"""
Figure generation for itereval results.

Produces publication-ready figures from IterativeSummary objects:
1. Convergence curve (cumulative solve rate by attempt)
2. Cost waterfall (input vs output cost breakdown)
3. Per-problem cost scatter (control vs treatment)
4. Solve-at-attempt histogram
5. Bootstrap CPS distribution

Requires matplotlib: pip install matplotlib
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from itereval.benchmarks.iterative_runner import IterativeSummary

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False


def _check_matplotlib():
    if not _HAS_MPL:
        raise ImportError(
            "matplotlib is required for figure generation. "
            "Install with: pip install matplotlib"
        )


# ---------------------------------------------------------------------------
# Style defaults
# ---------------------------------------------------------------------------

_COLORS = {
    "control": "#4A90D9",
    "treatment": "#D94A4A",
    "input": "#6CB4EE",
    "output": "#FF8C42",
}

_STYLE = {
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "figure.facecolor": "white",
}


def _apply_style():
    plt.rcParams.update(_STYLE)


# ---------------------------------------------------------------------------
# 1. Convergence curve
# ---------------------------------------------------------------------------

def plot_convergence_curve(
    summary: "IterativeSummary",
    output_path: Optional[Path] = None,
) -> Optional[Path]:
    """
    Cumulative solve rate by attempt number (line chart).

    Shows how accuracy improves with each retry for both arms.
    """
    _check_matplotlib()
    _apply_style()

    attempts = list(range(1, summary.max_attempts + 1))
    ctrl_rates = [r * 100 for r in summary.control_cumulative_solve_rate]
    treat_rates = [r * 100 for r in summary.treatment_cumulative_solve_rate]

    fig, ax = plt.subplots(figsize=(7, 4.5))

    ax.plot(attempts, ctrl_rates, "o-", color=_COLORS["control"],
            linewidth=2, markersize=8, label="Control")
    ax.plot(attempts, treat_rates, "s--", color=_COLORS["treatment"],
            linewidth=2, markersize=8, label="Treatment")

    ax.set_xlabel("Attempt Number")
    ax.set_ylabel("Cumulative Solve Rate (%)")
    ax.set_title("Convergence: Cumulative Solve Rate by Attempt")
    ax.set_xticks(attempts)
    ax.set_ylim(0, 105)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f%%"))
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    if output_path is None:
        output_path = Path(f"convergence_{summary.transform_name}.png")
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


# ---------------------------------------------------------------------------
# 2. Cost waterfall / stacked bar
# ---------------------------------------------------------------------------

def plot_cost_breakdown(
    summary: "IterativeSummary",
    output_path: Optional[Path] = None,
) -> Optional[Path]:
    """
    Stacked bar chart of input vs output costs for control vs treatment.

    Visually demonstrates that output tokens dominate cost.
    """
    _check_matplotlib()
    _apply_style()
    from itereval.utils import get_model_pricing

    pricing = get_model_pricing(summary.model)

    ctrl_in_cost = summary.control_total_input_tokens * pricing["input"] / 1_000_000
    ctrl_out_cost = summary.control_total_output_tokens * pricing["output"] / 1_000_000
    treat_in_cost = summary.treatment_total_input_tokens * pricing["input"] / 1_000_000
    treat_out_cost = summary.treatment_total_output_tokens * pricing["output"] / 1_000_000

    fig, ax = plt.subplots(figsize=(6, 5))

    labels = ["Control", "Treatment"]
    in_costs = [ctrl_in_cost, treat_in_cost]
    out_costs = [ctrl_out_cost, treat_out_cost]

    x = range(len(labels))
    bars_in = ax.bar(x, in_costs, 0.5, label="Input tokens",
                     color=_COLORS["input"])
    bars_out = ax.bar(x, out_costs, 0.5, bottom=in_costs,
                      label="Output tokens", color=_COLORS["output"])

    # Annotate totals
    for i, (ic, oc) in enumerate(zip(in_costs, out_costs)):
        total = ic + oc
        ax.text(i, total + 0.002, f"${total:.4f}", ha="center",
                va="bottom", fontweight="bold", fontsize=10)

    ax.set_ylabel("Cost (USD)")
    ax.set_title("Cost Breakdown: Input vs Output Tokens")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()

    if output_path is None:
        output_path = Path(f"cost_breakdown_{summary.transform_name}.png")
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


# ---------------------------------------------------------------------------
# 3. Per-problem cost scatter
# ---------------------------------------------------------------------------

def plot_cost_scatter(
    summary: "IterativeSummary",
    output_path: Optional[Path] = None,
) -> Optional[Path]:
    """
    Scatter plot of per-problem costs: control (x) vs treatment (y).

    Points below the diagonal represent problems where treatment is cheaper.
    """
    _check_matplotlib()
    _apply_style()

    results = summary.results
    if not results:
        return None

    ctrl_costs = [r.control_total_cost for r in results]
    treat_costs = [r.treatment_total_cost for r in results]

    fig, ax = plt.subplots(figsize=(6, 6))

    ax.scatter(ctrl_costs, treat_costs, alpha=0.6, s=30,
               color=_COLORS["treatment"], edgecolors="white", linewidth=0.5)

    # Diagonal (break-even line)
    max_cost = max(max(ctrl_costs), max(treat_costs)) * 1.1
    ax.plot([0, max_cost], [0, max_cost], "k--", alpha=0.4, linewidth=1,
            label="Break-even")

    ax.set_xlabel("Control Cost (USD)")
    ax.set_ylabel("Treatment Cost (USD)")
    ax.set_title("Per-Problem Cost: Control vs Treatment")
    ax.set_xlim(0, max_cost)
    ax.set_ylim(0, max_cost)
    ax.set_aspect("equal")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    # Count points below/above diagonal
    below = sum(1 for c, t in zip(ctrl_costs, treat_costs) if t < c)
    above = sum(1 for c, t in zip(ctrl_costs, treat_costs) if t > c)
    equal = len(ctrl_costs) - below - above
    ax.text(0.97, 0.03,
            f"Treatment cheaper: {below}\nTreatment costlier: {above}\nEqual: {equal}",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=9, bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                                  alpha=0.8))

    fig.tight_layout()

    if output_path is None:
        output_path = Path(f"cost_scatter_{summary.transform_name}.png")
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


# ---------------------------------------------------------------------------
# 4. Solve-at-attempt histogram
# ---------------------------------------------------------------------------

def plot_solve_histogram(
    summary: "IterativeSummary",
    output_path: Optional[Path] = None,
) -> Optional[Path]:
    """
    Bar chart showing how many problems were solved at each attempt number.
    """
    _check_matplotlib()
    _apply_style()

    ctrl_hist = summary.control_solve_at_histogram
    treat_hist = summary.treatment_solve_at_histogram

    if not ctrl_hist and not treat_hist:
        return None

    attempts = sorted(set(list(ctrl_hist.keys()) + list(treat_hist.keys())))
    ctrl_counts = [ctrl_hist.get(a, 0) for a in attempts]
    treat_counts = [treat_hist.get(a, 0) for a in attempts]

    fig, ax = plt.subplots(figsize=(6, 4.5))

    width = 0.35
    x_pos = range(len(attempts))
    ax.bar([x - width / 2 for x in x_pos], ctrl_counts, width,
           label="Control", color=_COLORS["control"])
    ax.bar([x + width / 2 for x in x_pos], treat_counts, width,
           label="Treatment", color=_COLORS["treatment"])

    ax.set_xlabel("Attempt Number")
    ax.set_ylabel("Problems Solved")
    ax.set_title("Solve-at-Attempt Distribution")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(a) for a in attempts])
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()

    if output_path is None:
        output_path = Path(f"solve_histogram_{summary.transform_name}.png")
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


# ---------------------------------------------------------------------------
# 5. Bootstrap CPS distribution
# ---------------------------------------------------------------------------

def plot_bootstrap_distribution(
    summary: "IterativeSummary",
    output_path: Optional[Path] = None,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> Optional[Path]:
    """
    Histogram of bootstrap CPS savings distribution with CI bounds.

    Resamples all N problems (with replacement), recomputes full CPS
    each iteration. Shows the distribution of savings and marks the
    95% CI bounds.
    """
    _check_matplotlib()
    _apply_style()

    results = summary.results
    if not results or len(results) < 2:
        return None

    n = len(results)
    rng = random.Random(seed)

    bootstrap_savings = []
    for _ in range(n_bootstrap):
        sample = rng.choices(results, k=n)
        ctrl_cost = sum(r.control_total_cost for r in sample)
        treat_cost = sum(r.treatment_total_cost for r in sample)
        ctrl_solved = sum(1 for r in sample if r.control_solved)
        treat_solved = sum(1 for r in sample if r.treatment_solved)
        if ctrl_solved > 0 and treat_solved > 0:
            ctrl_cps = ctrl_cost / ctrl_solved
            treat_cps = treat_cost / treat_solved
            if ctrl_cps > 0:
                bootstrap_savings.append((1 - treat_cps / ctrl_cps) * 100)

    if len(bootstrap_savings) < 10:
        return None

    fig, ax = plt.subplots(figsize=(7, 4.5))

    ax.hist(bootstrap_savings, bins=40, color=_COLORS["treatment"],
            alpha=0.7, edgecolor="white", linewidth=0.5)

    # CI bounds
    bootstrap_savings.sort()
    alpha = 0.05
    lower_idx = max(0, round(len(bootstrap_savings) * alpha / 2) - 1)
    upper_idx = min(len(bootstrap_savings) - 1, round(len(bootstrap_savings) * (1 - alpha / 2)) - 1)
    ci_lower = bootstrap_savings[lower_idx]
    ci_upper = bootstrap_savings[upper_idx]

    ax.axvline(ci_lower, color="black", linestyle="--", linewidth=1.5,
               label=f"95% CI: [{ci_lower:.1f}%, {ci_upper:.1f}%]")
    ax.axvline(ci_upper, color="black", linestyle="--", linewidth=1.5)
    ax.axvline(0, color="gray", linestyle=":", linewidth=1, alpha=0.7)

    # Observed savings
    observed = summary.cost_per_solve_savings_pct
    ax.axvline(observed, color=_COLORS["control"], linestyle="-",
               linewidth=2, label=f"Observed: {observed:.1f}%")

    ax.set_xlabel("CPS Savings (%)")
    ax.set_ylabel("Bootstrap Samples")
    ax.set_title(f"Bootstrap Distribution of CPS Savings (N={n}, {n_bootstrap} resamples)")
    ax.legend(loc="upper left")
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()

    if output_path is None:
        output_path = Path(f"bootstrap_{summary.transform_name}.png")
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


# ---------------------------------------------------------------------------
# Generate all figures
# ---------------------------------------------------------------------------

def generate_all_figures(
    summary: "IterativeSummary",
    output_dir: Path,
    prefix: str = "",
) -> list[Path]:
    """
    Generate all figures for an IterativeSummary.

    Args:
        summary: The summary to visualize.
        output_dir: Directory to save figures.
        prefix: Optional filename prefix.

    Returns:
        List of paths to generated figure files.
    """
    _check_matplotlib()
    output_dir.mkdir(parents=True, exist_ok=True)

    tag = f"{prefix}_" if prefix else ""
    name = summary.transform_name
    paths = []

    generators = [
        ("convergence", plot_convergence_curve),
        ("cost_breakdown", plot_cost_breakdown),
        ("cost_scatter", plot_cost_scatter),
        ("solve_histogram", plot_solve_histogram),
        ("bootstrap", plot_bootstrap_distribution),
    ]

    for fig_name, func in generators:
        try:
            path = func(
                summary,
                output_path=output_dir / f"{tag}{fig_name}_{name}.png",
            )
            if path:
                paths.append(path)
                print(f"  Figure: {path}")
        except Exception as e:
            print(f"  Warning: Failed to generate {fig_name}: {e}")

    return paths
