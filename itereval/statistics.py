"""
Statistical Analysis Module for LLM Evaluation.

Methods:
- Wilson score confidence intervals for binary outcomes
  (per Bowyer et al., "Don't Use the CLT in LLM Evals," ICML 2025)
- McNemar's test for paired binary comparisons
  (per Dietterich, 1998; Dror et al., ACL 2018)
- Bootstrap confidence intervals for continuous metrics
- Paired permutation test for per-problem cost differences

References:
    Bowyer, S., Aitchison, L., & Ivanova, D.R. (2025) "Position: Don't Use
    the CLT in LLM Evals With Fewer Than a Few Hundred Datapoints" - ICML 2025
    https://arxiv.org/abs/2503.01747

    Miller, E. (2024) "Adding Error Bars to Evals: A Statistical Approach
    to Language Model Evaluations" - Anthropic Research
    https://arxiv.org/abs/2411.00640

    Dietterich, T.G. (1998) "Approximate Statistical Tests for Comparing
    Supervised Classification Learning Algorithms" - Neural Computation

    Dror, R., et al. (2018) "The Hitchhiker's Guide to Testing Statistical
    Significance in Natural Language Processing" - ACL 2018
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

from scipy.stats import norm, chi2, binomtest, permutation_test


def bootstrap_confidence_interval(
    values: Sequence[float],
    confidence: float = 0.95,
    n_bootstrap: int = 1000,
    statistic: str = "mean",
    seed: Optional[int] = None,
) -> tuple[float, float]:
    """
    Compute bootstrap confidence interval (non-parametric).

    Useful when the distribution of the statistic is unknown.

    Args:
        values: Sample values
        confidence: Confidence level (default: 0.95)
        n_bootstrap: Number of bootstrap samples (default: 1000)
        statistic: "mean" or "median"
        seed: Random seed for reproducibility

    Returns:
        (ci_lower, ci_upper)
    """
    import random

    rng = random.Random(seed)

    n = len(values)
    if n < 2:
        val = values[0] if n == 1 else 0.0
        return val, val

    values_list = list(values)

    # Generate bootstrap samples
    bootstrap_stats = []
    for _ in range(n_bootstrap):
        sample = [rng.choice(values_list) for _ in range(n)]
        if statistic == "mean":
            stat = sum(sample) / n
        elif statistic == "median":
            sorted_sample = sorted(sample)
            mid = n // 2
            if n % 2 == 0:
                stat = (sorted_sample[mid - 1] + sorted_sample[mid]) / 2
            else:
                stat = sorted_sample[mid]
        else:
            raise ValueError(f"Unknown statistic: {statistic}")
        bootstrap_stats.append(stat)

    # Percentile method for CI
    bootstrap_stats.sort()
    alpha = 1 - confidence
    lower_idx = max(0, round(n_bootstrap * alpha / 2) - 1)
    upper_idx = min(len(bootstrap_stats) - 1, round(n_bootstrap * (1 - alpha / 2)) - 1)

    return bootstrap_stats[lower_idx], bootstrap_stats[upper_idx]


# =============================================================================
# Wilson Score Interval (for binary outcomes like pass@1)
# =============================================================================

@dataclass
class WilsonScoreResult:
    """Result of Wilson score interval calculation for a proportion."""
    proportion: float
    ci_lower: float
    ci_upper: float
    successes: int
    total: int
    confidence_level: float = 0.95

    def __str__(self) -> str:
        return (
            f"{self.proportion:.4f} "
            f"(95% Wilson CI: [{self.ci_lower:.4f}, {self.ci_upper:.4f}], "
            f"n={self.total})"
        )

    def format_percent(self) -> str:
        return (
            f"{self.proportion * 100:.1f}% "
            f"(95% Wilson CI: [{self.ci_lower * 100:.1f}%, {self.ci_upper * 100:.1f}%])"
        )

    def to_dict(self) -> dict:
        return {
            "proportion": self.proportion,
            "ci_lower": self.ci_lower,
            "ci_upper": self.ci_upper,
            "successes": self.successes,
            "total": self.total,
            "confidence_level": self.confidence_level,
        }


def wilson_score_interval(
    successes: int,
    total: int,
    confidence: float = 0.95,
) -> WilsonScoreResult:
    """
    Compute Wilson score confidence interval for a proportion.

    Wilson score intervals are preferred over normal/Wald intervals for
    binary outcomes because they:
    - Never produce intervals outside [0, 1]
    - Perform well for small samples (N < 300)
    - Are recommended by the ICML 2025 paper "Don't Use the CLT in LLM
      Evals With Fewer Than a Few Hundred Datapoints"

    Args:
        successes: Number of successes (e.g., problems passed)
        total: Total number of trials
        confidence: Confidence level (default: 0.95)

    Returns:
        WilsonScoreResult with proportion and CI bounds
    """
    if total == 0:
        return WilsonScoreResult(
            proportion=0.0, ci_lower=0.0, ci_upper=0.0,
            successes=0, total=0, confidence_level=confidence,
        )

    p = successes / total
    n = total

    z = norm.ppf(1 - (1 - confidence) / 2)

    z2 = z * z
    denominator = 1 + z2 / n
    centre = (p + z2 / (2 * n)) / denominator
    spread = z * math.sqrt((p * (1 - p) + z2 / (4 * n)) / n) / denominator

    ci_lower = max(0.0, centre - spread)
    ci_upper = min(1.0, centre + spread)

    return WilsonScoreResult(
        proportion=p,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        successes=successes,
        total=total,
        confidence_level=confidence,
    )


# =============================================================================
# McNemar's Test (for paired binary outcomes)
# =============================================================================

@dataclass
class McNemarResult:
    """Result of McNemar's test for paired binary outcomes."""
    statistic: float
    p_value: float
    method: str  # "exact" or "chi_squared"
    is_significant: bool
    both_passed: int
    control_only: int  # control passed, treatment failed
    treatment_only: int  # treatment passed, control failed
    both_failed: int
    alpha: float = 0.05

    def __str__(self) -> str:
        sig = "significant" if self.is_significant else "not significant"
        return (
            f"McNemar's ({self.method}): statistic={self.statistic:.4f}, "
            f"p={self.p_value:.4f} ({sig}), "
            f"discordant={self.control_only}+{self.treatment_only}"
        )

    def to_dict(self) -> dict:
        return {
            "statistic": self.statistic,
            "p_value": self.p_value,
            "method": self.method,
            "is_significant": self.is_significant,
            "both_passed": self.both_passed,
            "control_only": self.control_only,
            "treatment_only": self.treatment_only,
            "both_failed": self.both_failed,
            "alpha": self.alpha,
        }


def mcnemar_test(
    both_passed: int,
    control_only: int,
    treatment_only: int,
    both_failed: int,
    alpha: float = 0.05,
) -> McNemarResult:
    """
    Perform McNemar's test for paired binary outcomes.

    McNemar's test is the correct test for comparing two classifiers (or
    conditions) on the same dataset with binary outcomes, as recommended
    for pass@1 comparisons. It only uses discordant pairs (where one
    condition passed and the other failed).

    Uses exact binomial test when discordant pairs < 25, otherwise
    chi-squared approximation with continuity correction.

    Args:
        both_passed: Number of problems where both conditions passed
        control_only: Problems where only control passed (regressions)
        treatment_only: Problems where only treatment passed (improvements)
        both_failed: Number of problems where both conditions failed
        alpha: Significance level (default: 0.05)

    Returns:
        McNemarResult with test statistic and p-value
    """
    discordant = control_only + treatment_only

    # No discordant pairs — no difference to detect
    if discordant == 0:
        return McNemarResult(
            statistic=0.0, p_value=1.0, method="exact",
            is_significant=False,
            both_passed=both_passed, control_only=control_only,
            treatment_only=treatment_only, both_failed=both_failed,
            alpha=alpha,
        )

    # Choose method based on discordant pair count
    if discordant < 25:
        # Exact binomial test
        method = "exact"
        result = binomtest(control_only, discordant, 0.5)
        p_value = result.pvalue
        statistic = float(control_only)
    else:
        # Chi-squared with continuity correction
        method = "chi_squared"
        statistic = (abs(control_only - treatment_only) - 1) ** 2 / discordant
        p_value = 1 - chi2.cdf(statistic, df=1)

    return McNemarResult(
        statistic=statistic,
        p_value=p_value,
        method=method,
        is_significant=p_value < alpha,
        both_passed=both_passed,
        control_only=control_only,
        treatment_only=treatment_only,
        both_failed=both_failed,
        alpha=alpha,
    )


# =============================================================================
# Paired Permutation Test (for per-problem cost differences)
# =============================================================================

def paired_permutation_test(
    control_values: Sequence[float],
    treatment_values: Sequence[float],
    n_resamples: int = 9999,
    alternative: str = "greater",
    seed: Optional[int] = None,
) -> float:
    """
    Paired permutation test on per-problem cost differences.

    Non-parametric test for whether control costs are systematically higher
    (or lower) than treatment costs on the same problems. No distributional
    assumptions — works correctly with the right-skewed cost distributions
    typical of iterative LLM evaluations.

    Preferred here over a paired t-test because LLM cost distributions are
    heavily right-skewed, violating the normality assumption. Dror et al.
    (ACL 2018) discuss permutation tests as an assumption-free alternative
    for NLP system comparisons.

    Args:
        control_values: Per-problem costs for the control arm.
        treatment_values: Per-problem costs for the treatment arm (same order).
        n_resamples: Number of permutations (default: 9999).
        alternative: "greater" (control > treatment, i.e. treatment saves),
                     "less", or "two-sided".
        seed: Random seed for reproducibility.

    Returns:
        p-value from the permutation test.
    """
    import numpy as np

    if len(control_values) != len(treatment_values):
        raise ValueError(
            f"Paired test requires equal lengths: "
            f"got {len(control_values)} vs {len(treatment_values)}"
        )
    if len(control_values) < 2:
        return 1.0

    ctrl = np.array(control_values, dtype=float)
    treat = np.array(treatment_values, dtype=float)

    def statistic(x, y, axis):
        return np.mean(x - y, axis=axis)

    rng = np.random.default_rng(seed)
    result = permutation_test(
        (ctrl, treat),
        statistic,
        permutation_type="samples",
        n_resamples=n_resamples,
        alternative=alternative,
        random_state=rng,
    )
    return float(result.pvalue)
