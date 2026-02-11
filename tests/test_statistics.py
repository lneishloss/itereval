"""Tests for itereval.statistics — Wilson CIs, McNemar's, bootstrap, permutation."""

import math
import pytest

from itereval.statistics import (
    wilson_score_interval,
    WilsonScoreResult,
    mcnemar_test,
    McNemarResult,
    bootstrap_confidence_interval,
    paired_permutation_test,
)


# =========================================================================
# Wilson Score Interval
# =========================================================================

class TestWilsonScoreInterval:
    """Tests for Wilson score confidence interval."""

    def test_perfect_score(self):
        """100% pass rate should have CI upper bound of 1.0."""
        result = wilson_score_interval(100, 100)
        assert result.proportion == 1.0
        assert result.ci_upper == 1.0
        assert result.ci_lower > 0.95

    def test_zero_score(self):
        """0% pass rate should have CI lower bound of ~0.0."""
        result = wilson_score_interval(0, 100)
        assert result.proportion == 0.0
        assert result.ci_lower == pytest.approx(0.0, abs=1e-10)
        assert result.ci_upper < 0.05

    def test_fifty_percent(self):
        """50% rate with large N should have CI centered near 0.5."""
        result = wilson_score_interval(500, 1000)
        assert abs(result.proportion - 0.5) < 1e-10
        # CI should be symmetric around 0.5 for large N
        assert result.ci_lower > 0.46
        assert result.ci_upper < 0.54

    def test_small_sample(self):
        """Small sample should produce wider CI than large sample."""
        small = wilson_score_interval(8, 10)
        large = wilson_score_interval(800, 1000)
        small_width = small.ci_upper - small.ci_lower
        large_width = large.ci_upper - large.ci_lower
        assert small_width > large_width

    def test_bounds_within_zero_one(self):
        """CI bounds should always be in [0, 1]."""
        for s, n in [(0, 5), (1, 5), (5, 5), (3, 10), (99, 100)]:
            result = wilson_score_interval(s, n)
            assert 0.0 <= result.ci_lower <= result.ci_upper <= 1.0

    def test_empty_input(self):
        """Zero total should return zeros."""
        result = wilson_score_interval(0, 0)
        assert result.proportion == 0.0
        assert result.ci_lower == 0.0
        assert result.ci_upper == 0.0

    def test_humaneval_typical(self):
        """Typical HumanEval results: ~160/164 solved."""
        result = wilson_score_interval(160, 164)
        assert result.proportion == pytest.approx(160 / 164, rel=1e-6)
        assert result.ci_lower > 0.93
        assert result.ci_upper <= 1.0

    def test_result_fields(self):
        """Result dataclass should have all expected fields."""
        result = wilson_score_interval(50, 100)
        assert result.successes == 50
        assert result.total == 100
        assert result.confidence_level == 0.95

    def test_format_percent(self):
        """format_percent should return a readable string."""
        result = wilson_score_interval(80, 100)
        s = result.format_percent()
        assert "80.0%" in s
        assert "Wilson CI" in s

    def test_to_dict(self):
        """to_dict should return all fields."""
        result = wilson_score_interval(50, 100)
        d = result.to_dict()
        assert set(d.keys()) == {
            "proportion", "ci_lower", "ci_upper",
            "successes", "total", "confidence_level",
        }

    def test_custom_confidence(self):
        """99% CI should be wider than 95% CI."""
        ci95 = wilson_score_interval(50, 100, confidence=0.95)
        ci99 = wilson_score_interval(50, 100, confidence=0.99)
        width95 = ci95.ci_upper - ci95.ci_lower
        width99 = ci99.ci_upper - ci99.ci_lower
        assert width99 > width95


# =========================================================================
# McNemar's Test
# =========================================================================

class TestMcNemarTest:
    """Tests for McNemar's test for paired binary outcomes."""

    def test_no_discordant_pairs(self):
        """When conditions always agree, p-value should be 1.0."""
        result = mcnemar_test(90, 0, 0, 10)
        assert result.p_value == 1.0
        assert result.is_significant is False

    def test_perfect_treatment(self):
        """When treatment always beats control, should be significant."""
        # 0 control-only, 20 treatment-only: strongly favors treatment
        result = mcnemar_test(80, 0, 20, 0)
        assert result.p_value < 0.001
        assert result.is_significant == True

    def test_symmetric_discordant(self):
        """Equal discordant pairs should not be significant."""
        result = mcnemar_test(80, 5, 5, 10)
        assert result.p_value > 0.05
        assert result.is_significant is False

    def test_exact_method_small_discordant(self):
        """Small discordant count (<25) should use exact binomial."""
        result = mcnemar_test(100, 3, 1, 60)
        assert result.method == "exact"

    def test_chi_squared_large_discordant(self):
        """Large discordant count (>=25) should use chi-squared."""
        result = mcnemar_test(50, 15, 12, 23)
        assert result.method == "chi_squared"

    def test_result_fields(self):
        """Result should contain contingency table."""
        result = mcnemar_test(80, 5, 3, 12)
        assert result.both_passed == 80
        assert result.control_only == 5
        assert result.treatment_only == 3
        assert result.both_failed == 12

    def test_to_dict(self):
        result = mcnemar_test(80, 5, 3, 12)
        d = result.to_dict()
        assert "p_value" in d
        assert "method" in d
        assert "both_passed" in d

    def test_str_representation(self):
        result = mcnemar_test(80, 5, 3, 12)
        s = str(result)
        assert "McNemar" in s
        assert "p=" in s


# =========================================================================
# Bootstrap Confidence Interval
# =========================================================================

class TestBootstrapCI:
    """Tests for bootstrap confidence interval."""

    def test_constant_values(self):
        """Constant values should give zero-width CI."""
        values = [5.0] * 100
        lower, upper = bootstrap_confidence_interval(values, seed=42)
        assert lower == pytest.approx(5.0)
        assert upper == pytest.approx(5.0)

    def test_ci_contains_mean(self):
        """CI should contain the sample mean for normal-ish data."""
        values = list(range(100))
        mean = sum(values) / len(values)
        lower, upper = bootstrap_confidence_interval(values, seed=42)
        assert lower <= mean <= upper

    def test_wider_ci_with_higher_variance(self):
        """Higher variance data should have wider CI."""
        narrow = [50 + x * 0.1 for x in range(100)]
        wide = [50 + x * 10 for x in range(100)]
        nl, nu = bootstrap_confidence_interval(narrow, seed=42)
        wl, wu = bootstrap_confidence_interval(wide, seed=42)
        assert (wu - wl) > (nu - nl)

    def test_single_value(self):
        """Single value should return that value for both bounds."""
        lower, upper = bootstrap_confidence_interval([42.0])
        assert lower == 42.0
        assert upper == 42.0

    def test_reproducible_with_seed(self):
        """Same seed should give same result."""
        values = [float(x) for x in range(50)]
        r1 = bootstrap_confidence_interval(values, seed=123)
        r2 = bootstrap_confidence_interval(values, seed=123)
        assert r1 == r2

    def test_different_seeds_differ(self):
        """Different seeds should (usually) give different results."""
        values = [float(x) for x in range(50)]
        r1 = bootstrap_confidence_interval(values, seed=1)
        r2 = bootstrap_confidence_interval(values, seed=2)
        # Not guaranteed but extremely likely
        assert r1 != r2

    def test_median_statistic(self):
        """Median bootstrap should work."""
        values = [1.0, 2.0, 3.0, 100.0]
        lower, upper = bootstrap_confidence_interval(
            values, statistic="median", seed=42
        )
        assert lower < upper

    def test_invalid_statistic(self):
        """Unknown statistic should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown statistic"):
            bootstrap_confidence_interval([1.0, 2.0], statistic="mode")


# =========================================================================
# Paired Permutation Test
# =========================================================================

class TestPairedPermutationTest:
    """Tests for paired permutation test on per-problem costs."""

    def test_identical_values(self):
        """Identical costs should give p ≈ 1.0 (no difference)."""
        ctrl = [0.01] * 50
        treat = [0.01] * 50
        p = paired_permutation_test(ctrl, treat, seed=42)
        assert p > 0.3

    def test_clearly_higher_control(self):
        """Control systematically higher should give small p."""
        ctrl = [0.05] * 100
        treat = [0.02] * 100
        p = paired_permutation_test(ctrl, treat, seed=42, alternative="greater")
        assert p < 0.01

    def test_clearly_lower_control(self):
        """Control lower than treatment with 'greater' should give p ≈ 1."""
        ctrl = [0.01] * 100
        treat = [0.05] * 100
        p = paired_permutation_test(ctrl, treat, seed=42, alternative="greater")
        assert p > 0.95

    def test_unequal_lengths_raises(self):
        """Mismatched lengths should raise ValueError."""
        with pytest.raises(ValueError, match="equal lengths"):
            paired_permutation_test([1.0, 2.0], [1.0])

    def test_too_few_values(self):
        """Fewer than 2 values should return 1.0."""
        p = paired_permutation_test([1.0], [2.0])
        assert p == 1.0

    def test_p_value_range(self):
        """p-value should always be in [0, 1]."""
        import random
        rng = random.Random(42)
        ctrl = [rng.uniform(0.01, 0.05) for _ in range(30)]
        treat = [rng.uniform(0.005, 0.04) for _ in range(30)]
        p = paired_permutation_test(ctrl, treat, seed=42)
        assert 0.0 <= p <= 1.0

    def test_reproducible_with_seed(self):
        """Same seed should give same p-value."""
        import random
        rng = random.Random(99)
        ctrl = [rng.uniform(0, 1) for _ in range(20)]
        treat = [rng.uniform(0, 1) for _ in range(20)]
        p1 = paired_permutation_test(ctrl, treat, seed=42)
        p2 = paired_permutation_test(ctrl, treat, seed=42)
        assert p1 == p2
