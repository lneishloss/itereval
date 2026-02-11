"""Tests for itereval.utils — token estimation, pricing, and cost calculation."""

import pytest

from itereval.utils import (
    estimate_tokens,
    get_model_pricing,
    calculate_cost,
    MODEL_PRICING,
    NumpyEncoder,
)


# =========================================================================
# Token estimation
# =========================================================================

class TestEstimateTokens:
    """Tests for token count estimation."""

    def test_empty_string(self):
        assert estimate_tokens("") == 0

    def test_nonempty_returns_positive(self):
        assert estimate_tokens("hello world") >= 1

    def test_longer_text_more_tokens(self):
        short = estimate_tokens("hello")
        long = estimate_tokens("hello world this is a longer string with more tokens")
        assert long > short

    def test_minimum_one_for_nonempty(self):
        """Even a single character should return at least 1."""
        assert estimate_tokens("x") >= 1

    def test_model_hint_accepted(self):
        """Model parameter should not cause errors."""
        result = estimate_tokens("hello world", model="claude-sonnet-4-20250514")
        assert result >= 1

    def test_openai_model_hint(self):
        """OpenAI model names should be accepted."""
        result = estimate_tokens("hello world", model="gpt-4o")
        assert result >= 1


# =========================================================================
# Model pricing
# =========================================================================

class TestModelPricing:
    """Tests for model price lookup."""

    def test_known_model(self):
        pricing = get_model_pricing("claude-sonnet-4-20250514")
        assert pricing["input"] == 3.00
        assert pricing["output"] == 15.00

    def test_opus_pricing(self):
        pricing = get_model_pricing("claude-opus-4-20250514")
        assert pricing["input"] == 15.00
        assert pricing["output"] == 75.00

    def test_haiku_pricing(self):
        pricing = get_model_pricing("claude-haiku-4-20250514")
        assert pricing["input"] == 1.00
        assert pricing["output"] == 5.00

    def test_gpt4o_pricing(self):
        pricing = get_model_pricing("gpt-4o")
        assert pricing["input"] == 2.50
        assert pricing["output"] == 10.00

    def test_unknown_model_fallback(self):
        """Unknown model should fall back to Sonnet pricing."""
        pricing = get_model_pricing("some-unknown-model-xyz")
        assert pricing["input"] == 3.00
        assert pricing["output"] == 15.00

    def test_prefix_matching(self):
        """Pricing should match on prefix."""
        p1 = get_model_pricing("claude-sonnet-4-20250514")
        p2 = get_model_pricing("claude-sonnet-4-some-other-date")
        assert p1 == p2

    def test_output_always_more_expensive(self):
        """Output tokens should cost more than input for all models."""
        for model, pricing in MODEL_PRICING.items():
            assert pricing["output"] > pricing["input"], (
                f"Output should be more expensive for {model}"
            )


# =========================================================================
# Cost calculation
# =========================================================================

class TestCalculateCost:
    """Tests for API cost calculation."""

    def test_zero_tokens(self):
        assert calculate_cost(0, 0, "claude-sonnet-4-20250514") == 0.0

    def test_input_only(self):
        # 1M input tokens at $3/M = $3.00
        cost = calculate_cost(1_000_000, 0, "claude-sonnet-4-20250514")
        assert cost == pytest.approx(3.00)

    def test_output_only(self):
        # 1M output tokens at $15/M = $15.00
        cost = calculate_cost(0, 1_000_000, "claude-sonnet-4-20250514")
        assert cost == pytest.approx(15.00)

    def test_mixed(self):
        # 500 input @ $3/M + 200 output @ $15/M
        cost = calculate_cost(500, 200, "claude-sonnet-4-20250514")
        expected = 500 * 3.0 / 1_000_000 + 200 * 15.0 / 1_000_000
        assert cost == pytest.approx(expected)

    def test_typical_humaneval_problem(self):
        """Typical HumanEval call: ~200 input, ~150 output tokens."""
        cost = calculate_cost(200, 150, "claude-sonnet-4-20250514")
        # 200 * 3/1M + 150 * 15/1M = 0.0006 + 0.00225 = 0.00285
        assert cost == pytest.approx(0.00285)
        assert cost > 0
        assert cost < 0.01

    def test_output_dominates_cost(self):
        """With equal token counts, output cost should dominate."""
        cost = calculate_cost(1000, 1000, "claude-sonnet-4-20250514")
        input_cost = 1000 * 3.0 / 1_000_000
        output_cost = 1000 * 15.0 / 1_000_000
        assert output_cost > input_cost
        assert output_cost / cost > 0.8


# =========================================================================
# NumpyEncoder
# =========================================================================

class TestNumpyEncoder:
    """Tests for JSON serialization of numpy types."""

    def test_plain_types(self):
        """Non-numpy types should serialize normally."""
        import json
        data = {"a": 1, "b": 2.5, "c": True, "d": "hello"}
        result = json.dumps(data, cls=NumpyEncoder)
        assert '"a": 1' in result

    def test_numpy_int(self):
        """numpy integers should serialize to Python int."""
        try:
            import numpy as np
            import json
            data = {"val": np.int64(42)}
            result = json.loads(json.dumps(data, cls=NumpyEncoder))
            assert result["val"] == 42
            assert isinstance(result["val"], int)
        except ImportError:
            pytest.skip("numpy not installed")

    def test_numpy_float(self):
        """numpy floats should serialize to Python float."""
        try:
            import numpy as np
            import json
            data = {"val": np.float64(3.14)}
            result = json.loads(json.dumps(data, cls=NumpyEncoder))
            assert result["val"] == pytest.approx(3.14)
        except ImportError:
            pytest.skip("numpy not installed")

    def test_numpy_bool(self):
        try:
            import numpy as np
            import json
            data = {"val": np.bool_(True)}
            result = json.loads(json.dumps(data, cls=NumpyEncoder))
            assert result["val"] is True
        except ImportError:
            pytest.skip("numpy not installed")
