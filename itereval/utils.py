"""
Token estimation and model pricing utilities.

Provides token counting (via tiktoken or heuristic fallback) and
API cost calculation for major LLM providers.
"""

import json
import logging

logger = logging.getLogger(__name__)

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False


class NumpyEncoder(json.JSONEncoder):
    """Handle numpy types in JSON serialization."""
    def default(self, obj):
        if _HAS_NUMPY:
            if isinstance(obj, (np.bool_,)):
                return bool(obj)
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
        return super().default(obj)


def estimate_tokens(text: str, model: str = "") -> int:
    """
    Estimate token count using model-appropriate tokenization.

    Strategy:
    1. For OpenAI models: use tiktoken with model-specific encoding
    2. For all other models (including Claude): use tiktoken cl100k_base
       as a reasonable approximation (~5% error vs Claude's tokenizer)
    3. Fallback: len(text) // 4 heuristic if tiktoken unavailable

    Args:
        text: Text to estimate tokens for
        model: Model name (e.g. "claude-sonnet-4-20250514", "gpt-4o").

    Returns:
        Estimated token count (minimum 1 for non-empty text)
    """
    if not text:
        return 0
    return max(1, _count_tokens_tiktoken(text, model))


def _count_tokens_tiktoken(text: str, model: str = "") -> int:
    """Count tokens using tiktoken. Falls back to heuristic."""
    try:
        import tiktoken
    except ImportError:
        return len(text) // 4

    try:
        if any(m in model for m in ("gpt-4", "gpt-3.5", "o1", "o3")):
            enc = tiktoken.encoding_for_model(model)
        else:
            enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return len(text) // 4


# ---------------------------------------------------------------------------
# Model pricing
# ---------------------------------------------------------------------------

# Pricing per million tokens (USD). Source: provider pricing pages.
# Keys are prefix-matched against model names.
MODEL_PRICING: dict[str, dict[str, float]] = {
    # Anthropic Claude
    "claude-opus-4": {"input": 15.00, "output": 75.00},
    "claude-sonnet-4": {"input": 3.00, "output": 15.00},
    "claude-haiku-4": {"input": 1.00, "output": 5.00},
    "claude-3-5-sonnet": {"input": 3.00, "output": 15.00},
    "claude-3-5-haiku": {"input": 1.00, "output": 5.00},
    "claude-3-opus": {"input": 15.00, "output": 75.00},
    "claude-3-sonnet": {"input": 3.00, "output": 15.00},
    "claude-3-haiku": {"input": 0.25, "output": 1.25},
    # OpenAI
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4-turbo": {"input": 10.00, "output": 30.00},
    "gpt-4": {"input": 30.00, "output": 60.00},
    "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
    "o1": {"input": 15.00, "output": 60.00},
    "o3-mini": {"input": 1.10, "output": 4.40},
    # Google
    "gemini-1.5-pro": {"input": 3.50, "output": 10.50},
    "gemini-1.5-flash": {"input": 0.075, "output": 0.30},
    "gemini-2.0-flash": {"input": 0.10, "output": 0.40},
}


def get_model_pricing(model: str) -> dict[str, float]:
    """
    Look up pricing for a model by prefix matching.

    Returns:
        {"input": price_per_M, "output": price_per_M}
        Falls back to Sonnet 4 pricing if model is unknown.
    """
    for prefix, pricing in MODEL_PRICING.items():
        if model.startswith(prefix):
            return pricing
    logger.warning(
        f"Unknown model '{model}', falling back to Sonnet 4 pricing "
        f"($3.00/$15.00 per M tokens). Update MODEL_PRICING in utils.py "
        f"if this is incorrect."
    )
    return {"input": 3.00, "output": 15.00}


def calculate_cost(input_tokens: int, output_tokens: int, model: str) -> float:
    """
    Calculate API cost in USD for a given token usage.

    Args:
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        model: Model name for pricing lookup

    Returns:
        Cost in USD
    """
    pricing = get_model_pricing(model)
    return (
        input_tokens * pricing["input"] / 1_000_000
        + output_tokens * pricing["output"] / 1_000_000
    )
