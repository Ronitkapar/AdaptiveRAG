"""
config.pricing
--------------
Canonical price table for cost estimation across models and providers.
Prices are per 1M tokens in USD.
"""

from typing import Any

PRICING_VERSION = "2026-q1"

# Unit prices in USD per 1,000,000 tokens
MODEL_PRICING: dict[str, dict[str, float]] = {
    # Groq generation models
    "openai/gpt-oss-120b": {
        "input_per_million": 0.15,
        "output_per_million": 0.60,
    },
    "openai/gpt-oss-20b": {
        "input_per_million": 0.075,
        "output_per_million": 0.30,
    },
    "llama-3.1-8b-instant": {
        "input_per_million": 0.05,
        "output_per_million": 0.08,
    },
    # AICredits / OpenAI embeddings
    "text-embedding-3-large": {
        "input_per_million": 0.13,
        "output_per_million": 0.0,
    },
    "openai/text-embedding-3-large": {
        "input_per_million": 0.13,
        "output_per_million": 0.0,
    },
    "text-embedding-3-small": {
        "input_per_million": 0.02,
        "output_per_million": 0.0,
    },
}


def calculate_cost_usd(
    model: str,
    input_tokens: int = 0,
    output_tokens: int = 0,
) -> float:
    """Calculate estimated cost in USD based on model pricing table."""
    pricing = MODEL_PRICING.get(model)
    if not pricing:
        return 0.0

    cost = (input_tokens / 1_000_000.0) * pricing["input_per_million"] + (
        output_tokens / 1_000_000.0
    ) * pricing["output_per_million"]
    return round(cost, 7)
