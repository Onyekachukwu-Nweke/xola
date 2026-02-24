"""LLM pricing table and cost calculation.

Pricing data is sourced from OpenAI's published rates as of 2024-12.
Costs are in USD per token.

Usage::

    >>> from llm_surface.costs import calculate_cost
    >>> calculate_cost("gpt-4o", tokens_in=500, tokens_out=200)
    0.00325
"""

# Per-token pricing in USD.
# Keys are OpenAI model names (or prefixes for versioned models).
PRICING: dict[str, dict[str, float]] = {
    "gpt-4o": {
        "input": 2.50 / 1_000_000,
        "output": 10.00 / 1_000_000,
    },
    "gpt-4o-mini": {
        "input": 0.15 / 1_000_000,
        "output": 0.60 / 1_000_000,
    },
    "gpt-4-turbo": {
        "input": 10.00 / 1_000_000,
        "output": 30.00 / 1_000_000,
    },
    "gpt-3.5-turbo": {
        "input": 0.50 / 1_000_000,
        "output": 1.50 / 1_000_000,
    },
    "text-embedding-3-small": {
        "input": 0.02 / 1_000_000,
    },
    "text-embedding-3-large": {
        "input": 0.13 / 1_000_000,
    },
}


def calculate_cost(model: str, tokens_in: int, tokens_out: int = 0) -> float:
    """Calculate the estimated cost in USD for a single LLM call.

    If the model is not found in the pricing table, returns 0.0 (unknown
    models are not an error — they just don't contribute to cost tracking).

    Args:
        model:      OpenAI model name (e.g. ``"gpt-4o"``).
        tokens_in:  Number of input (prompt) tokens.
        tokens_out: Number of output (completion) tokens. Defaults to 0
                    (embedding models have no output tokens).

    Returns:
        Estimated cost in USD.
    """
    prices = PRICING.get(model, {})
    input_price = prices.get("input", 0.0)
    output_price = prices.get("output", 0.0)
    return tokens_in * input_price + tokens_out * output_price
