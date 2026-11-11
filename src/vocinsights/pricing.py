"""Per-call cost accounting. Prices in USD per million tokens."""

from __future__ import annotations

PRICES = {
    "claude-haiku-5-5": {"input": 0.10, "output": 0.50},
    "claude-opus-5-5": {"input": 4.00, "output": 20.00},
}

# Rules and the classifier run on CPU we already pay for: about 2 ms of a $0.10/hour core per text.
LOCAL_COST_PER_TEXT = 0.00000006


def call_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    p = PRICES[model]
    return (input_tokens * p["input"] + output_tokens * p["output"]) / 1_000_000
