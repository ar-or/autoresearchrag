"""Shared token-cost calculator for all evaluators.

Maps model names to per-million-token pricing and computes dollar costs
from (input, cached, output) token counts.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelPricing:
    input_per_m: float
    cached_per_m: float
    output_per_m: float


# Pricing per 1M tokens (USD).  Last updated: 2026-03-13.
# Add new models here as needed.
MODEL_PRICING: dict[str, ModelPricing] = {
    # OpenAI — GPT-5 family
    "gpt-5.4": ModelPricing(2.50, 0.25, 15.00),
    "gpt-5.2": ModelPricing(1.75, 0.175, 14.00),
    "gpt-5.1": ModelPricing(1.25, 0.125, 10.00),
    "gpt-5": ModelPricing(1.25, 0.125, 10.00),
    "gpt-5-mini": ModelPricing(0.25, 0.025, 2.00),
    "gpt-5-nano": ModelPricing(0.05, 0.005, 0.40),
    # OpenAI — GPT-4.1 family
    "gpt-4.1": ModelPricing(3.50, 0.875, 14.00),
    "gpt-4.1-mini": ModelPricing(0.70, 0.175, 2.80),
    "gpt-4.1-nano": ModelPricing(0.20, 0.05, 0.80),
    # OpenAI — older
    "gpt-4o": ModelPricing(2.50, 1.25, 10.00),
    "gpt-4o-mini": ModelPricing(0.15, 0.075, 0.60),
    "o3-mini": ModelPricing(1.10, 0.55, 4.40),
    # Anthropic — Claude 4.6
    "claude-opus-4-6": ModelPricing(5.00, 0.50, 25.00),
    "claude-sonnet-4-6": ModelPricing(3.00, 0.30, 15.00),
    # Anthropic — Claude 4.5
    "claude-opus-4-5": ModelPricing(5.00, 0.50, 25.00),
    "claude-sonnet-4-5": ModelPricing(3.00, 0.30, 15.00),
    "claude-haiku-4-5": ModelPricing(1.00, 0.10, 5.00),
    # Anthropic — Claude 4.1
    "claude-opus-4-1": ModelPricing(15.00, 1.50, 75.00),
    # Anthropic — Claude 4
    "claude-opus-4": ModelPricing(15.00, 1.50, 75.00),
    "claude-sonnet-4": ModelPricing(3.00, 0.30, 15.00),
}

_DEFAULT_PRICING = ModelPricing(0.25, 0.025, 2.00)


class CostCalculator:
    """Calculate dollar costs from token usage and model name."""

    def __init__(self, model: str = "unknown") -> None:
        self._model = model
        self._pricing = self._resolve(model)

    @property
    def model(self) -> str:
        return self._model

    @property
    def pricing(self) -> ModelPricing:
        return self._pricing

    @staticmethod
    def _resolve(model: str) -> ModelPricing:
        if model in MODEL_PRICING:
            return MODEL_PRICING[model]
        # Try prefix match (e.g. "gpt-4.1-mini-2025-04-14" -> "gpt-4.1-mini")
        for key in sorted(MODEL_PRICING, key=len, reverse=True):
            if model.startswith(key):
                return MODEL_PRICING[key]
        return _DEFAULT_PRICING

    def cost(self, input_tokens: int, cached_tokens: int, output_tokens: int) -> float:
        """Return dollar cost for a single query."""
        p = self._pricing
        uncached = input_tokens - cached_tokens
        return (uncached * p.input_per_m + cached_tokens * p.cached_per_m + output_tokens * p.output_per_m) / 1_000_000

    def format_pricing_line(self) -> str:
        p = self._pricing
        return f"input=${p.input_per_m}  cached=${p.cached_per_m}  output=${p.output_per_m}"
