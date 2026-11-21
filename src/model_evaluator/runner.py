"""Running one model call and recording the result in the shape every evaluation logs."""

from __future__ import annotations

import time
from typing import Any

from .attachments import provider_messages
from .providers import PROVIDERS, provider_for

# A result dict: {model, provider, seconds, stop_reason, response, usage, error}, plus "score" once scored
Result = dict[str, Any]


def run_model(model: str, messages: list, system: str | None = None, output_config: dict | None = None) -> Result:
    """Make one non-streaming call through the provider's ``chat()`` and return the fields every evaluation records.

    ``output_config`` (Anthropic-style structured output) is translated by each provider; the grader uses it.
    """
    provider_name = provider_for(model)
    provider = PROVIDERS[provider_name]
    messages = provider_messages(provider, messages)
    start = time.perf_counter()
    response, usage = provider.chat(
        messages,
        model=model,
        system=system,
        stream=False,
        return_full_response=True,
        return_usage=True,
        output_config=output_config,
    )
    return {
        "model": model,
        "provider": provider_name,
        "seconds": time.perf_counter() - start,
        "stop_reason": provider.stop_reason(response),
        "response": provider.text_from_response(response),
        "usage": usage,
        "error": None,
    }


def run_model_safe(model: str, messages: list, system: str | None = None, output_config: dict | None = None) -> Result:
    """Like :func:`run_model`, but an error becomes a result, so one unavailable model doesn't stop a comparison."""
    start = time.perf_counter()
    try:
        return run_model(model, messages, system=system, output_config=output_config)
    except Exception as e:  # noqa: BLE001 - any failure is reported in the result
        try:
            provider_name: str | None = provider_for(model)
        except ValueError:
            provider_name = None
        return {
            "model": model,
            "provider": provider_name,
            "seconds": time.perf_counter() - start,
            "stop_reason": None,
            "response": "",
            "usage": None,
            "error": f"{type(e).__name__}: {e}",
        }


def call_cost(result: Result) -> float | None:
    """The total cost of a result's call, or ``None`` if it failed or the model is unpriced."""
    usage = result.get("usage")
    return usage["cost"]["total_cost"] if usage and usage.get("cost") else None
