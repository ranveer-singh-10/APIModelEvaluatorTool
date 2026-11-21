"""Shared fixtures. No test makes a real API call: model calls are replaced with fakes."""

from __future__ import annotations

from typing import Any

import pytest


def make_result(
    model: str = "claude-haiku-5-5",
    response: str = "Canberra",
    cost: float | None = 0.001,
    error: str | None = None,
    provider: str = "anthropic",
    seconds: float = 0.5,
) -> dict[str, Any]:
    """A result dict shaped like ``run_model()``'s."""
    if error:
        return {
            "model": model,
            "provider": provider,
            "seconds": seconds,
            "stop_reason": None,
            "response": "",
            "usage": None,
            "error": error,
        }
    usage = {
        "input_tokens": 10,
        "output_tokens": 5,
        "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": 0,
        "total_tokens": 15,
        "cost": None if cost is None else {"total_cost": cost},
    }
    return {
        "model": model,
        "provider": provider,
        "seconds": seconds,
        "stop_reason": "end_turn",
        "response": response,
        "usage": usage,
        "error": None,
    }


@pytest.fixture
def log_file(tmp_path):
    return tmp_path / "logs" / "eval_log.json"


@pytest.fixture
def no_openrouter_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
