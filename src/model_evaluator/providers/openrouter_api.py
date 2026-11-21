"""Open-weight models through OpenRouter's OpenAI-compatible Chat Completions API.

Exposes the shared provider interface (``MODEL_PRICING``, ``chat()``, ``text_from_response()``, ``stop_reason()``,
``usage_summary()``, ``user_content()``), plus ``configured()``, which says whether the API key is set.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Sequence
from typing import Any

from openai import OpenAI

DEFAULT_MODEL = "deepseek/deepseek-v4.1-flash-1017"
BASE_URL = "https://openrouter.ai/api/v1"
API_KEY_ENV = "OPENROUTER_API_KEY"

# USD per million tokens, from https://openrouter.ai/api/v1/models (checked 2026-11-15). Estimates only: each response
# reports what OpenRouter actually charged (usage.cost), and usage_summary uses that as the total when it's there.
# A cache_read of None means OpenRouter lists no cache price, so cached tokens are estimated at the input price.
MODEL_PRICING: dict[str, dict[str, float | None]] = {
    "deepseek/deepseek-v4.1-flash-1017": {"input": 0.02, "output": 0.108, "cache_read": 0.007},
    "google/gemma-4-31b-it": {"input": 0.20, "output": 0.40, "cache_read": None},
    "openai/gpt-oss-120b": {"input": 0.15, "output": 0.60, "cache_read": 0.075},
    "qwen/qwen3.8-27b": {"input": 0.15, "output": 1.875, "cache_read": 0.0375},
    "mistralai/mistral-large-4-0": {"input": 0.70, "output": 2.10, "cache_read": 0.07},
    "moonshotai/kimi-k3.1-pro": {"input": 2.00, "output": 10.00, "cache_read": 0.20},
}

# PDFs sent to a model that can't read files natively are converted by OpenRouter's file-parser plugin, whose default
# engine is paid OCR (mistral-ocr, $2 per 1,000 pages). cloudflare-ai converts them to text for free.
PDF_ENGINE = "cloudflare-ai"

_client: OpenAI | None = None


def configured() -> bool:
    """Whether the API key is set. ``default_models()`` leaves this provider out until it is."""
    return bool(os.environ.get(API_KEY_ENV))


def _get_client() -> OpenAI:
    # Created on first use, so importing this module works before the key is set.
    # No HTTP-Referer / X-Title headers: they're optional and would identify the project to OpenRouter.
    global _client
    if _client is None:
        if not configured():
            raise RuntimeError(f"{API_KEY_ENV} isn't set: add it to .env to use OpenRouter models")
        _client = OpenAI(api_key=os.environ[API_KEY_ENV], base_url=BASE_URL)
    return _client


def list_models() -> list[str]:
    """Return the ids of every model OpenRouter offers."""
    return [model.id for model in _get_client().models.list()]


# ---------- Building conversations ----------


def add_user_message(messages: list[dict], content: str | list) -> None:
    messages.append({"role": "user", "content": content})


def add_assistant_message(messages: list[dict], content: str | list) -> None:
    messages.append({"role": "assistant", "content": content})


def user_content(text: str, attachments: Sequence[dict] = ()) -> str | list[dict]:
    """Turn a prompt and provider-neutral attachments into Chat Completions content parts.

    Images only work with vision models (OpenRouter returns an error otherwise). An attachment's ``"cache"`` flag is
    ignored: most open-model providers on OpenRouter cache repeated prompt prefixes by themselves.
    """
    if not attachments:
        return text
    parts: list[dict[str, Any]] = []
    for a in attachments:
        if a["type"] == "image":
            parts.append({"type": "image_url", "image_url": {"url": f"data:{a['media_type']};base64,{a['data']}"}})
        elif a["type"] == "pdf":
            parts.append(
                {
                    "type": "file",
                    "file": {"filename": a["name"], "file_data": f"data:application/pdf;base64,{a['data']}"},
                }
            )
        elif a["type"] == "text":
            parts.append({"type": "text", "text": f"File: {a['name']}\n\n{a['data']}"})
        else:
            raise ValueError(f"Unsupported attachment type: {a['type']}")
    if text:
        parts.append({"type": "text", "text": text})
    return parts


def _has_pdf(messages: list[dict]) -> bool:
    return any(
        isinstance(m.get("content"), list)
        and any(isinstance(p, dict) and p.get("type") == "file" for p in m["content"])
        for m in messages
        if isinstance(m, dict)
    )


# ---------- Reading responses ----------


def text_from_response(response: Any) -> str:
    return (response.choices[0].message.content or "") if response.choices else ""


def stop_reason(response: Any) -> str | None:
    """OpenRouter's normalised ``finish_reason``: ``"stop"`` is a normal finish, ``"length"`` means it was cut off."""
    return response.choices[0].finish_reason if response.choices else None


def _check_error(response: Any) -> None:
    # OpenRouter can report a failure inside an HTTP 200 body, e.g. when a provider fails after the response started
    error = getattr(response, "error", None)
    if error:
        message = error.get("message") if isinstance(error, dict) else str(error)
        raise RuntimeError(f"OpenRouter error: {message}")


# ---------- Usage and cost ----------


def _token_counts(usage: Any) -> dict[str, int]:
    prompt = usage.prompt_tokens or 0
    details = usage.prompt_tokens_details
    cache_read = (getattr(details, "cached_tokens", 0) or 0) if details else 0
    cache_write = (getattr(details, "cache_write_tokens", 0) or 0) if details else 0
    out_details = usage.completion_tokens_details
    return {
        # prompt_tokens includes cached tokens and cache writes; report uncached input to match the other modules
        "input_tokens": max(prompt - cache_read - cache_write, 0),
        # Includes reasoning tokens, which are billed as output
        "output_tokens": usage.completion_tokens or 0,
        "cache_read_input_tokens": cache_read,
        "cache_creation_input_tokens": cache_write,
        "reasoning_tokens": (getattr(out_details, "reasoning_tokens", 0) or 0) if out_details else 0,
    }


def estimate_cost(model: str, usage: Any) -> dict[str, float | None] | None:
    """Cost breakdown from ``MODEL_PRICING``; ``total_cost`` is OpenRouter's reported charge when there is one."""
    pricing = MODEL_PRICING.get(model)
    reported = getattr(usage, "cost", None)
    reported = reported if isinstance(reported, (int, float)) and not isinstance(reported, bool) else None
    if pricing is None:
        if reported is None:
            print(f"WARNING: no pricing for model '{model}', cost not estimated")
            return None
        return {
            "input_cost": 0.0,
            "output_cost": 0.0,
            "cache_write_cost": 0.0,
            "cache_read_cost": 0.0,
            "total_cost": reported,
            "estimated_total_cost": None,
        }

    counts = _token_counts(usage)
    per_token = 1_000_000
    cache_read_price = pricing["cache_read"] if pricing["cache_read"] is not None else pricing["input"]
    input_cost = counts["input_tokens"] * pricing["input"] / per_token
    output_cost = counts["output_tokens"] * pricing["output"] / per_token
    cache_read_cost = counts["cache_read_input_tokens"] * cache_read_price / per_token
    # OpenRouter lists no cache-write price for these models, so writes are estimated at the input price
    cache_write_cost = counts["cache_creation_input_tokens"] * pricing["input"] / per_token
    estimated = input_cost + output_cost + cache_write_cost + cache_read_cost
    return {
        "input_cost": input_cost,
        "output_cost": output_cost,
        "cache_write_cost": cache_write_cost,
        "cache_read_cost": cache_read_cost,
        "total_cost": reported if reported is not None else estimated,
        # Kept so the estimate can be checked against what OpenRouter charged
        "estimated_total_cost": estimated,
    }


def usage_summary(model: str, usage: Any) -> dict[str, Any]:
    """Token counts in the shape shared by every provider, plus ``reasoning_tokens`` and the ``cost``."""
    summary: dict[str, Any] = _token_counts(usage)
    summary["total_tokens"] = (
        summary["input_tokens"]
        + summary["output_tokens"]
        + summary["cache_read_input_tokens"]
        + summary["cache_creation_input_tokens"]
    )
    summary["cost"] = estimate_cost(model, usage)
    return summary


# ---------- Calling the API ----------


def chat(
    messages: list[dict],
    model: str = DEFAULT_MODEL,
    system: str | None = None,
    temperature: float = 1.0,
    stream: bool = True,
    output_config: dict | None = None,
    tools: Iterable[dict] | None = None,
    return_full_response: bool = False,
    return_usage: bool = False,
    max_tokens: int = 4000,
) -> Any:
    """Send a conversation to an OpenRouter model and return the reply.

    ``system`` becomes a system message, and an Anthropic-style ``output_config`` becomes a JSON-schema
    ``response_format`` that only providers supporting it may serve. ``temperature`` is accepted but not sent.
    """
    params: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": ([{"role": "system", "content": system}] if system else []) + list(messages),
    }
    extra_body: dict[str, Any] = {}

    if output_config:
        # require_parameters makes OpenRouter use only providers that support the schema, rather than silently
        # ignoring it
        fmt = output_config["format"]
        params["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "output", "strict": True, "schema": fmt["schema"]},
        }
        extra_body["provider"] = {"require_parameters": True}

    if _has_pdf(params["messages"]):
        extra_body["plugins"] = [{"id": "file-parser", "pdf": {"engine": PDF_ENGINE}}]

    if tools:
        params["tools"] = list(tools)
    if extra_body:
        params["extra_body"] = extra_body

    client = _get_client()
    if stream:
        with client.chat.completions.stream(**params) as chat_stream:
            # One pass over the events, as in the other modules
            for event in chat_stream:
                if event.type == "content.delta":
                    print(event.delta, end="", flush=True)
                elif event.type == "tool_calls.function.arguments.delta":
                    print(event.arguments_delta, end="", flush=True)
            print()
            final = chat_stream.get_final_completion()
    else:
        final = client.chat.completions.create(**params)
    _check_error(final)

    result = final if return_full_response else text_from_response(final)
    if return_usage:
        if final.usage is None:
            raise RuntimeError("OpenRouter returned no usage for this response")
        return result, usage_summary(model, final.usage)
    return result
