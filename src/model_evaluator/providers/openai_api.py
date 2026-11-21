"""OpenAI models through the Responses API.

Exposes the shared provider interface used by the evaluation code: ``MODEL_PRICING``, ``chat()``,
``text_from_response()``, ``stop_reason()``, ``usage_summary()`` and ``user_content()``.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

from openai import OpenAI
from openai.types.responses import Response

DEFAULT_MODEL = "gpt-6-luna"

# USD per million tokens, standard tier. Estimates only, from https://developers.openai.com/api/docs/pricing
# (checked 2026-11-13)
MODEL_PRICING: dict[str, dict[str, float]] = {
    "gpt-6-astra": {"input": 10.00, "output": 50.00, "cache_read": 1.00},
    "gpt-6.1-sol": {"input": 2.00, "output": 10.00, "cache_read": 0.10},
    "gpt-6-sol": {"input": 2.00, "output": 10.00, "cache_read": 0.20},
    "gpt-6.1-terra": {"input": 1.00, "output": 5.00, "cache_read": 0.05},
    "gpt-6-luna": {"input": 0.10, "output": 0.50, "cache_read": 0.01},
}

# Every GPT-6 model charges more once the prompt goes over 272K tokens
LONG_CONTEXT_THRESHOLD = 272_000
LONG_CONTEXT_PRICING: dict[str, dict[str, float]] = {
    "gpt-6-astra": {"input": 20.00, "output": 75.00, "cache_read": 2.00},
    "gpt-6.1-sol": {"input": 4.00, "output": 15.00, "cache_read": 0.20},
    "gpt-6-sol": {"input": 4.00, "output": 15.00, "cache_read": 0.40},
    "gpt-6.1-terra": {"input": 2.00, "output": 7.50, "cache_read": 0.10},
    "gpt-6-luna": {"input": 0.20, "output": 0.75, "cache_read": 0.02},
}

# The 30 minute cache rate. 12 hour writes cost 2x, but the usage doesn't say which TTL was used, so this
# assumes 30 minutes
CACHE_WRITE_MULTIPLIER = 1.25

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    """Return the shared client, created on first use so importing this module needs no API key."""
    global _client
    if _client is None:
        _client = OpenAI()
    return _client


def list_models() -> list[str]:
    """Return the ids of the models this API key can use."""
    return [model.id for model in _get_client().models.list()]


# ---------- Building conversations ----------
# The Responses API takes the same {"role", "content"} items as the Anthropic messages list


def add_user_message(messages: list[dict], content: str | list) -> None:
    messages.append({"role": "user", "content": content})


def add_assistant_message(messages: list[dict], content: str | list) -> None:
    messages.append({"role": "assistant", "content": content})


def user_content(text: str, attachments: Sequence[dict] = ()) -> str | list[dict]:
    """Turn a prompt and provider-neutral attachments into Responses API input parts.

    An attachment's ``"cache"`` flag (a Claude cache breakpoint) is ignored: OpenAI caches the start of a repeated
    prompt by itself.
    """
    if not attachments:
        return text
    parts: list[dict[str, Any]] = []
    for a in attachments:
        if a["type"] == "image":
            parts.append(
                {"type": "input_image", "detail": "auto", "image_url": f"data:{a['media_type']};base64,{a['data']}"}
            )
        elif a["type"] == "pdf":
            parts.append(
                {"type": "input_file", "filename": a["name"], "file_data": f"data:application/pdf;base64,{a['data']}"}
            )
        elif a["type"] == "text":
            # Text files go in as text, headed with the file name, since input_file is meant for documents like PDFs
            parts.append({"type": "input_text", "text": f"File: {a['name']}\n\n{a['data']}"})
        else:
            raise ValueError(f"Unsupported attachment type: {a['type']}")
    if text:
        parts.append({"type": "input_text", "text": text})
    return parts


def add_response_output(messages: list[dict], response: Response) -> None:
    """Copy every output item (reasoning, function calls, messages) so a tool-use loop keeps them in the history."""
    messages.extend(item.model_dump(exclude_none=True) for item in response.output)


def build_tool_result(call_id: str, output: str) -> dict:
    return {"type": "function_call_output", "call_id": call_id, "output": output}


# ---------- Reading responses ----------


def text_from_response(response: Response) -> str:
    return response.output_text


def stop_reason(response: Response) -> str | None:
    """``"completed"``, or why the response is incomplete (e.g. ``max_output_tokens``): OpenAI has no stop_reason."""
    if response.status == "incomplete" and response.incomplete_details:
        return response.incomplete_details.reason
    return response.status


# ---------- Usage and cost ----------


def _token_counts(usage: Any) -> dict[str, int]:
    input_details = usage.input_tokens_details
    output_details = usage.output_tokens_details
    cache_read_tokens = (input_details.cached_tokens or 0) if input_details else 0
    cache_write_tokens = (getattr(input_details, "cache_write_tokens", 0) or 0) if input_details else 0
    return {
        # OpenAI's input_tokens includes cached tokens; report uncached input to match the Anthropic numbers
        "input_tokens": max((usage.input_tokens or 0) - cache_read_tokens - cache_write_tokens, 0),
        # Includes reasoning tokens, which are billed as output
        "output_tokens": usage.output_tokens or 0,
        "cache_read_input_tokens": cache_read_tokens,
        "cache_creation_input_tokens": cache_write_tokens,
        "reasoning_tokens": (output_details.reasoning_tokens or 0) if output_details else 0,
    }


def estimate_cost(model: str, usage: Any) -> dict[str, float] | None:
    """Estimate a call's cost in USD from ``MODEL_PRICING``, or ``None`` for an unpriced model."""
    pricing = MODEL_PRICING.get(model)
    if pricing is None:
        print(f"WARNING: no pricing for model '{model}', cost not estimated")
        return None

    counts = _token_counts(usage)
    # OpenAI's input_tokens is the whole prompt, cached tokens and cache writes included
    long_context = LONG_CONTEXT_PRICING.get(model)
    if long_context and (usage.input_tokens or 0) > LONG_CONTEXT_THRESHOLD:
        pricing = long_context

    per_token = 1_000_000
    input_cost = counts["input_tokens"] * pricing["input"] / per_token
    output_cost = counts["output_tokens"] * pricing["output"] / per_token
    cache_read_cost = counts["cache_read_input_tokens"] * pricing["cache_read"] / per_token
    cache_write_cost = counts["cache_creation_input_tokens"] * pricing["input"] * CACHE_WRITE_MULTIPLIER / per_token

    return {
        "input_cost": input_cost,
        "output_cost": output_cost,
        "cache_write_cost": cache_write_cost,
        "cache_read_cost": cache_read_cost,
        "total_cost": input_cost + output_cost + cache_write_cost + cache_read_cost,
    }


def usage_summary(model: str, usage: Any) -> dict[str, Any]:
    """Token counts in the shape shared by every provider, plus ``reasoning_tokens`` and the estimated ``cost``."""
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
    """Send a conversation through the Responses API and return the reply.

    Mirrors ``anthropic_api.chat``: ``system`` is sent as ``instructions``, an Anthropic-style ``output_config``
    becomes ``text.format``, and ``temperature`` is accepted but not sent.
    """
    params: dict[str, Any] = {"model": model, "max_output_tokens": max_tokens, "input": messages}

    if system:
        params["instructions"] = system
    if output_config:
        # Accepts the Anthropic-style {"format": {"type": "json_schema", "schema": ...}}; OpenAI also wants a name
        params["text"] = {"format": {"name": "output", "strict": True, **output_config["format"]}}
    if tools:
        params["tools"] = list(tools)

    client = _get_client()
    if stream:
        with client.responses.stream(**params) as response_stream:
            # One pass over the events, as in anthropic_api.chat
            for event in response_stream:
                if event.type in ("response.output_text.delta", "response.function_call_arguments.delta"):
                    print(event.delta, end="", flush=True)
            print()
            final = response_stream.get_final_response()
    else:
        final = client.responses.create(**params)

    result = final if return_full_response else text_from_response(final)
    if return_usage:
        return result, usage_summary(model, final.usage)
    return result
