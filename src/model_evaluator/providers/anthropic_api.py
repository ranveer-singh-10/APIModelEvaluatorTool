"""Claude models through the Anthropic Messages API.

Exposes the shared provider interface used by the evaluation code: ``MODEL_PRICING``, ``chat()``,
``text_from_response()``, ``stop_reason()``, ``usage_summary()`` and ``user_content()``.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

from anthropic import Anthropic
from anthropic.types import Message

DEFAULT_MODEL = "claude-haiku-5-5"

# USD per million tokens. Estimates only - check against https://claude.com/pricing
MODEL_PRICING: dict[str, dict[str, float]] = {
    "claude-fable-5-1": {"input": 10.00, "output": 50.00, "cache_read": 0.25},
    "claude-opus-5-5": {"input": 4.00, "output": 20.00, "cache_read": 0.20},
    "claude-opus-5": {"input": 5.00, "output": 25.00, "cache_read": 0.50},
    "claude-sonnet-5-5": {"input": 2.00, "output": 10.00, "cache_read": 0.20},
    "claude-sonnet-5": {"input": 2.00, "output": 10.00, "cache_read": 0.20},
    "claude-sonnet-4-6": {"input": 3.00, "output": 15.00, "cache_read": 0.30},
    "claude-haiku-5-5": {"input": 0.10, "output": 0.50, "cache_read": 0.01},
    "claude-haiku-4-5": {"input": 1.00, "output": 5.00, "cache_read": 0.10},
}

# Haiku 5.5 charges more once the prompt goes over 100K tokens
LONG_CONTEXT_PRICING: dict[str, dict[str, float]] = {
    "claude-haiku-5-5": {"threshold": 100_000, "input": 0.50, "output": 2.50, "cache_read": 0.05},
}

CACHE_WRITE_5M_MULTIPLIER = 1.25
CACHE_WRITE_1H_MULTIPLIER = 2.0

_client: Anthropic | None = None


def _get_client() -> Anthropic:
    """Return the shared client, created on first use so importing this module needs no API key."""
    global _client
    if _client is None:
        _client = Anthropic()
    return _client


def list_models() -> list[str]:
    """Return the ids of the models this API key can use."""
    return [model.id for model in _get_client().models.list()]


# ---------- Building conversations ----------


def add_user_message(messages: list[dict], content: str | Message | list) -> None:
    """Append a user turn. A ``Message`` has its content blocks copied, which keeps tool results intact."""
    messages.append({"role": "user", "content": content.content if isinstance(content, Message) else content})


def add_assistant_message(messages: list[dict], content: str | Message | list) -> None:
    """Append an assistant turn. Pass the full ``Message`` in tool-use loops so its ``tool_use`` blocks are kept."""
    messages.append({"role": "assistant", "content": content.content if isinstance(content, Message) else content})


def user_content(text: str, attachments: Sequence[dict] = ()) -> str | list[dict]:
    """Turn a prompt and provider-neutral attachments into Claude content blocks.

    Each attachment is ``{"type": "image" | "pdf" | "text", "name", "media_type", "data"}``: base64 for images and
    PDFs, plain text for text files. ``"cache": True`` puts a prompt-cache breakpoint on that block.
    """
    if not attachments:
        return text
    blocks: list[dict[str, Any]] = []
    for a in attachments:
        if a["type"] == "image":
            blocks.append(
                {"type": "image", "source": {"type": "base64", "media_type": a["media_type"], "data": a["data"]}}
            )
        elif a["type"] == "pdf":
            blocks.append(
                {
                    "type": "document",
                    "title": a["name"],
                    "source": {"type": "base64", "media_type": "application/pdf", "data": a["data"]},
                }
            )
        elif a["type"] == "text":
            blocks.append(
                {
                    "type": "document",
                    "title": a["name"],
                    "source": {"type": "text", "media_type": "text/plain", "data": a["data"]},
                }
            )
        else:
            raise ValueError(f"Unsupported attachment type: {a['type']}")
        # A cache breakpoint: everything up to and including this block is cached, so later requests that start
        # the same way (e.g. the chat's context files) pay the cache-read rate for it
        if a.get("cache"):
            blocks[-1]["cache_control"] = {"type": "ephemeral"}
    # Files first, then the question about them, which is the order Claude's docs recommend
    if text:
        blocks.append({"type": "text", "text": text})
    return blocks


def tool_results(response: Message, outputs: dict[str, str], is_error: bool = False) -> list[dict]:
    """Build the ``tool_result`` blocks answering every tool call in ``response``.

    ``outputs`` maps each ``tool_use`` block's id to the text to send back.
    """
    return [
        {"type": "tool_result", "tool_use_id": block.id, "content": outputs[block.id], "is_error": is_error}
        for block in response.content
        if block.type == "tool_use"
    ]


# ---------- Reading responses ----------


def text_from_response(response: Message) -> str:
    """Join the text blocks of a response, one per line."""
    return "\n".join(block.text for block in response.content if block.type == "text")


def stop_reason(response: Message) -> str | None:
    return response.stop_reason


def show_web_search_response(response: Message) -> None:
    """Print the searches, results, text and citations of a response that used the web search tool."""
    print(f"stop_reason: {response.stop_reason}")
    print(f"Searches used: {response.usage.server_tool_use.web_search_requests}\n")

    for block in response.content:
        if block.type == "server_tool_use":
            print(f"SERVER TOOL: {block.name}")
            print(f"  input: {block.input}")
            query = block.input.get("query") if isinstance(block.input, dict) else None
            if query:
                print(f"SEARCH QUERY: {query}")
        elif block.type == "web_search_tool_result":
            if isinstance(block.content, list):
                for result in block.content:
                    print(f"  RESULT: {result.title}\n          {result.url}")
            else:
                print(f"  SEARCH ERROR: {block.content.error_code}")
        elif block.type == "text":
            print(f"\nTEXT: {block.text}")
            for c in block.citations or []:
                print(f"   CITED: {c.title} ({c.url})")
                print(f'          "{c.cited_text[:100]}..."')
        print()


# ---------- Usage and cost ----------


def estimate_cost(model: str, usage: Any) -> dict[str, float] | None:
    """Estimate a call's cost in USD from ``MODEL_PRICING``, or ``None`` for an unpriced model."""
    pricing = MODEL_PRICING.get(model)
    if pricing is None:
        print(f"WARNING: no pricing for model '{model}', cost not estimated")
        return None

    input_tokens = usage.input_tokens or 0
    output_tokens = usage.output_tokens or 0
    cache_read_tokens = usage.cache_read_input_tokens or 0
    cache_write_tokens = usage.cache_creation_input_tokens or 0

    long_context = LONG_CONTEXT_PRICING.get(model)
    prompt_tokens = input_tokens + cache_read_tokens + cache_write_tokens
    if long_context and prompt_tokens > long_context["threshold"]:
        pricing = long_context

    # Split cache writes by TTL when the API reports it, otherwise assume the 5 minute TTL
    cache_creation = getattr(usage, "cache_creation", None)
    if cache_creation is not None:
        write_5m = cache_creation.ephemeral_5m_input_tokens or 0
        write_1h = cache_creation.ephemeral_1h_input_tokens or 0
    else:
        write_5m, write_1h = cache_write_tokens, 0

    per_token = 1_000_000
    input_cost = input_tokens * pricing["input"] / per_token
    output_cost = output_tokens * pricing["output"] / per_token
    cache_read_cost = cache_read_tokens * pricing["cache_read"] / per_token
    cache_write_cost = (
        write_5m * pricing["input"] * CACHE_WRITE_5M_MULTIPLIER
        + write_1h * pricing["input"] * CACHE_WRITE_1H_MULTIPLIER
    ) / per_token

    return {
        "input_cost": input_cost,
        "output_cost": output_cost,
        "cache_write_cost": cache_write_cost,
        "cache_read_cost": cache_read_cost,
        "total_cost": input_cost + output_cost + cache_write_cost + cache_read_cost,
    }


def usage_summary(model: str, usage: Any) -> dict[str, Any]:
    """Token counts in the shape shared by every provider, plus the estimated ``cost``."""
    summary: dict[str, Any] = {
        "input_tokens": usage.input_tokens or 0,
        "output_tokens": usage.output_tokens or 0,
        "cache_read_input_tokens": usage.cache_read_input_tokens or 0,
        "cache_creation_input_tokens": usage.cache_creation_input_tokens or 0,
    }
    summary["total_tokens"] = sum(summary.values())
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
    fine_grained: bool = False,
    return_usage: bool = False,
    max_tokens: int = 4000,
) -> Any:
    """Send a conversation to Claude and return the reply.

    Returns the reply text, or the full ``Message`` with ``return_full_response=True`` (needed for tool use and web
    search). With ``return_usage=True`` it returns ``(result, usage_summary)``. ``temperature`` is accepted for parity
    with the other providers but not sent: API version 2026-10-21 rejects sampling parameters with a 400.
    """
    params: dict[str, Any] = {"model": model, "max_tokens": max_tokens, "messages": messages}

    if output_config:
        params["output_config"] = output_config
    if system:
        params["system"] = system
    if tools:
        if fine_grained:
            tools = [{**tool, "eager_input_streaming": True} if "input_schema" in tool else tool for tool in tools]
        params["tools"] = list(tools)

    client = _get_client()
    if stream:
        with client.messages.stream(**params) as message_stream:
            # One pass over the events: a second loop (e.g. text_stream) would find the stream already used up
            for event in message_stream:
                if event.type == "text":
                    print(event.text, end="", flush=True)
                elif event.type == "input_json":
                    print(event.partial_json, end="", flush=True)
            print()
            final = message_stream.get_final_message()
    else:
        final = client.messages.create(**params)

    result = final if return_full_response else "".join(b.text for b in final.content if b.type == "text")
    if return_usage:
        return result, usage_summary(model, final.usage)
    return result
