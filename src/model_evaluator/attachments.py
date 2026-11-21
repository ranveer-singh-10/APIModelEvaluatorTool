"""Validation and preparation of file attachments.

Attachments travel in one provider-neutral shape, ``{"type", "name", "media_type", "data"}``, and each provider's
``user_content()`` turns them into its own content blocks. Images and PDFs are base64; text files are plain text.
"""

from __future__ import annotations

import base64
import binascii
from collections.abc import Iterable
from types import ModuleType

ATTACHMENT_MEDIA_TYPES: dict[str, set[str]] = {
    "image": {"image/png", "image/jpeg", "image/gif", "image/webp"},
    "pdf": {"application/pdf"},
    "text": {"text/plain"},
}
# Bytes per file, and for all attachments sent in one request. Every turn resends the conversation's attachments,
# and Anthropic rejects requests over 32 MB (base64 adds a third), so the total stays well under that.
ATTACHMENT_MAX_BYTES = {"image": 5_000_000, "pdf": 10_000_000, "text": 1_000_000}
ATTACHMENTS_MAX_TOTAL_BYTES = 20_000_000


def check_attachments(attachments: list[dict]) -> list[dict]:
    """Validate attachments sent by the chat page and return each one's :func:`attachment_summary`.

    The summaries are worked out while checking, so callers can add up the bytes and log them without decoding the
    data again. Raises ``ValueError`` for an unsupported type, bad base64, or a file that is empty or too large.
    """
    if not isinstance(attachments, list):
        raise ValueError("attachments must be a list")
    summaries = []
    for a in attachments:
        if (
            not isinstance(a, dict)
            or set(a) != {"type", "name", "media_type", "data"}
            or a["type"] not in ATTACHMENT_MEDIA_TYPES
            or a["media_type"] not in ATTACHMENT_MEDIA_TYPES[a["type"]]
            or not isinstance(a["name"], str)
            or not 0 < len(a["name"]) <= 255
            or not isinstance(a["data"], str)
        ):
            raise ValueError("each attachment needs a supported type and media_type, a name and its data")
        if a["type"] == "text":
            size = len(a["data"].encode("utf-8"))
        else:
            try:
                size = len(base64.b64decode(a["data"], validate=True))
            except (binascii.Error, ValueError):
                raise ValueError(f"attachment {a['name']} is not valid base64") from None
        limit = ATTACHMENT_MAX_BYTES[a["type"]]
        if size == 0 or size > limit:
            raise ValueError(f"attachment {a['name']} is empty or over the {limit // 1_000_000} MB limit")
        summaries.append({"name": a["name"], "type": a["type"], "media_type": a["media_type"], "bytes": size})
    return summaries


def total_bytes(summaries: Iterable[dict]) -> int:
    return sum(s["bytes"] for s in summaries)


def attachment_summary(attachments: Iterable[dict]) -> list[dict]:
    """What the log records about each attachment: never the data, which would bloat the log."""
    return [
        {
            "name": a["name"],
            "type": a["type"],
            "media_type": a["media_type"],
            "bytes": len(a["data"].encode("utf-8")) if a["type"] == "text" else len(base64.b64decode(a["data"])),
        }
        for a in attachments
    ]


def with_context_files(messages: list[dict], context_files: list[dict]) -> list[dict]:
    """Put conversation-wide context files at the start of the first user message.

    They go ahead of that turn's own attachments, so every turn resends them. (The system prompt can't hold them: the
    APIs take only text there.) The last one is marked ``"cache"``: they start every request unchanged, so a provider
    that needs a cache breakpoint (Claude) puts one there and later turns pay the cache-read rate for them.
    """
    if not context_files or not messages:
        return messages
    context = [*context_files[:-1], {**context_files[-1], "cache": True}]
    first = messages[0]
    return [{**first, "attachments": context + list(first.get("attachments") or [])}, *messages[1:]]


def provider_messages(provider: ModuleType, messages: list) -> list:
    """Convert each message that has ``attachments`` into ``provider``'s content blocks; pass others on unchanged."""
    converted = []
    for m in messages:
        if isinstance(m, dict) and "attachments" in m:
            attachments = m["attachments"]
            m = {k: v for k, v in m.items() if k != "attachments"}
            if attachments:
                m["content"] = provider.user_content(m["content"], attachments)
        converted.append(m)
    return converted
