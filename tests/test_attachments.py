import base64

import pytest

from model_evaluator.attachments import (
    ATTACHMENT_MAX_BYTES,
    attachment_summary,
    check_attachments,
    provider_messages,
    with_context_files,
)
from model_evaluator.providers import anthropic_api


def image(data: bytes = b"\x89PNG fake", name: str = "a.png"):
    return {"type": "image", "name": name, "media_type": "image/png", "data": base64.b64encode(data).decode()}


def text(content: str = "hello", name: str = "notes.txt"):
    return {"type": "text", "name": name, "media_type": "text/plain", "data": content}


def test_check_attachments_returns_summaries():
    summaries = check_attachments([image(b"12345"), text("héllo")])
    assert summaries == [
        {"name": "a.png", "type": "image", "media_type": "image/png", "bytes": 5},
        {"name": "notes.txt", "type": "text", "media_type": "text/plain", "bytes": 6},
    ]
    assert attachment_summary([image(b"12345"), text("héllo")]) == summaries


@pytest.mark.parametrize(
    "attachment",
    [
        {**image(), "media_type": "image/svg+xml"},  # unsupported media type
        {**image(), "extra": 1},  # unexpected key
        {**image(), "name": ""},  # no name
        {**image(), "data": "not base64!"},  # bad base64
        text(""),  # empty
        text("x" * (ATTACHMENT_MAX_BYTES["text"] + 1)),  # too large
    ],
)
def test_check_attachments_rejects(attachment):
    with pytest.raises(ValueError):
        check_attachments([attachment])


def test_check_attachments_needs_a_list():
    with pytest.raises(ValueError):
        check_attachments(image())


def test_with_context_files_goes_first_and_caches_the_last():
    messages = [{"role": "user", "content": "q", "attachments": [image()]}, {"role": "assistant", "content": "a"}]
    ctx = [text("one", "1.txt"), text("two", "2.txt")]

    combined = with_context_files(messages, ctx)

    names = [a["name"] for a in combined[0]["attachments"]]
    assert names == ["1.txt", "2.txt", "a.png"]
    assert "cache" not in combined[0]["attachments"][0]
    assert combined[0]["attachments"][1]["cache"] is True
    # The input isn't changed
    assert "cache" not in ctx[1] and len(messages[0]["attachments"]) == 1


def test_with_context_files_without_files_is_unchanged():
    messages = [{"role": "user", "content": "q"}]
    assert with_context_files(messages, []) is messages


def test_provider_messages_converts_only_messages_with_attachments():
    messages = [
        {"role": "user", "content": "q", "attachments": [text()]},
        {"role": "assistant", "content": "a"},
        {"role": "user", "content": "plain", "attachments": []},
    ]
    converted = provider_messages(anthropic_api, messages)
    assert converted[0]["content"][0]["type"] == "document"
    assert converted[1] == {"role": "assistant", "content": "a"}
    assert converted[2] == {"role": "user", "content": "plain"}
