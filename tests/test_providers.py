from types import SimpleNamespace as NS

import pytest
from openai import ContentFilterFinishReasonError, LengthFinishReasonError
from openai._models import construct_type
from openai.types.chat import ChatCompletion

from model_evaluator.providers import (
    PROVIDERS,
    anthropic_api,
    default_models,
    openai_api,
    openrouter_api,
    provider_for,
)

REQUIRED_INTERFACE = ("MODEL_PRICING", "chat", "text_from_response", "stop_reason", "usage_summary", "user_content")


@pytest.mark.parametrize("name", PROVIDERS)
def test_every_provider_has_the_shared_interface(name):
    for attr in REQUIRED_INTERFACE:
        assert hasattr(PROVIDERS[name], attr), f"{name} is missing {attr}"


@pytest.mark.parametrize(
    ("model", "provider"),
    [
        ("claude-haiku-5-5", "anthropic"),
        ("gpt-6-luna", "openai"),
        ("deepseek/deepseek-v4.1-flash-1017", "openrouter"),
        # Unpriced models fall back to the name
        ("claude-some-future-model", "anthropic"),
        ("o4-mini", "openai"),
        # A "/" means OpenRouter even when the vendor is OpenAI
        ("openai/gpt-oss-120b", "openrouter"),
        ("openai/some-unpriced-model", "openrouter"),
    ],
)
def test_provider_for(model, provider):
    assert provider_for(model) == provider


def test_provider_for_unknown_model():
    with pytest.raises(ValueError, match="Can't tell"):
        provider_for("llama-local")


def test_default_models_leaves_out_unconfigured_openrouter(no_openrouter_key):
    models = default_models()
    assert "claude-haiku-5-5" in models
    assert not any("/" in m for m in models)


def test_default_models_includes_openrouter_once_configured(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    assert "openai/gpt-oss-120b" in default_models()


def test_default_models_named_provider_is_included_even_unconfigured(no_openrouter_key):
    assert default_models(["openrouter"]) == list(openrouter_api.MODEL_PRICING)


def test_default_models_unknown_provider():
    with pytest.raises(ValueError, match="Unknown provider"):
        default_models(["nope"])


def test_openrouter_without_key_fails_clearly(no_openrouter_key, monkeypatch):
    monkeypatch.setattr(openrouter_api, "_client", None)
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        openrouter_api.chat([{"role": "user", "content": "hi"}])


# ---------- user_content ----------

IMAGE = {"type": "image", "name": "a.png", "media_type": "image/png", "data": "aGk="}
PDF = {"type": "pdf", "name": "a.pdf", "media_type": "application/pdf", "data": "aGk="}
TEXT = {"type": "text", "name": "notes.txt", "media_type": "text/plain", "data": "hello"}


@pytest.mark.parametrize("module", [anthropic_api, openai_api, openrouter_api])
def test_user_content_without_attachments_is_plain_text(module):
    assert module.user_content("hi") == "hi"


def test_anthropic_user_content_puts_files_first_and_marks_cache():
    blocks = anthropic_api.user_content("question", [IMAGE, {**TEXT, "cache": True}])
    assert [b["type"] for b in blocks] == ["image", "document", "text"]
    assert "cache_control" not in blocks[0]
    assert blocks[1]["cache_control"] == {"type": "ephemeral"}
    assert blocks[-1]["text"] == "question"


def test_openai_user_content_types():
    parts = openai_api.user_content("q", [IMAGE, PDF, TEXT])
    assert [p["type"] for p in parts] == ["input_image", "input_file", "input_text", "input_text"]
    assert parts[0]["image_url"].startswith("data:image/png;base64,")


def test_openrouter_user_content_types():
    parts = openrouter_api.user_content("q", [IMAGE, PDF, TEXT])
    assert [p["type"] for p in parts] == ["image_url", "file", "text", "text"]


@pytest.mark.parametrize("module", [anthropic_api, openai_api, openrouter_api])
def test_user_content_rejects_unknown_type(module):
    with pytest.raises(ValueError, match="Unsupported"):
        module.user_content("q", [{"type": "video", "name": "v", "media_type": "video/mp4", "data": ""}])


# ---------- Cost estimates ----------


def anthropic_usage(input_tokens=0, output_tokens=0, cache_read=0, cache_write=0, cache_creation=None):
    return NS(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_input_tokens=cache_read,
        cache_creation_input_tokens=cache_write,
        cache_creation=cache_creation,
    )


def test_anthropic_cost():
    # claude-sonnet-5-5: $2 in, $10 out, $0.20 cache read per million
    cost = anthropic_api.estimate_cost("claude-sonnet-5-5", anthropic_usage(1_000_000, 100_000, 500_000))
    assert cost["input_cost"] == pytest.approx(2.0)
    assert cost["output_cost"] == pytest.approx(1.0)
    assert cost["cache_read_cost"] == pytest.approx(0.1)
    assert cost["total_cost"] == pytest.approx(3.1)


def test_anthropic_cache_writes_by_ttl():
    split = NS(ephemeral_5m_input_tokens=1_000_000, ephemeral_1h_input_tokens=1_000_000)
    usage = anthropic_usage(cache_write=2_000_000, cache_creation=split)
    # $2 input: 5 minute writes at 1.25x, 1 hour writes at 2x
    assert anthropic_api.estimate_cost("claude-sonnet-5-5", usage)["cache_write_cost"] == pytest.approx(2.5 + 4.0)


def test_anthropic_long_context_pricing():
    # Over 100K prompt tokens Haiku 5.5 charges $0.50 instead of $0.10 per million input tokens
    cost = anthropic_api.estimate_cost("claude-haiku-5-5", anthropic_usage(200_000))
    assert cost["input_cost"] == pytest.approx(0.1)


def test_anthropic_unpriced_model():
    assert anthropic_api.estimate_cost("claude-unknown", anthropic_usage(10)) is None


def test_anthropic_usage_summary_totals():
    summary = anthropic_api.usage_summary("claude-haiku-5-5", anthropic_usage(10, 20, 30, 40))
    assert summary["total_tokens"] == 100
    assert summary["cost"]["total_cost"] > 0


def openai_usage(input_tokens, output_tokens, cached=0, reasoning=0):
    return NS(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        input_tokens_details=NS(cached_tokens=cached),
        output_tokens_details=NS(reasoning_tokens=reasoning),
    )


def test_openai_reports_uncached_input():
    summary = openai_api.usage_summary("gpt-6-luna", openai_usage(1_000, 500, cached=400, reasoning=200))
    assert summary["input_tokens"] == 600
    assert summary["cache_read_input_tokens"] == 400
    assert summary["reasoning_tokens"] == 200
    assert summary["total_tokens"] == 1_500


def test_openai_long_context_pricing():
    # gpt-6-luna: $0.10 per million input, $0.20 above 272K prompt tokens
    cost = openai_api.estimate_cost("gpt-6-luna", openai_usage(300_000, 0))
    assert cost["input_cost"] == pytest.approx(0.06)


def openrouter_usage(prompt, completion, cached=0, cost=None):
    return NS(
        prompt_tokens=prompt,
        completion_tokens=completion,
        prompt_tokens_details=NS(cached_tokens=cached),
        completion_tokens_details=NS(reasoning_tokens=0),
        cost=cost,
    )


def test_openrouter_prefers_reported_cost():
    cost = openrouter_api.estimate_cost("openai/gpt-oss-120b", openrouter_usage(1_000_000, 0, cost=0.5))
    assert cost["total_cost"] == 0.5
    assert cost["estimated_total_cost"] == pytest.approx(0.15)


def test_openrouter_missing_cache_price_uses_input_price():
    # gemma lists no cache price, so cached tokens cost the input price ($0.20)
    cost = openrouter_api.estimate_cost("google/gemma-4-31b-it", openrouter_usage(1_000_000, 0, cached=1_000_000))
    assert cost["cache_read_cost"] == pytest.approx(0.2)
    assert cost["input_cost"] == 0


def test_openrouter_unpriced_model_with_reported_cost():
    cost = openrouter_api.estimate_cost("vendor/unpriced", openrouter_usage(10, 10, cost=0.01))
    assert cost["total_cost"] == 0.01
    assert cost["estimated_total_cost"] is None


# ---------- OpenRouter error and streaming handling (fake client, no API calls) ----------


def openrouter_completion(content="The answer is 42", finish_reason="stop", usage=True, choice_error=None, error=None):
    """A ChatCompletion built the way the SDK builds real responses (leniently, keeping OpenRouter's extra fields)."""
    choice = {"index": 0, "finish_reason": finish_reason, "message": {"role": "assistant", "content": content}}
    if choice_error:
        choice["error"] = choice_error
    value = {"id": "x", "object": "chat.completion", "created": 0, "model": "m", "choices": [choice]}
    if usage:
        value["usage"] = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, "cost": 0.001}
    if error:
        value["error"] = error
    return construct_type(type_=ChatCompletion, value=value)


class FakeStream:
    def __init__(self, deltas=(), final=None, raise_in_loop=None, raise_at_end=None):
        self.deltas, self.final = deltas, final
        self.raise_in_loop, self.raise_at_end = raise_in_loop, raise_at_end

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __iter__(self):
        for delta in self.deltas:
            yield NS(type="content.delta", delta=delta)
        if self.raise_in_loop:
            raise self.raise_in_loop

    def get_final_completion(self):
        if self.raise_at_end:
            raise self.raise_at_end
        return self.final


@pytest.fixture
def fake_openrouter(monkeypatch):
    """Install a fake client; returns a function that sets what it answers and records the request params."""
    sent = {}

    def install(stream=None, completion=None):
        def record(answer):
            def call(**params):
                sent.update(params)
                return answer

            return call

        client = NS(chat=NS(completions=NS(stream=record(stream), create=record(completion))))
        monkeypatch.setattr(openrouter_api, "_client", client)
        return sent

    return install


def test_openrouter_normal_reply(fake_openrouter):
    fake_openrouter(completion=openrouter_completion())
    text, usage = openrouter_api.chat([], stream=False, return_usage=True)
    assert text == "The answer is 42"
    assert usage["cost"]["total_cost"] == 0.001


@pytest.mark.parametrize(
    ("completion", "message"),
    [
        # A provider that fails mid-generation: OpenRouter puts the error on the choice
        (openrouter_completion(finish_reason="error", choice_error={"code": 502, "message": "Upstream died"}),
         "Upstream died"),
        (openrouter_completion(finish_reason="error"), "failed during generation"),
        (openrouter_completion(error={"code": 500, "message": "Top-level failure"}), "Top-level failure"),
    ],
)  # fmt: skip
def test_openrouter_errors_in_a_200_body_raise(fake_openrouter, completion, message):
    fake_openrouter(completion=completion)
    with pytest.raises(RuntimeError, match=message):
        openrouter_api.chat([], stream=False)


def test_openrouter_streamed_reply_cut_off_by_max_tokens_is_kept(fake_openrouter, capsys):
    final = openrouter_completion(content="The answer is 42 and", finish_reason="length")
    sent = fake_openrouter(
        stream=FakeStream(["The answer is ", "42 and"], raise_at_end=LengthFinishReasonError(completion=final))
    )

    response, usage = openrouter_api.chat([], stream=True, return_full_response=True, return_usage=True)

    assert openrouter_api.text_from_response(response) == "The answer is 42 and"
    assert openrouter_api.stop_reason(response) == "length"
    assert usage["cost"]["total_cost"] == 0.001
    assert sent["stream_options"] == {"include_usage": True}
    assert capsys.readouterr().out == "The answer is 42 and\n"


def test_openrouter_mid_stream_cut_off_keeps_text_but_has_no_usage(fake_openrouter):
    # With strict function tools the SDK raises mid-stream, before the usage chunk arrives
    snapshot = openrouter_completion(content="partial", finish_reason="length", usage=False)
    install = fake_openrouter

    install(stream=FakeStream(["partial"], raise_in_loop=LengthFinishReasonError(completion=snapshot)))
    assert openrouter_api.chat([], stream=True) == "partial"

    install(stream=FakeStream(["partial"], raise_in_loop=LengthFinishReasonError(completion=snapshot)))
    with pytest.raises(RuntimeError, match="no usage"):
        openrouter_api.chat([], stream=True, return_usage=True)


def test_openrouter_content_filter_without_a_completion_still_raises(fake_openrouter):
    fake_openrouter(stream=FakeStream(raise_at_end=ContentFilterFinishReasonError()))
    with pytest.raises(ContentFilterFinishReasonError):
        openrouter_api.chat([], stream=True)


# ---------- Anthropic response helpers ----------


def test_anthropic_text_from_response_keeps_cited_sentences_whole():
    # With web search, Claude splits one answer into text blocks around each citation
    response = NS(
        content=[
            NS(type="text", text="Canberra is "),
            NS(type="web_search_tool_result"),
            NS(type="text", text="the capital."),
        ]
    )
    assert anthropic_api.text_from_response(response) == "Canberra is the capital."


def test_anthropic_show_web_search_response_without_a_search(capsys):
    response = NS(stop_reason="end_turn", usage=NS(server_tool_use=None), content=[])
    anthropic_api.show_web_search_response(response)
    assert "Searches used: 0" in capsys.readouterr().out
