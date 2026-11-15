from dotenv import load_dotenv
load_dotenv()

import os

from openai import OpenAI

# Open-weight models through OpenRouter's OpenAI-compatible Chat Completions API. Same interface as AnthropicAPI and
# OpenAIAPI (MODEL_PRICING, chat, text_from_response, stop_reason, usage_summary, user_content), plus configured().

SPACER = "--------------------------"
BASE_URL = "https://openrouter.ai/api/v1"
API_KEY_ENV = "OPENROUTER_API_KEY"

# USD per million tokens, from https://openrouter.ai/api/v1/models (checked 2026-11-15). Estimates only: each response
# reports what OpenRouter actually charged (usage.cost), and usage_summary uses that as the total when it's there.
# A cache_read of None means OpenRouter lists no cache price, so cached tokens are estimated at the input price.
MODEL_PRICING = {
    "deepseek/deepseek-v4.1-flash-1017": {"input": 0.02, "output": 0.108, "cache_read": 0.007},
    "google/gemma-4-31b-it":             {"input": 0.20, "output": 0.40,  "cache_read": None},
    "openai/gpt-oss-120b":               {"input": 0.15, "output": 0.60,  "cache_read": 0.075},
    "qwen/qwen3.8-27b":                  {"input": 0.15, "output": 1.875, "cache_read": 0.0375},
    "mistralai/mistral-large-4-0":       {"input": 0.70, "output": 2.10,  "cache_read": 0.07},
    "moonshotai/kimi-k3.1-pro":          {"input": 2.00, "output": 10.00, "cache_read": 0.20},
}

# PDFs sent to a model that can't read files natively are converted by OpenRouter's file-parser plugin, whose default
# engine is paid OCR (mistral-ocr, $2 per 1,000 pages). cloudflare-ai converts them to text for free.
PDF_ENGINE = "cloudflare-ai"

_client = None

def configured():
    # evaluation.default_models() leaves this provider out until the key is in .env, so default runs don't fail on it
    return bool(os.environ.get(API_KEY_ENV))

def _get_client():
    # Created on first use, so importing this module (and evaluation.py) works before the key is set.
    # No HTTP-Referer / X-Title headers: they're optional and would identify the project to OpenRouter.
    global _client
    if _client is None:
        if not configured():
            raise RuntimeError(f"{API_KEY_ENV} isn't set: add it to .env to use OpenRouter models")
        _client = OpenAI(api_key=os.environ[API_KEY_ENV], base_url=BASE_URL)
    return _client

def list_models():
    for model in _get_client().models.list():
        print(model.id)

def add_user_message(messages, text):
    messages.append({"role": "user", "content": text})

def add_assistant_message(messages, text):
    messages.append({"role": "assistant", "content": text})

def user_content(text, attachments=()):
    # Turns a prompt and provider-neutral attachments into Chat Completions content parts; matches the other modules.
    # Images only work with vision models (OpenRouter returns an error otherwise). An attachment's "cache" flag is
    # ignored: most open-model providers on OpenRouter cache repeated prompt prefixes by themselves.
    if not attachments:
        return text
    parts = []
    for a in attachments:
        if a["type"] == "image":
            parts.append({"type": "image_url", "image_url": {"url": f"data:{a['media_type']};base64,{a['data']}"}})
        elif a["type"] == "pdf":
            parts.append({"type": "file", "file": {"filename": a["name"], "file_data": f"data:application/pdf;base64,{a['data']}"}})
        elif a["type"] == "text":
            parts.append({"type": "text", "text": f"File: {a['name']}\n\n{a['data']}"})
        else:
            raise ValueError(f"Unsupported attachment type: {a['type']}")
    if text:
        parts.append({"type": "text", "text": text})
    return parts

def _has_pdf(messages):
    return any(isinstance(m.get("content"), list) and any(isinstance(p, dict) and p.get("type") == "file" for p in m["content"])
               for m in messages if isinstance(m, dict))

def text_from_response(response):
    return (response.choices[0].message.content or "") if response.choices else ""

def stop_reason(response):
    # OpenRouter's normalised finish_reason: "stop" is a normal finish; "length" means max_tokens cut it short
    return response.choices[0].finish_reason if response.choices else None

def _token_counts(usage):
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

def estimate_cost(model, usage):
    # The breakdown from MODEL_PRICING; total_cost is replaced by OpenRouter's reported charge when there is one
    pricing = MODEL_PRICING.get(model)
    reported = getattr(usage, "cost", None)
    reported = reported if isinstance(reported, (int, float)) and not isinstance(reported, bool) else None
    if pricing is None:
        if reported is None:
            print(f"WARNING: no pricing for model '{model}', cost not estimated")
            return None
        return {"input_cost": 0.0, "output_cost": 0.0, "cache_write_cost": 0.0, "cache_read_cost": 0.0,
                "total_cost": reported, "estimated_total_cost": None}

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

def usage_summary(model, usage):
    # Same keys as the other modules, plus reasoning_tokens like OpenAIAPI
    summary = _token_counts(usage)
    summary["total_tokens"] = (
        summary["input_tokens"] + summary["output_tokens"]
        + summary["cache_read_input_tokens"] + summary["cache_creation_input_tokens"]
    )
    summary["cost"] = estimate_cost(model, usage)
    return summary

def _check_error(response):
    # OpenRouter can report a failure inside an HTTP 200 body, e.g. when a provider fails after the response started
    error = getattr(response, "error", None)
    if error:
        message = error.get("message") if isinstance(error, dict) else str(error)
        raise RuntimeError(f"OpenRouter error: {message}")

def chat(messages, model="deepseek/deepseek-v4.1-flash-1017", system=None, temperature=1.0, stream=True, output_config=None,
         tools=None, return_full_response=False, return_usage=False, max_tokens=4000):
    # temperature is accepted for parity with the other modules but not sent
    params = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": ([{"role": "system", "content": system}] if system else []) + list(messages),
    }
    extra_body = {}

    if output_config:
        # Accepts the Anthropic-style {"format": {"type": "json_schema", "schema": ...}}. require_parameters makes
        # OpenRouter use only providers that support it, rather than silently ignoring the schema.
        fmt = output_config["format"]
        params["response_format"] = {"type": "json_schema", "json_schema": {"name": "output", "strict": True, "schema": fmt["schema"]}}
        extra_body["provider"] = {"require_parameters": True}

    if _has_pdf(params["messages"]):
        extra_body["plugins"] = [{"id": "file-parser", "pdf": {"engine": PDF_ENGINE}}]

    if tools:
        params["tools"] = tools
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

def tester():
    messages = []
    add_user_message(messages, "What is quantum computing? Answer in one sentence")
    answer = chat(messages)
    print(SPACER)
    add_assistant_message(messages, answer)
    add_user_message(messages, "Write another sentence")
    chat(messages)

#list_models()
#tester()
