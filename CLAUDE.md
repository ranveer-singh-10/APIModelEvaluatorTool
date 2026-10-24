# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

A single-file Python sandbox (`AnthropicAPI.py`) for experimenting with the Anthropic Messages API and estimating what each call costs.

## Environment

- Python 3.13, virtualenv in `.venv/` (PyCharm-managed). No `requirements.txt`; dependencies are `anthropic` and `python-dotenv`.
- `ANTHROPIC_API_KEY` is loaded from `.env` via `load_dotenv()` at import time. `.claude/settings.json` denies reading `.env` — don't try to open it.
- There are no tests, linter config, or build step.

## Running

```
.venv/Scripts/python AnthropicAPI.py
```

The file has no `main`; experiments are the `tester*` / `structured_data_tester` functions, invoked by uncommenting a call at the bottom of the file. Every run makes real (billed) API calls.

## Architecture

- A module-level `client = Anthropic()` is shared by all helpers.
- `chat()` is the central entry point; everything new should go through it rather than the legacy `make_request` / `chat_old`. It builds a `params` dict and optionally adds `system`, `output_config` (structured JSON output), and `tools` (with `fine_grained=True` adding `eager_input_streaming` to custom tools). The `temperature` argument is accepted but never sent: the installed SDK uses API version 2026-10-21, which rejects sampling parameters with a 400.
  - Defaults: `model="claude-haiku-5-5"`, `stream=True`, `max_tokens=4000`.
  - Returns concatenated text blocks, or the full `Message` when `return_full_response=True` (needed for tool use and web search). With `return_usage=True` it returns `(result, usage_summary)` instead.
- Cost estimation: `MODEL_PRICING` (USD per million tokens, hand-maintained, plus `LONG_CONTEXT_PRICING` for Haiku 5.5 above 100K prompt tokens) feeds `estimate_cost(model, usage)`. It charges cache writes at 1.25× input (5-minute TTL) or 2× (1-hour TTL), and returns `None` for models not in the table. `usage_summary()` combines the token counts and cost. When adding a model, add its prices to `MODEL_PRICING`.
- `evaluate(model, request)` makes one non-streaming call through `chat()` and prints the response, token counts and estimated cost.
- Conversation history is a plain list of dicts built with `add_user_message` / `add_assistant_message`. The `_multi` variants accept a full `Message` and copy its `content` blocks — use these in tool-use loops so `tool_use` blocks are preserved.
- `build_tooluse_json` builds a `tool_result` payload but takes the ID from `content[0]`, which is wrong when Claude emits text before the tool call.
- Known bug: in streaming mode `chat()` iterates the raw events first, which exhausts the stream, so the subsequent `text_stream` loop prints nothing (the final returned text is still correct).
