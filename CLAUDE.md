# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

A single-file Python sandbox (`AnthropicAPI.py`) for experimenting with the Anthropic Messages API. Despite the repo name, no cost/usage analysis is implemented yet — if adding it, the source data is `response.usage` on each returned `Message`.

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
- `chat()` is the central entry point; everything new should go through it rather than the legacy `make_request` / `chat_old`. It builds a `params` dict and optionally adds `system`, `output_config` (structured JSON output), and `tools` (with `fine_grained=True` adding `eager_input_streaming` to custom tools). Temperature is passed via `extra_body`, not `params`.
  - Defaults: `model="claude-haiku-5-5"`, `stream=True`, `max_tokens=4000`.
  - Returns concatenated text blocks, or the full `Message` when `return_full_response=True` (needed for tool use, web search, and usage data).
- Conversation history is a plain list of dicts built with `add_user_message` / `add_assistant_message`. The `_multi` variants accept a full `Message` and copy its `content` blocks — use these in tool-use loops so `tool_use` blocks are preserved.
- `build_tooluse_json` builds a `tool_result` payload but takes the ID from `content[0]`, which is wrong when Claude emits text before the tool call.
- Known bug: in streaming mode `chat()` iterates the raw events first, which exhausts the stream, so the subsequent `text_stream` loop prints nothing (the final returned text is still correct).
