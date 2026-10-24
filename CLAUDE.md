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
- `evaluate(model, request)` makes one non-streaming call through `chat()` and prints the response, token counts and estimated cost. `evaluate_all_models(prompt)` does the same for every model in `MODEL_PRICING` and prints a comparison table. With `log=True`, both append to `logs/eval_log.json` (a JSON array of runs) via `log_evaluation()`.
- `view_logs()` starts a local server on 127.0.0.1 (port 8765, or the next free one) in a daemon thread and opens `web/log_viewer.html`, a self-contained page with no dependencies for filtering, charting and exporting the log. The server answers only `/` (the page) and `/log.json` (the log file) and returns 404 for everything else, so it never exposes the project folder or `.env`. Keep it that way if you change it. The page can also open a log file directly, without the server.
- Conversation history is a plain list of dicts built with `add_user_message` / `add_assistant_message`. The `_multi` variants accept a full `Message` and copy its `content` blocks — use these in tool-use loops so `tool_use` blocks are preserved.
- `build_tooluse_json` builds a `tool_result` payload but takes the ID from `content[0]`, which is wrong when Claude emits text before the tool call.
- Streaming: `chat()` goes through the stream's events once, printing text and tool-input chunks as they arrive. Don't add a second loop such as `text_stream`: the first loop uses up the stream.

## Workflow for any feature or task
1. **Explore**: read the relevant files first. Don't edit anything yet.
2. **Plan**: write a short numbered plan (files to change, approach, tests).
   Then STOP and wait for my approval.
3. **Code**: only after I approve.
4. **Commit**: suggest a commit message and wait. Don't run git commit
   yourself.
