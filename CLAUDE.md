# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

A Python sandbox for experimenting with the Anthropic Messages API (`AnthropicAPI.py`) and the OpenAI Responses API (`OpenAIAPI.py`), and for comparing what each model's calls cost (`evaluation.py`).

## Environment

- Python 3.13, virtualenv in `.venv/` (PyCharm-managed). No `requirements.txt`; dependencies are `anthropic`, `openai` and `python-dotenv`, plus `nbstripout` for the notebook.
- `.gitattributes` runs notebooks through the `nbstripout` git filter, so outputs and execution metadata are stripped when a notebook is committed (the working copy keeps them). The filter itself lives in `.git/config`, so a fresh clone needs `.venv/Scripts/nbstripout --install --attributes .gitattributes` once; until then, git silently commits notebooks with their outputs. Once installed, the filter is required, so a notebook commit fails if `.venv` is missing.
- `ANTHROPIC_API_KEY` and `OPENAI_API_KEY` are loaded from `.env` via `load_dotenv()` at import time. `.claude/settings.json` denies reading `.env` — don't try to open it.
- There are no tests, linter config, or build step.

## Running

```
.venv/Scripts/python AnthropicAPI.py
.venv/Scripts/python OpenAIAPI.py
.venv/Scripts/python evaluation.py
```

None of the files has a `main`; experiments are the `tester*` / `structured_data_tester` / `evaluate*` / `chat_ui` calls, invoked by uncommenting a call at the bottom of the file. `notebook/eval_testing.ipynb` does `from evaluation import *`; its `evaluate*` cells are commented out the same way, so Run All makes no API calls. Every run makes real (billed) API calls.

## Architecture

### Provider modules

`AnthropicAPI.py` and `OpenAIAPI.py` each expose the same interface, which `evaluation.py` relies on: `MODEL_PRICING`, `chat(messages, model=, system=, stream=, return_full_response=, return_usage=)`, `text_from_response(response)`, `stop_reason(response)`, `usage_summary()` returning the same keys, and `user_content(text, attachments)`. `user_content` turns provider-neutral attachments (`{"type": "image" | "pdf" | "text", "name", "media_type", "data"}`, with base64 data for images and PDFs and plain text for text files) into that provider's content blocks: Claude `image` / `document` blocks, or OpenAI `input_image` / `input_file` / `input_text` parts. Keep that interface in step when changing either module, and keep provider-specific code out of `evaluation.py`.

### AnthropicAPI.py

- A module-level `client = Anthropic()` is shared by all helpers.
- `chat()` is the central entry point; everything new should go through it rather than the legacy `make_request` / `chat_old`. It builds a `params` dict and optionally adds `system`, `output_config` (structured JSON output), and `tools` (with `fine_grained=True` adding `eager_input_streaming` to custom tools). The `temperature` argument is accepted but never sent: the installed SDK uses API version 2026-10-21, which rejects sampling parameters with a 400.
  - Defaults: `model="claude-haiku-5-5"`, `stream=True`, `max_tokens=4000`.
  - Returns concatenated text blocks, or the full `Message` when `return_full_response=True` (needed for tool use and web search). With `return_usage=True` it returns `(result, usage_summary)` instead.
- Cost estimation: `MODEL_PRICING` (USD per million tokens, hand-maintained, plus `LONG_CONTEXT_PRICING` for Haiku 5.5 above 100K prompt tokens) feeds `estimate_cost(model, usage)`. It charges cache writes at 1.25× input (5-minute TTL) or 2× (1-hour TTL), and returns `None` for models not in the table. `usage_summary()` combines the token counts and cost. When adding a model, add its prices to `MODEL_PRICING`.
- Conversation history is a plain list of dicts built with `add_user_message` / `add_assistant_message`. The `_multi` variants accept a full `Message` and copy its `content` blocks — use these in tool-use loops so `tool_use` blocks are preserved.
- `build_tooluse_json` builds a `tool_result` payload but takes the ID from `content[0]`, which is wrong when Claude emits text before the tool call.
- Streaming: `chat()` goes through the stream's events once, printing text and tool-input chunks as they arrive. Don't add a second loop such as `text_stream`: the first loop uses up the stream.

### OpenAIAPI.py

- A module-level `client = OpenAI()`. `chat()` mirrors the Anthropic one on the Responses API: `system` is sent as `instructions`, an Anthropic-style `output_config` becomes `text={"format": ...}` (with `name` and `strict` added), and `max_tokens` becomes `max_output_tokens`. `temperature` is accepted but not sent. Default model is `gpt-6-luna`. Streaming goes through the events once, printing `response.output_text.delta` and function-call argument chunks.
- `stop_reason()` returns `status`, or `incomplete_details.reason` (e.g. `max_output_tokens`) for an incomplete response.
- Usage: OpenAI's `input_tokens` includes cached tokens and cache writes. `usage_summary()` reports uncached input as `input_tokens` to match Anthropic, and adds `reasoning_tokens` (already counted in `output_tokens`, which is what's billed). `estimate_cost()` charges cache writes at 1.25× input (the 30-minute rate; 12-hour writes cost 2×, but the usage doesn't say which was used). `MODEL_PRICING` covers the GPT-6 family only, with standard-tier prices from OpenAI's pricing page, plus `LONG_CONTEXT_PRICING` for prompts above 272K tokens.
- Tool-use helpers: `add_response_output(messages, response)` appends every output item (reasoning, function calls) to the history; `build_tool_result(call_id, output)` builds a `function_call_output` item.

### evaluation.py

- `PROVIDERS` maps a provider name to its module. `provider_for(model)` finds the provider whose `MODEL_PRICING` contains the model, falling back to the `claude-` / `gpt-` / `o*` name prefix. `run_model()` makes one non-streaming call and returns the result dict that gets logged, which includes `provider`.
- `evaluate(model, request)` makes one non-streaming call through `chat()` and prints the response, token counts and estimated cost. `evaluate_all_models(prompt)` does the same for every priced model of every provider (or only `providers=[...]`, or an explicit `models=[...]`) and prints a comparison table. With `log=True`, both append to `logs/eval_log.json` (a JSON array of runs) via `log_evaluation()`.
- `view_logs()` starts a local server on 127.0.0.1 (port 8765, or the next free one) in a daemon thread and opens `web/log_viewer.html`, a self-contained page with no dependencies for filtering, charting and exporting the log. `chat_ui()` starts the same server (via `_ensure_server()`) and opens `web/chat.html`; from a script, pass `block=True` so the script doesn't exit and take the daemon server with it. The server answers only `GET /` (the log viewer), `/log.json` (the log file), `/chat` (the chat page) and `/models` (`default_models()` with their providers), plus `POST /run`, and returns 404 for everything else, so it never exposes the project folder or `.env`. Keep it that way if you change it.
- Attachments: a message dict with an `attachments` list is converted per model by `_provider_messages()` (in `run_model()`) through the provider's `user_content`. `_check_attachments()` enforces the allowed types (PNG/JPEG/GIF/WebP images, PDFs, UTF-8 text sent as `text/plain`), valid base64, and `ATTACHMENT_MAX_BYTES` per file (5 MB image, 10 MB PDF, 1 MB text). `ATTACHMENTS_MAX_TOTAL_BYTES` (20 MB) applies to everything one request resends. The log records only `attachment_summary()` (name, type, media type, bytes), never the file data.
- `POST /run` takes `{prompt, system, models, log, session_id, histories, attachments}` and makes billed calls. It is refused (403) unless `Host` and `Origin` are the server's own 127.0.0.1/localhost address and the body is JSON, so other websites open in the browser can't trigger calls. It only accepts models from `default_models()`. `histories` (optional) maps each model to its own conversation so far, as messages alternating `user`/`assistant` (checked by `_check_history`). User turns may carry `attachments`, and a user turn may be attachments with no text. The server keeps no state: each model is sent its history plus the new prompt. It runs the models in parallel through `_run_model_safe()` (the error-to-result wrapper `evaluate_all_models` also uses) and sends back one NDJSON line per model as each finishes, then a `done` line. Unless `log` is false it logs one entry per turn with `type: "chat"`, under `_log_lock`. With a `session_id`, `log_evaluation()` also records `session_id`, `turn` and `session_total_cost` (this turn's `total_cost` plus that of the session's earlier logged turns), and the `done` line returns `turn` and `session_total_cost`.
- `web/chat.html` has a checkbox for every priced model (all checked by default), a prompt box with an optional system prompt, and one tab per model holding that model's whole conversation, with each call's tokens and cost and the model's session cost. The page holds the conversation: each model keeps its own history, and a failed or empty reply is left out of it. The models, system prompt and Log checkbox are locked after the first send until **Clear conversation** starts a new session. A turn the server refused (nothing billed) is taken back. Each conversation gets an 8-character session id, shown in the summary line with a copy button and a **View in log** link to `/?session=<id>`. Files can be attached with the **Attach files** button, by dropping them on the prompt box or by pasting images. The page applies the same type and size limits as the server before sending. Attachments become part of each model's history and are resent every turn.
- Both pages explain their controls with tooltips: any element with a `data-tip` attribute shows that text in a bubble on hover or keyboard focus, and section headings have a focusable ⓘ (`.info`) button. Each page carries its own copy of the small tooltip script and styles. Give new controls a `data-tip`, and keep the text in step when a control's behaviour changes.
- `web/log_viewer.html` has a **Chat session** filter listing the sessions in the log (id, start time, turns, first prompt). It reads `?session=<id>` on load and keeps it in the address bar. A chat turn's run card shows its session id, turn and session total so far, with a link to filter to that session. Run cards list a turn's attachments. The search box matches session ids and attachment names, and the CSV export has `session_id`, `turn`, `session_total_cost` and `attachments` columns. The page can also open a log file directly, without the server. It groups results by model, filters by provider (the dropdown lists the providers in the log, and the model checkboxes show only the selected provider's models; entries logged before `provider` was recorded take it from another result for the same model, or else from the model name, using a copy of `provider_for`'s prefix fallback in `inferProvider`), and treats both `end_turn` and `completed` as normal stops.

## Workflow for any feature or task
1. **Explore**: read the relevant files first. Don't edit anything yet.
2. **Plan**: write a short numbered plan (files to change, approach, tests).
   Then STOP and wait for my approval.
3. **Code**: only after I approve.
4. **Commit**: suggest a commit message and wait. Don't run git commit
   yourself.
