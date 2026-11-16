# API Model Evaluator Tool

A Python sandbox for calling models from several providers through one interface and comparing **what each call costs** against **how accurate the answer is**.

- **Anthropic** (Claude) through the Messages API: `AnthropicAPI.py`
- **OpenAI** (GPT-6 family) through the Responses API: `OpenAIAPI.py`
- **Open-weight models** (DeepSeek, Gemma, Qwen, Mistral, Kimi, gpt-oss) through OpenRouter: `OpenRouterAPI.py`
- **Evaluation**: `evaluation.py` runs prompts or datasets against any mix of these models, scores the answers from 1 to 5, logs every run, and serves a local chat page and log viewer.

> **Every run makes real, billed API calls.** The cost of each call is estimated from the hand-maintained price tables and shown next to its result.

## Setup

You need Python 3.13. The project has no `requirements.txt`, so install the packages directly:

```
python -m venv .venv
.venv/Scripts/pip install anthropic openai python-dotenv nbstripout
```

Put your API keys in a `.env` file in the project root. They're loaded automatically at import time, and `.env` is git-ignored.

```
ANTHROPIC_API_KEY=...
OPENAI_API_KEY=...
OPENROUTER_API_KEY=...   # optional: without it, default runs leave the OpenRouter models out
```

On a fresh clone, install the notebook output filter once, so notebooks are committed without their outputs:

```
.venv/Scripts/nbstripout --install --attributes .gitattributes
```

## Usage

None of the files has a `main`. Each one ends with commented-out example calls: uncomment the one you want and run the file.

```
.venv/Scripts/python evaluation.py
```

You can also import it, for example from `notebook/eval_testing.ipynb`:

```python
from evaluation import *

# One model, one prompt: prints the response, token counts and estimated cost
evaluate("claude-haiku-5-5", "What is quantum computing? Answer in one sentence")

# Every priced model (or providers=[...] / models=[...]) side by side in a table
evaluate_all_models("What is quantum computing? Answer in one sentence", providers=["openai"])

# Score the answers, and append the run to logs/eval_log.json
evaluate_all_models("What is the capital of Australia? Answer with just the city name.",
                    expected="Canberra", log=True)

# Run a whole dataset against several models in parallel (asks for confirmation first)
evaluate_batch("datasets/example.json", models=["claude-haiku-5-5", "gpt-6-luna"], log=True)

# Browser pages (served locally on 127.0.0.1:8765, or the next free port)
chat_ui(block=True)   # block=True keeps the server running when started from a script
view_logs()
```

The provider names are `anthropic`, `openai` and `openrouter`. Each provider module can also be used on its own through its `chat(messages, model=, system=, stream=, ...)` function.

## Scoring

Scores use a single 1–5 scale, and **4 or more counts as a pass**. Scoring is turned on when you pass `expected=` and/or `rubric=`. There are three scorers:

| Scorer | Needs | How it scores |
| --- | --- | --- |
| `llm_judge` (default) | `expected` or `rubric` | A grader model (`claude-sonnet-5-5` by default, set with `judge_model=`) returns a 1–5 score and its reasoning. It judges content only, not formatting. |
| `exact` | `expected` | 5 if the normalised response equals the expected answer, otherwise 1 |
| `contains` | `expected` | 5 if the response contains the expected answer, otherwise 1 |

The grader's own cost is tracked separately from the cost of the model being graded.

### Datasets

`evaluate_batch()` takes a list of cases or a path to a `.json` / `.jsonl` file. Each case looks like this:

```json
{"id": "capital-australia", "prompt": "What is the capital of Australia?", "expected": "Canberra"}
{"id": "explain-recursion", "prompt": "Explain recursion to a beginner.", "rubric": "Mentions a base case; avoids jargon."}
```

`id` is optional (missing ids become `case-N`). Two datasets are included: `datasets/example.json` (11 easy-to-moderate cases) and `datasets/hard.json` (15 harder ones).

## Web pages

- **Chat** (`web/chat.html`, opened by `chat_ui()`): sends one prompt to every checked model at once, with a tab holding each model's own conversation. It shows each call's tokens and cost, accepts images, PDFs and text files as attachments, and can include context files that every model gets for the whole conversation.
- **Log viewer** (`web/log_viewer.html`, opened by `view_logs()`): filters the log by provider, model, batch, chat session and score, and charts cost, tokens and accuracy against cost. It exports to CSV and lets you grade any result 1–5 yourself, with a note. A grade you give replaces the grader's score, and results where you and the grader disagree are flagged.

The server only listens on localhost, and it refuses requests that don't come from its own pages.

## Project layout

```
AnthropicAPI.py      Claude client, pricing table and cost estimates
OpenAIAPI.py         OpenAI client, pricing table and cost estimates
OpenRouterAPI.py     OpenRouter client for open-weight models
evaluation.py        Evaluation, scoring, batch runs, logging and the local web server
datasets/            Evaluation cases (example.json, hard.json)
logs/eval_log.json   Log of every run made with log=True
web/                 Chat page and log viewer (self-contained HTML, no dependencies)
notebook/            Jupyter notebook for running evaluations
CLAUDE.md            Detailed notes on the internals
```

## Adding a model or provider

- **New model:** add its prices (USD per million tokens) to the provider's `MODEL_PRICING`. Only priced models are offered by default, and models not in the table get no cost estimate.
- **New provider:** write a module with the same interface as the existing three (`MODEL_PRICING`, `chat()`, `text_from_response()`, `stop_reason()`, `usage_summary()`, `user_content()`), then register it in `PROVIDERS` in `evaluation.py`. Keep provider-specific code out of `evaluation.py`.

See [`CLAUDE.md`](CLAUDE.md) for more on how the pieces fit together.
