import os
import time
import json
import base64
import binascii
import threading
import webbrowser
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import AnthropicAPI
import OpenAIAPI

SPACER = "--------------------------"

# Each provider module exposes the same interface: MODEL_PRICING, chat(), text_from_response() and stop_reason()
PROVIDERS = {
    "anthropic": AnthropicAPI,
    "openai": OpenAIAPI,
}

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
# Next to this file, so the notebook and scripts share one log whatever their working directory
LOG_FILE = os.path.join(PROJECT_DIR, "logs", "eval_log.json")
LOG_VIEWER_PAGE = os.path.join(PROJECT_DIR, "web", "log_viewer.html")
CHAT_PAGE = os.path.join(PROJECT_DIR, "web", "chat.html")

def provider_for(model):
    for name, provider in PROVIDERS.items():
        if model in provider.MODEL_PRICING:
            return name
    # Unpriced models still run (with no cost estimate) if the name says which provider they belong to.
    # web/log_viewer.html copies these prefixes in inferProvider, so keep the two in step.
    if model.startswith("claude-"):
        return "anthropic"
    if model.startswith(("gpt-", "o1", "o3", "o4")):
        return "openai"
    raise ValueError(f"Can't tell which provider model '{model}' belongs to")

def default_models(providers=None):
    return [model for name in (providers or PROVIDERS) for model in PROVIDERS[name].MODEL_PRICING]

# Attachments travel in one provider-neutral shape, {"type", "name", "media_type", "data"}, and each provider's
# user_content() turns them into its own content blocks. Images and PDFs are base64; text files are plain text.
ATTACHMENT_MEDIA_TYPES = {
    "image": {"image/png", "image/jpeg", "image/gif", "image/webp"},
    "pdf": {"application/pdf"},
    "text": {"text/plain"},
}
# Bytes per file, and for all attachments sent in one request. Every turn resends the conversation's attachments,
# and Anthropic rejects requests over 32 MB (base64 adds a third), so the total stays well under that.
ATTACHMENT_MAX_BYTES = {"image": 5_000_000, "pdf": 10_000_000, "text": 1_000_000}
ATTACHMENTS_MAX_TOTAL_BYTES = 20_000_000

def _check_attachments(attachments):
    # Validates attachments from the chat page and returns their total size in bytes
    if not isinstance(attachments, list):
        raise ValueError("attachments must be a list")
    total = 0
    for a in attachments:
        if (not isinstance(a, dict) or set(a) != {"type", "name", "media_type", "data"}
                or a["type"] not in ATTACHMENT_MEDIA_TYPES or a["media_type"] not in ATTACHMENT_MEDIA_TYPES[a["type"]]
                or not isinstance(a["name"], str) or not 0 < len(a["name"]) <= 255
                or not isinstance(a["data"], str)):
            raise ValueError("each attachment needs a supported type and media_type, a name and its data")
        if a["type"] == "text":
            size = len(a["data"].encode("utf-8"))
        else:
            try:
                size = len(base64.b64decode(a["data"], validate=True))
            except (binascii.Error, ValueError):
                raise ValueError(f"attachment {a['name']} is not valid base64")
        if size == 0 or size > ATTACHMENT_MAX_BYTES[a["type"]]:
            raise ValueError(f"attachment {a['name']} is empty or over the {ATTACHMENT_MAX_BYTES[a['type']] // 1_000_000} MB limit")
        total += size
    return total

def attachment_summary(attachments):
    # What the log records about each attachment: never the data, which would bloat the log
    return [
        {"name": a["name"], "type": a["type"], "media_type": a["media_type"],
         "bytes": len(a["data"].encode("utf-8")) if a["type"] == "text" else len(base64.b64decode(a["data"]))}
        for a in attachments
    ]

def with_context_files(messages, context_files):
    # Context files belong to the whole conversation: they go at the start of its first user message, ahead of that
    # turn's own attachments, so every turn resends them. (The system prompt can't hold them: both APIs take only text there.)
    if not context_files or not messages:
        return messages
    first = messages[0]
    return [{**first, "attachments": list(context_files) + list(first.get("attachments") or [])}] + messages[1:]

def _provider_messages(provider, messages):
    # A message with "attachments" becomes that provider's content blocks; anything else is passed on unchanged
    converted = []
    for m in messages:
        if isinstance(m, dict) and "attachments" in m:
            attachments = m["attachments"]
            m = {k: v for k, v in m.items() if k != "attachments"}
            if attachments:
                m["content"] = provider.user_content(m["content"], attachments)
        converted.append(m)
    return converted

def run_model(model, messages, system=None):
    # One non-streaming call through the provider's chat(), returning the fields every evaluation records
    provider_name = provider_for(model)
    provider = PROVIDERS[provider_name]
    messages = _provider_messages(provider, messages)
    start = time.perf_counter()
    response, usage = provider.chat(messages, model=model, system=system, stream=False, return_full_response=True, return_usage=True)
    return {
        "model": model,
        "provider": provider_name,
        "seconds": time.perf_counter() - start,
        "stop_reason": provider.stop_reason(response),
        "response": provider.text_from_response(response),
        "usage": usage,
        "error": None,
    }

def _run_model_safe(model, messages, system=None):
    # Like run_model, but an error becomes a result, so one unavailable model doesn't stop a comparison
    start = time.perf_counter()
    try:
        return run_model(model, messages, system=system)
    except Exception as e:
        try:
            provider_name = provider_for(model)
        except ValueError:
            provider_name = None
        return {
            "model": model,
            "provider": provider_name,
            "seconds": time.perf_counter() - start,
            "stop_reason": None,
            "response": "",
            "usage": None,
            "error": f"{type(e).__name__}: {e}",
        }

def _read_log(log_file):
    if not os.path.exists(log_file):
        return []
    with open(log_file, encoding="utf-8") as f:
        content = f.read().strip()
    if not content:
        return []
    entries = json.loads(content)
    if not isinstance(entries, list):
        raise ValueError(f"{log_file} is not a JSON list of log entries")
    return entries

def log_evaluation(kind, prompt, system, results, log_file=LOG_FILE, session_id=None, attachments=None, context_files=None):
    # The log file is one JSON array of runs, newest last, so a webpage can load it with a single JSON.parse
    entries = _read_log(log_file)
    total_cost = sum(r["usage"]["cost"]["total_cost"] for r in results if r["usage"] and r["usage"]["cost"])

    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "type": kind,
        "prompt": prompt,
        "system": system,
        "total_cost": total_cost,
    }
    # Names, types and sizes only (see attachment_summary)
    if attachments:
        entry["attachments"] = attachments
    if context_files:
        entry["context_files"] = context_files
    if session_id is not None:
        # One entry per turn of a multi-turn chat; the session's running total comes from its earlier logged turns
        earlier = [e for e in entries if isinstance(e, dict) and e.get("session_id") == session_id]
        entry["session_id"] = session_id
        entry["turn"] = len(earlier) + 1
        entry["session_total_cost"] = total_cost + sum(
            e["total_cost"] for e in earlier if isinstance(e.get("total_cost"), (int, float))
        )
    entry["results"] = results

    entries.append(entry)
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    with open(log_file, "w", encoding="utf-8") as f:
        # default=str covers SDK content blocks when a prompt is passed as a messages list
        json.dump(entries, f, indent=2, ensure_ascii=False, default=str)

    print(f"Logged to {log_file}")
    return entry

def evaluate(model, request, system=None, log=False, log_file=LOG_FILE):
    # Both providers take the same {"role", "content"} message list
    messages = [{"role": "user", "content": request}] if isinstance(request, str) else request

    result = run_model(model, messages, system=system)
    usage = result["usage"]

    print(f"MODEL: {model} ({result['provider']})")
    print(SPACER)
    print(result["response"])
    print(SPACER)
    print(f"stop_reason: {result['stop_reason']}")
    print(f"Input tokens:       {usage['input_tokens']}")
    print(f"Output tokens:      {usage['output_tokens']}")
    if "reasoning_tokens" in usage:
        print(f"  of which reasoning: {usage['reasoning_tokens']}")
    print(f"Cache read tokens:  {usage['cache_read_input_tokens']}")
    print(f"Cache write tokens: {usage['cache_creation_input_tokens']}")
    print(f"Total tokens:       {usage['total_tokens']}")

    cost = usage["cost"]
    if cost:
        print(f"Input cost:         ${cost['input_cost']:.6f}")
        print(f"Output cost:        ${cost['output_cost']:.6f}")
        print(f"Cache cost:         ${cost['cache_write_cost'] + cost['cache_read_cost']:.6f}")
        print(f"Estimated total:    ${cost['total_cost']:.6f}")
    print(SPACER)

    if log:
        log_evaluation("evaluate", request, system, [result], log_file=log_file)

    return result

def evaluate_all_models(prompt, models=None, providers=None, system=None, response_width=60, log=False, log_file=LOG_FILE):
    # models wins over providers; with neither, every priced model of every provider runs
    models = models or default_models(providers)
    results = []

    for model in models:
        messages = [{"role": "user", "content": prompt}]
        print(f"Running {model}...")
        results.append(_run_model_safe(model, messages, system=system))

    headers = ["Model", "Provider", "Time (s)", "Input", "Output", "Total", "Cost ($)", "Response"]
    rows = []
    for r in results:
        usage = r["usage"]
        if usage:
            cost = usage["cost"]["total_cost"] if usage["cost"] else None
            text = " ".join(r["response"].split())
            rows.append([
                r["model"],
                r["provider"],
                f"{r['seconds']:.2f}",
                str(usage["input_tokens"]),
                str(usage["output_tokens"]),
                str(usage["total_tokens"]),
                f"{cost:.6f}" if cost is not None else "n/a",
                text if len(text) <= response_width else text[:response_width - 3] + "...",
            ])
        else:
            error = r["error"]
            rows.append([r["model"], r["provider"] or "?", f"{r['seconds']:.2f}", "-", "-", "-", "-",
                         "ERROR: " + (error if len(error) <= response_width else error[:response_width - 3] + "...")])

    widths = [max(len(row[i]) for row in [headers] + rows) for i in range(len(headers))]
    numeric = {2, 3, 4, 5, 6}

    def format_row(row):
        return " | ".join(cell.rjust(widths[i]) if i in numeric else cell.ljust(widths[i]) for i, cell in enumerate(row))

    print()
    print(f"PROMPT: {prompt}")
    print(format_row(headers))
    print("-+-".join("-" * w for w in widths))
    for row in rows:
        print(format_row(row))
    print()

    if log:
        # Logs every table row in full: responses and errors are not truncated
        log_evaluation("evaluate_all_models", prompt, system, results, log_file=log_file)

    return results

def _check_history(history):
    # A chat history from the page: turns alternating user then assistant, ending with an assistant reply.
    # User turns may carry attachments; returns their total size in bytes.
    if not isinstance(history, list) or len(history) % 2:
        raise ValueError("each history must be a list of user/assistant pairs")
    total = 0
    for i, message in enumerate(history):
        user = i % 2 == 0
        allowed_keys = ({"role", "content"}, {"role", "content", "attachments"}) if user else ({"role", "content"},)
        if (not isinstance(message, dict) or set(message) not in allowed_keys
                or message["role"] != ("user" if user else "assistant")
                or not isinstance(message["content"], str)):
            raise ValueError("history messages must alternate user and assistant")
        attachments = message.get("attachments") or []
        total += _check_attachments(attachments)
        # A user turn may be only attachments; every other message needs text
        if not message["content"].strip() and not attachments:
            raise ValueError("history messages need text or attachments")
    return total

class _LogViewerHandler(BaseHTTPRequestHandler):
    # Serves only the pages, the log and the model list - never the project folder, which holds .env
    log_file = LOG_FILE
    # A multi-turn chat resends every model's whole conversation, attachments included, so this allows for long ones
    max_body_bytes = 40_000_000

    def do_GET(self):
        path = self.path.split("?")[0]
        if path in ("/", "/index.html"):
            self._send_file(LOG_VIEWER_PAGE, "text/html; charset=utf-8")
        elif path == "/log.json":
            if os.path.exists(self.log_file):
                self._send_file(self.log_file, "application/json; charset=utf-8")
            else:
                self._send_bytes(b"[]", "application/json; charset=utf-8")
        elif path == "/chat":
            self._send_file(CHAT_PAGE, "text/html; charset=utf-8")
        elif path == "/models":
            models = [{"model": m, "provider": provider_for(m)} for m in default_models()]
            self._send_bytes(json.dumps(models).encode("utf-8"), "application/json; charset=utf-8")
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path.split("?")[0] != "/run":
            self.send_error(404)
            return
        # /run makes billed calls, so only this server's own pages may use it. A page on another site can still
        # send a POST to localhost: its Origin won't match, and a JSON body needs a preflight we never answer.
        # Checking Host as well stops a DNS-rebinding site that points its own name at 127.0.0.1.
        port = self.server.server_address[1]
        own = {f"127.0.0.1:{port}", f"localhost:{port}"}
        if (self.headers.get("Host") not in own
                or self.headers.get("Origin") not in {f"http://{h}" for h in own}
                or self.headers.get_content_type() != "application/json"):
            self.send_error(403)
            return

        try:
            length = int(self.headers.get("Content-Length", 0))
            if not 0 < length <= self.max_body_bytes:
                raise ValueError("missing or oversized body")
            body = json.loads(self.rfile.read(length))
            prompt, system, models = body["prompt"], body.get("system") or None, body["models"]
            attachments = body.get("attachments") or []
            new_bytes = _check_attachments(attachments)
            # Files for the whole conversation, added to each model's first user message (with_context_files)
            context_files = body.get("context_files") or []
            context_bytes = _check_attachments(context_files)
            allowed = set(default_models())
            if (not isinstance(prompt, str) or not (prompt.strip() or attachments)
                    or not (system is None or isinstance(system, str))
                    or not isinstance(models, list) or not models
                    or any(m not in allowed for m in models)):
                raise ValueError("expected a prompt and a list of priced models")
            # Optional, for a multi-turn chat: the session's id and each model's own conversation so far
            session_id = body.get("session_id")
            histories = body.get("histories") or {}
            if not (session_id is None or (isinstance(session_id, str) and 0 < len(session_id) <= 100)):
                raise ValueError("session_id must be a short string")
            if not isinstance(histories, dict) or any(m not in models for m in histories):
                raise ValueError("histories must map requested models to their messages")
            # Each model is sent its own history plus the new prompt, so the limit applies to the largest of them
            history_bytes = max((_check_history(h) for h in histories.values()), default=0)
            if context_bytes + new_bytes + history_bytes > ATTACHMENTS_MAX_TOTAL_BYTES:
                raise ValueError(f"attachments in this conversation are over the {ATTACHMENTS_MAX_TOTAL_BYTES // 1_000_000} MB limit")
        except (ValueError, KeyError, TypeError) as e:
            self.send_error(400, str(e))
            return
        models = list(dict.fromkeys(models))

        # One NDJSON line per model as it finishes, so the page can fill in each tab without waiting for the slowest
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()

        results = {}
        client_gone = False
        with ThreadPoolExecutor(max_workers=min(len(models), 16)) as pool:
            futures = [
                pool.submit(_run_model_safe, m, with_context_files(
                    histories.get(m, []) + [{"role": "user", "content": prompt, "attachments": attachments}], context_files), system)
                for m in models
            ]
            for future in as_completed(futures):
                result = future.result()
                results[result["model"]] = result
                if client_gone:
                    continue
                try:
                    line = json.dumps({"type": "result", "result": result}, ensure_ascii=False, default=str)
                    self.wfile.write(line.encode("utf-8") + b"\n")
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    # The calls are already paid for, so finish them and still log the run
                    client_gone = True

        done = {"type": "done", "logged": False}
        if body.get("log", True):
            try:
                with _log_lock:
                    entry = log_evaluation("chat", prompt, system, [results[m] for m in models],
                                           log_file=self.log_file, session_id=session_id,
                                           attachments=attachment_summary(attachments),
                                           context_files=attachment_summary(context_files))
                done["logged"] = True
                if session_id is not None:
                    done["turn"] = entry["turn"]
                    done["session_total_cost"] = entry["session_total_cost"]
            except (OSError, ValueError) as e:
                done["log_error"] = f"{type(e).__name__}: {e}"
        if not client_gone:
            try:
                self.wfile.write(json.dumps(done).encode("utf-8") + b"\n")
            except (BrokenPipeError, ConnectionResetError):
                pass

    def _send_file(self, file_path, content_type):
        with open(file_path, "rb") as f:
            self._send_bytes(f.read(), content_type)

    def _send_bytes(self, body, content_type):
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        pass

_log_viewer_server = None
# Two chat runs finishing together would otherwise both read the log, and one run would be lost
_log_lock = threading.Lock()

def _ensure_server(log_file, port):
    # One server serves both the log viewer and the chat page
    global _log_viewer_server
    _LogViewerHandler.log_file = log_file

    if _log_viewer_server is None:
        for candidate in range(port, port + 10):
            try:
                _log_viewer_server = ThreadingHTTPServer(("127.0.0.1", candidate), _LogViewerHandler)
                break
            except OSError:
                continue
        else:
            raise OSError(f"No free port between {port} and {port + 9}")
        # Daemon thread so a notebook cell returns straight away and the server stops with the kernel
        threading.Thread(target=_log_viewer_server.serve_forever, daemon=True).start()

    return f"http://127.0.0.1:{_log_viewer_server.server_address[1]}"

def view_logs(log_file=LOG_FILE, port=8765, open_browser=True):
    url = _ensure_server(log_file, port) + "/"
    print(f"Log viewer running at {url} (log: {log_file})")
    if open_browser:
        webbrowser.open(url)
    return url

def chat_ui(log_file=LOG_FILE, port=8765, open_browser=True, block=False):
    # Each prompt sent from the page makes real (billed) calls to every checked model
    url = _ensure_server(log_file, port) + "/chat"
    print(f"Chat UI running at {url} (log: {log_file})")
    if open_browser:
        webbrowser.open(url)
    if block:
        # From a script the daemon server thread would stop as soon as the script ends, so wait here instead
        print("Press Ctrl+C to stop")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
    return url

#evaluate("claude-haiku-5-5", "What is quantum computing? Answer in one sentence")
#evaluate("gpt-6-luna", "What is quantum computing? Answer in one sentence")
#evaluate_all_models("What is quantum computing? Answer in one sentence")
#evaluate_all_models("What is quantum computing? Answer in one sentence", providers=["openai"])
#chat_ui(block=True)