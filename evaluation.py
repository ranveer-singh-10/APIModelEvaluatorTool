import os
import time
import json
import threading
import webbrowser
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

def run_model(model, messages, system=None):
    # One non-streaming call through the provider's chat(), returning the fields every evaluation records
    provider_name = provider_for(model)
    provider = PROVIDERS[provider_name]
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

def log_evaluation(kind, prompt, system, results, log_file=LOG_FILE):
    # The log file is one JSON array of runs, newest last, so a webpage can load it with a single JSON.parse
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "type": kind,
        "prompt": prompt,
        "system": system,
        "total_cost": sum(
            r["usage"]["cost"]["total_cost"] for r in results if r["usage"] and r["usage"]["cost"]
        ),
        "results": results,
    }

    entries = []
    if os.path.exists(log_file):
        with open(log_file, encoding="utf-8") as f:
            content = f.read().strip()
        if content:
            entries = json.loads(content)
            if not isinstance(entries, list):
                raise ValueError(f"{log_file} is not a JSON list of log entries")

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
        start = time.perf_counter()
        try:
            results.append(run_model(model, messages, system=system))
        except Exception as e:
            # Keep going so one unavailable model doesn't stop the comparison
            try:
                provider_name = provider_for(model)
            except ValueError:
                provider_name = None
            results.append({
                "model": model,
                "provider": provider_name,
                "seconds": time.perf_counter() - start,
                "stop_reason": None,
                "response": "",
                "usage": None,
                "error": f"{type(e).__name__}: {e}",
            })

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

class _LogViewerHandler(BaseHTTPRequestHandler):
    # Serves only the viewer page and the log - never the project folder, which holds .env
    log_file = LOG_FILE

    def do_GET(self):
        path = self.path.split("?")[0]
        if path in ("/", "/index.html"):
            self._send_file(LOG_VIEWER_PAGE, "text/html; charset=utf-8")
        elif path == "/log.json":
            if os.path.exists(self.log_file):
                self._send_file(self.log_file, "application/json; charset=utf-8")
            else:
                self._send_bytes(b"[]", "application/json; charset=utf-8")
        else:
            self.send_error(404)

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

def view_logs(log_file=LOG_FILE, port=8765, open_browser=True):
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

    url = f"http://127.0.0.1:{_log_viewer_server.server_address[1]}/"
    print(f"Log viewer running at {url} (log: {log_file})")
    if open_browser:
        webbrowser.open(url)
    return url

#evaluate("claude-haiku-5-5", "What is quantum computing? Answer in one sentence")
#evaluate("gpt-6-luna", "What is quantum computing? Answer in one sentence")
#evaluate_all_models("What is quantum computing? Answer in one sentence")
#evaluate_all_models("What is quantum computing? Answer in one sentence", providers=["openai"])