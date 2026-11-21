"""Local web server for the chat page and the log viewer.

It listens on 127.0.0.1 only and answers just ``GET /`` (the log viewer), ``/log.json``, ``/chat`` and ``/models``,
plus ``POST /run`` and ``POST /score``. Everything else is a 404, so it never exposes the project folder or ``.env``.
"""

from __future__ import annotations

import contextlib
import json
import socket
import threading
import time
import webbrowser
from concurrent.futures import ThreadPoolExecutor, as_completed
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from pathlib import Path
from typing import Any

from .attachments import ATTACHMENTS_MAX_TOTAL_BYTES, check_attachments, total_bytes, with_context_files
from .config import LOG_FILE
from .logstore import LOG_LOCK, PathLike, log_evaluation, set_manual_score
from .providers import default_models, provider_for
from .runner import run_model_safe

DEFAULT_PORT = 8765
JSON_TYPE = "application/json; charset=utf-8"
HTML_TYPE = "text/html; charset=utf-8"


def _page(name: str) -> bytes:
    return (resources.files("model_evaluator") / "web" / name).read_bytes()


def check_history(
    history: list, files: dict, file_summaries: dict, has_context: bool = False
) -> tuple[list[dict], int, set[str]]:
    """Validate one model's chat history sent by the page.

    Turns alternate user then assistant, ending with an assistant reply. User turns refer to their attachments by id
    (``attachment_ids``) in the request's ``files``, so a file in every model's history is sent once rather than once
    per model. Returns the history with those attachments filled in, their total size in bytes, and the ids it used.
    """
    if not isinstance(history, list) or len(history) % 2:
        raise ValueError("each history must be a list of user/assistant pairs")
    resolved, total, used = [], 0, set()
    for i, message in enumerate(history):
        user = i % 2 == 0
        allowed_keys = ({"role", "content"}, {"role", "content", "attachment_ids"}) if user else ({"role", "content"},)
        if (
            not isinstance(message, dict)
            or set(message) not in allowed_keys
            or message["role"] != ("user" if user else "assistant")
            or not isinstance(message["content"], str)
        ):
            raise ValueError("history messages must alternate user and assistant")
        ids = message.get("attachment_ids") or []
        if not isinstance(ids, list) or any(not isinstance(x, str) or x not in files for x in ids):
            raise ValueError("attachment_ids must refer to the request's files")
        total += sum(file_summaries[x]["bytes"] for x in ids)
        used.update(ids)
        # Every message needs text, except a user turn with attachments, or the first one when context files go with it
        if not message["content"].strip() and not ids and not (has_context and i == 0):
            raise ValueError("history messages need text or attachments")
        entry: dict[str, Any] = {"role": message["role"], "content": message["content"]}
        if ids:
            entry["attachments"] = [files[x] for x in ids]
        resolved.append(entry)
    return resolved, total, used


class RequestHandler(BaseHTTPRequestHandler):
    """Serves only the pages, the log and the model list - never the project folder, which holds ``.env``."""

    log_file: PathLike = LOG_FILE
    # A multi-turn chat resends every model's whole conversation. Each history file is sent once (files), so the
    # attachments are at most ATTACHMENTS_MAX_TOTAL_BYTES plus a third for base64, which leaves room for the text.
    max_body_bytes = 40_000_000
    # /score only edits the log, so its body is small
    max_score_body_bytes = 100_000

    def do_GET(self) -> None:
        path = self.path.split("?")[0]
        if path in ("/", "/index.html"):
            self._send_bytes(_page("log_viewer.html"), HTML_TYPE)
        elif path == "/log.json":
            log_path = Path(self.log_file)
            self._send_bytes(log_path.read_bytes() if log_path.exists() else b"[]", JSON_TYPE)
        elif path == "/chat":
            self._send_bytes(_page("chat.html"), HTML_TYPE)
        elif path == "/models":
            models = [{"model": m, "provider": provider_for(m)} for m in default_models()]
            self._send_bytes(json.dumps(models).encode("utf-8"), JSON_TYPE)
        else:
            self.send_error(404)

    def _from_own_page(self) -> bool:
        # /run makes billed calls and /score edits the log, so only this server's own pages may use them. A page on
        # another site can still send a POST to localhost: its Origin won't match, and a JSON body needs a preflight
        # we never answer. Checking Host as well stops a DNS-rebinding site that points its own name at 127.0.0.1.
        port = self.server.server_address[1]
        own = {f"127.0.0.1:{port}", f"localhost:{port}"}
        return (
            self.headers.get("Host") in own
            and self.headers.get("Origin") in {f"http://{h}" for h in own}
            and self.headers.get_content_type() == "application/json"
        )

    def do_POST(self) -> None:
        path = self.path.split("?")[0]
        if path not in ("/run", "/score"):
            self.send_error(404)
            return
        if not self._from_own_page():
            self.send_error(403)
            return
        if path == "/score":
            self._score()
        else:
            self._run()

    def _score(self) -> None:
        # A manual grade from the log viewer: {entry, timestamp, model, value (1-5 or null to clear), note}
        try:
            length = int(self.headers.get("Content-Length", 0))
            if not 0 < length <= self.max_score_body_bytes:
                raise ValueError("missing or oversized body")
            body = json.loads(self.rfile.read(length))
            if (
                not isinstance(body, dict)
                or not isinstance(body.get("model"), str)
                or not isinstance(body.get("timestamp"), str)
            ):
                raise ValueError("expected entry, timestamp, model, value and note")
            result = set_manual_score(
                body.get("entry"),
                body["timestamp"],
                body["model"],
                body.get("value"),
                body.get("note"),
                log_file=self.log_file,
            )
        except LookupError as e:
            self.send_error(409, str(e))
            return
        except (ValueError, TypeError) as e:
            self.send_error(400, str(e))
            return
        except OSError as e:
            self.send_error(500, f"couldn't write the log: {e}")
            return
        self._send_bytes(json.dumps({"result": result}, ensure_ascii=False, default=str).encode("utf-8"), JSON_TYPE)

    def _parse_run(self) -> dict[str, Any]:
        """Read and validate a /run body. Raises ``ValueError``, ``KeyError`` or ``TypeError`` on bad input."""
        length = int(self.headers.get("Content-Length", 0))
        if length <= 0:
            raise ValueError("missing body")
        if length > self.max_body_bytes:
            raise ValueError(f"body over the {self.max_body_bytes // 1_000_000} MB limit")
        body = json.loads(self.rfile.read(length))
        prompt, system, models = body["prompt"], body.get("system") or None, body["models"]
        attachments = body.get("attachments") or []
        attachment_summaries = check_attachments(attachments)
        # Files for the whole conversation, added to each model's first user message (with_context_files)
        context_files = body.get("context_files") or []
        context_summaries = check_attachments(context_files)
        # The histories' attachments, each sent once and referred to by id
        files = body.get("files") or {}
        if not isinstance(files, dict) or any(not 0 < len(k) <= 100 for k in files):
            raise ValueError("files must map short ids to attachments")
        file_summaries = dict(zip(files, check_attachments(list(files.values())), strict=True))
        histories = body.get("histories") or {}
        if not isinstance(histories, dict):
            raise ValueError("histories must map requested models to their messages")
        # With context files, the first message may be empty: the context is what it sends
        first_turn = not any(histories.values())
        allowed = set(default_models())
        if (
            not isinstance(prompt, str)
            or not (prompt.strip() or attachments or (context_files and first_turn))
            or not (system is None or isinstance(system, str))
            or not isinstance(models, list)
            or not models
            or any(m not in allowed for m in models)
        ):
            raise ValueError("expected a prompt and a list of priced models")
        # Optional, for a multi-turn chat: the session's id and each model's own conversation so far
        session_id = body.get("session_id")
        if not (session_id is None or (isinstance(session_id, str) and 0 < len(session_id) <= 100)):
            raise ValueError("session_id must be a short string")
        if any(m not in models for m in histories):
            raise ValueError("histories must map requested models to their messages")
        checked = {
            m: check_history(h, files, file_summaries, has_context=bool(context_files)) for m, h in histories.items()
        }
        if set(files) - set().union(*(used for _, _, used in checked.values())):
            raise ValueError("files holds attachments no history refers to")
        # Each model is sent its own history plus the new prompt, so the limit applies to the largest of them
        history_bytes = max((size for _, size, _ in checked.values()), default=0)
        if (
            total_bytes(context_summaries) + total_bytes(attachment_summaries) + history_bytes
            > ATTACHMENTS_MAX_TOTAL_BYTES
        ):
            raise ValueError(
                f"attachments in this conversation are over the {ATTACHMENTS_MAX_TOTAL_BYTES // 1_000_000} MB limit"
            )
        return {
            "prompt": prompt,
            "system": system,
            "models": list(dict.fromkeys(models)),
            "attachments": attachments,
            "attachment_summaries": attachment_summaries,
            "context_files": context_files,
            "context_summaries": context_summaries,
            "histories": {m: resolved for m, (resolved, _, _) in checked.items()},
            "session_id": session_id,
            "log": body.get("log", True),
        }

    def _run(self) -> None:
        try:
            req = self._parse_run()
        except (ValueError, KeyError, TypeError) as e:
            self.send_error(400, str(e))
            return
        models = req["models"]

        # One NDJSON line per model as it finishes, so the page can fill in each tab without waiting for the slowest
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()

        new_turn = {"role": "user", "content": req["prompt"], "attachments": req["attachments"]}
        results = {}
        client_gone = False
        with ThreadPoolExecutor(max_workers=min(len(models), 16)) as pool:
            futures = [
                pool.submit(
                    run_model_safe,
                    m,
                    with_context_files([*req["histories"].get(m, []), new_turn], req["context_files"]),
                    req["system"],
                )
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

        done: dict[str, Any] = {"type": "done", "logged": False}
        if req["log"]:
            try:
                with LOG_LOCK:
                    entry = log_evaluation(
                        "chat",
                        req["prompt"],
                        req["system"],
                        [results[m] for m in models],
                        log_file=self.log_file,
                        session_id=req["session_id"],
                        attachments=req["attachment_summaries"],
                        context_files=req["context_summaries"],
                    )
                done["logged"] = True
                if req["session_id"] is not None:
                    done["turn"] = entry["turn"]
                    done["session_total_cost"] = entry["session_total_cost"]
            except (OSError, ValueError) as e:
                done["log_error"] = f"{type(e).__name__}: {e}"
        if not client_gone:
            with contextlib.suppress(BrokenPipeError, ConnectionResetError):
                self.wfile.write(json.dumps(done).encode("utf-8") + b"\n")

    def _send_bytes(self, body: bytes, content_type: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - matches the base class signature
        pass


class LocalServer(ThreadingHTTPServer):
    """A threaded HTTP server that binds its port exclusively on Windows.

    ``HTTPServer`` turns on ``SO_REUSEADDR``. On Linux and macOS that only lets a restarted server rebind quickly, but
    on Windows it lets a second server listen on a port that is already in use: both then answer requests, so a stale
    viewer (say, in a notebook kernel) or any other local program could take requests meant for this one, ``/run``
    included. So on Windows the port is bound exclusively: binding fails if the port is in use, and nothing else can
    bind it while this server holds it. :func:`ensure_server` then moves on to the next port.
    """

    if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
        allow_reuse_address = False

        def server_bind(self) -> None:
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            super().server_bind()


_server: LocalServer | None = None


def ensure_server(log_file: PathLike = LOG_FILE, port: int = DEFAULT_PORT) -> str:
    """Start the shared server (once) in a daemon thread and return its base URL."""
    global _server
    RequestHandler.log_file = log_file

    if _server is None:
        for candidate in range(port, port + 10):
            try:
                _server = LocalServer(("127.0.0.1", candidate), RequestHandler)
                break
            except OSError:
                continue
        else:
            raise OSError(f"No free port between {port} and {port + 9}")
        if candidate != port:
            print(f"Port {port} is in use (another viewer, perhaps in a notebook kernel?); using {candidate}")
        # Daemon thread so a notebook cell returns straight away and the server stops with the kernel
        threading.Thread(target=_server.serve_forever, daemon=True).start()

    return f"http://127.0.0.1:{_server.server_address[1]}"


def _wait_for_ctrl_c() -> None:
    # From a script the daemon server thread would stop as soon as the script ends, so wait here instead
    print("Press Ctrl+C to stop")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass


def view_logs(
    log_file: PathLike = LOG_FILE, port: int = DEFAULT_PORT, open_browser: bool = True, block: bool = False
) -> str:
    """Serve the log viewer and open it in the browser. Returns its URL."""
    url = ensure_server(log_file, port) + "/"
    print(f"Log viewer running at {url} (log: {log_file})")
    if open_browser:
        webbrowser.open(url)
    if block:
        _wait_for_ctrl_c()
    return url


def chat_ui(
    log_file: PathLike = LOG_FILE, port: int = DEFAULT_PORT, open_browser: bool = True, block: bool = False
) -> str:
    """Serve the chat page and open it in the browser. Each prompt sent from it makes real, billed calls."""
    url = ensure_server(log_file, port) + "/chat"
    print(f"Chat UI running at {url} (log: {log_file})")
    if open_browser:
        webbrowser.open(url)
    if block:
        _wait_for_ctrl_c()
    return url
