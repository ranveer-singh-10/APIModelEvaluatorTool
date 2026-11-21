"""The local server, on a real port. Model calls are faked, so nothing is billed."""

import http.client
import json
import threading

import pytest

from model_evaluator import server
from model_evaluator.logstore import log_evaluation, read_log
from model_evaluator.server import LocalServer, RequestHandler, check_history

from .conftest import make_result


@pytest.fixture
def running(log_file, monkeypatch, no_openrouter_key):
    calls = []

    def fake_run(model, messages, system=None, output_config=None):
        calls.append({"model": model, "messages": messages, "system": system})
        return make_result(model=model, response=f"hi from {model}", cost=0.01)

    monkeypatch.setattr(server, "run_model_safe", fake_run)
    handler = type("Handler", (RequestHandler,), {"log_file": log_file})
    httpd = LocalServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()
    yield httpd.server_address[1], log_file, calls
    httpd.shutdown()
    httpd.server_close()


def request(port, method, path, body=None, headers=None):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    own = {"Host": f"127.0.0.1:{port}", "Origin": f"http://127.0.0.1:{port}", "Content-Type": "application/json"}
    data = json.dumps(body).encode() if body is not None else None
    conn.request(method, path, body=data, headers={**own, **(headers or {})})
    response = conn.getresponse()
    return response.status, response.read()


@pytest.mark.parametrize("path", ["/", "/chat"])
def test_serves_pages(running, path):
    status, body = request(running[0], "GET", path)
    assert status == 200 and body.startswith(b"<!")


def test_log_json_is_empty_list_without_a_log(running):
    assert request(running[0], "GET", "/log.json") == (200, b"[]")


def test_models_lists_configured_models(running):
    status, body = request(running[0], "GET", "/models")
    models = json.loads(body)
    assert status == 200
    assert {"model": "claude-haiku-5-5", "provider": "anthropic"} in models
    assert all(m["provider"] != "openrouter" for m in models)


@pytest.mark.parametrize("path", ["/.env", "/pyproject.toml", "/../.env", "/web/chat.html"])
def test_nothing_else_is_served(running, path):
    assert request(running[0], "GET", path)[0] == 404


@pytest.mark.parametrize(
    "headers",
    [
        {"Origin": "http://evil.example"},
        {"Host": "evil.example:80"},
        {"Content-Type": "text/plain"},
    ],
)
def test_run_refuses_other_origins(running, headers):
    port, _, calls = running
    status, _ = request(port, "POST", "/run", {"prompt": "hi", "models": ["claude-haiku-5-5"]}, headers)
    assert status == 403
    assert calls == []


def test_run_refuses_unpriced_models(running):
    port, _, calls = running
    assert request(port, "POST", "/run", {"prompt": "hi", "models": ["claude-unknown"]})[0] == 400
    assert calls == []


def test_run_streams_results_and_logs_the_turn(running):
    port, log_file, calls = running
    body = {
        "prompt": "next",
        "system": "be brief",
        "models": ["claude-haiku-5-5", "gpt-6-luna"],
        "session_id": "s1",
        "histories": {"claude-haiku-5-5": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]},
    }
    status, raw = request(port, "POST", "/run", body)

    lines = [json.loads(line) for line in raw.decode().splitlines()]
    assert status == 200
    assert {line["result"]["model"] for line in lines[:-1]} == {"claude-haiku-5-5", "gpt-6-luna"}
    assert lines[-1] == {"type": "done", "logged": True, "turn": 1, "session_total_cost": pytest.approx(0.02)}
    # Each model gets its own history plus the new prompt
    sent = {c["model"]: c["messages"] for c in calls}
    assert [m["content"] for m in sent["claude-haiku-5-5"]] == ["q", "a", "next"]
    assert [m["content"] for m in sent["gpt-6-luna"]] == ["next"]
    entry = read_log(log_file)[0]
    assert (entry["type"], entry["session_id"], entry["system"]) == ("chat", "s1", "be brief")


def test_score_endpoint(running):
    port, log_file, _ = running
    ts = log_evaluation("evaluate", "q", None, [make_result()], log_file=log_file)["timestamp"]

    status, body = request(
        port, "POST", "/score", {"entry": 0, "timestamp": ts, "model": "claude-haiku-5-5", "value": 4}
    )
    assert status == 200 and json.loads(body)["result"]["score"]["value"] == 4

    stale = {"entry": 0, "timestamp": "old", "model": "claude-haiku-5-5", "value": 4}
    assert request(port, "POST", "/score", stale)[0] == 409
    bad = {"entry": 0, "timestamp": ts, "model": "claude-haiku-5-5", "value": 9}
    assert request(port, "POST", "/score", bad)[0] == 400


# ---------- check_history ----------

FILE = {"type": "text", "name": "a.txt", "media_type": "text/plain", "data": "hello"}


def test_check_history_resolves_attachment_ids():
    history = [{"role": "user", "content": "", "attachment_ids": ["f1"]}, {"role": "assistant", "content": "ok"}]
    resolved, size, used = check_history(history, {"f1": FILE}, {"f1": {"bytes": 5}})
    assert resolved[0]["attachments"] == [FILE]
    assert (size, used) == (5, {"f1"})


@pytest.mark.parametrize(
    "history",
    [
        [{"role": "user", "content": "q"}],  # odd length
        [{"role": "assistant", "content": "a"}, {"role": "user", "content": "q"}],  # wrong order
        [{"role": "user", "content": "q", "attachment_ids": ["missing"]}, {"role": "assistant", "content": "a"}],
        [{"role": "user", "content": " "}, {"role": "assistant", "content": "a"}],  # no text or files
    ],
)
def test_check_history_rejects(history):
    with pytest.raises(ValueError):
        check_history(history, {}, {})


def test_check_history_allows_context_only_first_turn():
    history = [{"role": "user", "content": ""}, {"role": "assistant", "content": "a"}]
    check_history(history, {}, {}, has_context=True)
