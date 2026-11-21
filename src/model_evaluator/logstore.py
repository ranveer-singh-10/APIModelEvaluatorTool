"""The run log: one JSON array of runs, newest last, so a web page can load it with a single ``JSON.parse``."""

from __future__ import annotations

import json
import os
import threading
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .config import LOG_FILE
from .runner import Result, call_cost
from .scoring import SCORE_MAX, SCORE_MIN, judge_cost

PathLike = str | os.PathLike[str]

MANUAL_NOTE_MAX = 2000

# Two chat runs finishing together would otherwise both read the log, and one run would be lost
LOG_LOCK = threading.Lock()


def read_log(log_file: PathLike = LOG_FILE) -> list[dict]:
    """Return every logged run, or an empty list if there is no log yet."""
    path = Path(log_file)
    if not path.exists():
        return []
    content = path.read_text(encoding="utf-8").strip()
    if not content:
        return []
    entries = json.loads(content)
    if not isinstance(entries, list):
        raise ValueError(f"{log_file} is not a JSON list of log entries")
    return entries


def write_log(entries: list[dict], log_file: PathLike = LOG_FILE) -> None:
    path = Path(log_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        # default=str covers SDK content blocks when a prompt is passed as a messages list
        json.dump(entries, f, indent=2, ensure_ascii=False, default=str)


def make_entry(
    kind: str, prompt: Any, system: str | None, results: Iterable[Result], extra: dict | None = None
) -> dict[str, Any]:
    """Build a log entry. ``total_cost`` is what the models cost; grading costs are recorded apart as ``judge_cost``."""
    results = list(results)
    entry: dict[str, Any] = {
        "timestamp": datetime.now(UTC).isoformat(),
        "type": kind,
        "prompt": prompt,
        "system": system,
        "total_cost": sum(c for r in results if (c := call_cost(r)) is not None),
    }
    if any(r.get("score") for r in results):
        entry["judge_cost"] = judge_cost(results)
    # Scoring and batch fields (expected, rubric, scorer, batch_id, case_id), left out when not given
    entry.update({k: v for k, v in (extra or {}).items() if v is not None})
    return entry


def log_evaluation(
    kind: str,
    prompt: Any,
    system: str | None,
    results: list[Result],
    log_file: PathLike = LOG_FILE,
    session_id: str | None = None,
    attachments: list[dict] | None = None,
    context_files: list[dict] | None = None,
    extra: dict | None = None,
) -> dict[str, Any]:
    """Append one run to the log and return its entry.

    With a ``session_id`` (a chat turn), the entry also records its ``turn`` number and ``session_total_cost``.
    Attachments are logged as summaries (names, types and sizes), never their data.
    """
    entries = read_log(log_file)
    entry = make_entry(kind, prompt, system, results, extra)
    if attachments:
        entry["attachments"] = attachments
    if context_files:
        entry["context_files"] = context_files
    if session_id is not None:
        # One entry per turn of a multi-turn chat; the session's running total comes from its earlier logged turns
        earlier = [e for e in entries if isinstance(e, dict) and e.get("session_id") == session_id]
        entry["session_id"] = session_id
        entry["turn"] = len(earlier) + 1
        entry["session_total_cost"] = entry["total_cost"] + sum(
            e["total_cost"] for e in earlier if isinstance(e.get("total_cost"), (int, float))
        )
    entry["results"] = results

    entries.append(entry)
    write_log(entries, log_file)
    print(f"Logged to {log_file}")
    return entry


def append_entries(new_entries: list[dict], log_file: PathLike = LOG_FILE) -> None:
    """Append several entries in one write, under the log lock."""
    with LOG_LOCK:
        write_log(read_log(log_file) + new_entries, log_file)


def set_manual_score(
    entry_index: int,
    timestamp: str,
    model: str,
    value: int | None,
    note: str | None = None,
    log_file: PathLike = LOG_FILE,
) -> Result:
    """Grade one logged result by hand (1-5), or clear the manual grade with ``value=None``.

    The entry is found by its index and must still have this ``timestamp``, so a log that changed since it was read
    is refused with ``LookupError`` rather than grading the wrong result. A manual grade replaces the result's score
    and keeps the automatic score it replaced under ``"previous"``; grading again keeps that original, and clearing
    puts it back (or leaves no score). Returns the updated result.
    """
    if value is not None and (type(value) is not int or not SCORE_MIN <= value <= SCORE_MAX):
        raise ValueError(f"a manual grade must be a whole number from {SCORE_MIN} to {SCORE_MAX}")
    if note is not None and (not isinstance(note, str) or len(note) > MANUAL_NOTE_MAX):
        raise ValueError(f"the note must be text of at most {MANUAL_NOTE_MAX} characters")
    note = (note or "").strip() or None
    with LOG_LOCK:
        entries = read_log(log_file)
        if (
            type(entry_index) is not int
            or not 0 <= entry_index < len(entries)
            or not isinstance(entries[entry_index], dict)
            or entries[entry_index].get("timestamp") != timestamp
        ):
            raise LookupError(
                "that run isn't where it was in the log, which has changed since it was loaded: reload it"
            )
        matches = [
            r for r in entries[entry_index].get("results") or [] if isinstance(r, dict) and r.get("model") == model
        ]
        if len(matches) != 1:
            raise LookupError(f"the run has no single result for {model}: reload the log")
        result = matches[0]
        if result.get("error"):
            raise ValueError("a failed call has no response to grade")
        current = result.get("score")
        manual = isinstance(current, dict) and current.get("scorer") == "manual"
        original = current.get("previous") if manual else current
        if value is None:
            # Back to the automatic score, or to no score key at all, as the result was before it was graded
            if original is None:
                result.pop("score", None)
            else:
                result["score"] = original
        else:
            result["score"] = {
                "value": value,
                "scorer": "manual",
                "reasoning": note,
                "judge_model": None,
                "judge_usage": None,
                "error": None,
                "previous": original,
            }
        write_log(entries, log_file)
    return result
