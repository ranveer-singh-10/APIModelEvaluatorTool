"""Running a dataset of cases against many models in parallel and summarising accuracy against cost."""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .config import LOG_FILE, PROJECT_ROOT
from .evaluation import print_table
from .logstore import PathLike, append_entries, make_entry
from .providers import default_models
from .runner import Result, call_cost, run_model_safe
from .scoring import (
    JUDGE_MODEL,
    PASS_SCORE,
    SCORE_MAX,
    check_scoring,
    format_score,
    judge_cost,
    score_response,
    score_value,
    scoring_fields,
)

CASE_KEYS = {"id", "prompt", "expected", "rubric"}


def _resolve_path(path: PathLike) -> Path:
    # Relative paths are tried from the working directory, then the project folder
    candidate = Path(path)
    if not candidate.exists() and not candidate.is_absolute() and (PROJECT_ROOT / candidate).exists():
        return PROJECT_ROOT / candidate
    return candidate


def load_cases(cases: list[dict] | PathLike) -> list[dict]:
    """Load and check evaluation cases.

    ``cases`` is a list of ``{"prompt", "expected"?, "rubric"?, "id"?}``, or the path of a ``.json`` file holding that
    list or a ``.jsonl`` file with one case per line. Missing ids become ``case-N``.
    """
    if isinstance(cases, (str, os.PathLike)):
        path = _resolve_path(cases)
        with path.open(encoding="utf-8") as f:
            if path.suffix.lower() == ".jsonl":
                cases = [json.loads(line) for line in f if line.strip()]
            else:
                cases = json.load(f)
    if not isinstance(cases, list) or not cases:
        raise ValueError("cases must be a non-empty list of {prompt, expected?, rubric?, id?}")
    checked, ids = [], set()
    for i, case in enumerate(cases, 1):
        if (
            not isinstance(case, dict)
            or not set(case) <= CASE_KEYS
            or not isinstance(case.get("prompt"), str)
            or not case["prompt"].strip()
            or any(case.get(k) is not None and not isinstance(case[k], str) for k in ("expected", "rubric"))
        ):
            raise ValueError(f"case {i} needs a prompt, with optional expected, rubric and id strings")
        case_id = str(case.get("id") or f"case-{i}")
        if case_id in ids:
            raise ValueError(f"case id {case_id} is used twice")
        ids.add(case_id)
        checked.append(
            {
                "id": case_id,
                "prompt": case["prompt"],
                "expected": case.get("expected") or None,
                "rubric": case.get("rubric") or None,
            }
        )
    return checked


def summarise_batch(models: Sequence[str], results: list[Result]) -> list[dict[str, Any]]:
    """One row per model: how accurate it was against what it cost.

    Failed calls count as errors, not wrong answers. ``cost_per_pass`` is the cost of the scored calls divided by the
    passes (grading excluded).
    """
    summary = []
    for model in models:
        rs = [r for r in results if r["model"] == model]
        ok = [r for r in rs if not r["error"]]
        costs = [c for r in ok if (c := call_cost(r)) is not None]
        scored = [r for r in ok if score_value(r) is not None]
        scores = [score_value(r) for r in scored]
        passes = sum(s >= PASS_SCORE for s in scores)
        scored_cost = sum(c for r in scored if (c := call_cost(r)) is not None)
        summary.append(
            {
                "model": model,
                "provider": rs[0]["provider"] if rs else None,
                "cases": len(rs),
                "errors": len(rs) - len(ok),
                "scored": len(scored),
                "avg_score": sum(scores) / len(scores) if scores else None,
                "pass_rate": passes / len(scores) if scores else None,
                "total_cost": sum(costs) if costs else None,
                "avg_cost": sum(costs) / len(costs) if costs else None,
                "cost_per_pass": scored_cost / passes if passes else None,
                "avg_seconds": sum(r["seconds"] for r in ok) / len(ok) if ok else None,
            }
        )
    return summary


def _print_summary(batch_id: str, cases: list[dict], summary: list[dict], results: list[Result], scorer, judge_model):
    def money(v: float | None) -> str:
        return f"{v:.6f}" if v is not None else "-"

    def number(v: float | None, fmt: str) -> str:
        return format(v, fmt) if v is not None else "-"

    headers = ["Model", "Provider", "Cases", "Errors", "Avg score", "Pass rate", "Total ($)", "Avg ($)",
               "Per pass ($)", "Avg time (s)"]  # fmt: skip
    rows = [
        [
            s["model"],
            s["provider"] or "?",
            str(s["cases"]),
            str(s["errors"]),
            number(s["avg_score"], ".2f"),
            number(s["pass_rate"], ".0%"),
            money(s["total_cost"]),
            money(s["avg_cost"]),
            money(s["cost_per_pass"]),
            number(s["avg_seconds"], ".2f"),
        ]
        for s in summary
    ]
    print()
    print(f"BATCH {batch_id}: {len(cases)} cases" + (f", scored by {scorer}" if scorer else ""))
    print_table(headers, rows, numeric=set(range(2, 10)))
    model_cost = sum(s["total_cost"] or 0 for s in summary)
    grader = f", grader cost ${judge_cost(results):.6f} ({judge_model})" if scorer == "llm_judge" else ""
    print(f"Model cost ${model_cost:.6f}{grader}")
    if scorer:
        print(f"Pass = score {PASS_SCORE}+ out of {SCORE_MAX}; per pass = cost of the scored calls / passes")
    print()


def evaluate_batch(
    cases: list[dict] | PathLike,
    models: Sequence[str] | None = None,
    providers: Iterable[str] | None = None,
    system: str | None = None,
    scorer: str | None = "llm_judge",
    judge_model: str = JUDGE_MODEL,
    max_workers: int = 8,
    confirm: bool = True,
    log: bool = False,
    log_file: PathLike = LOG_FILE,
) -> dict[str, Any] | None:
    """Run every case against every model in parallel, score each response and print accuracy against cost.

    ``scorer=None`` runs the cases without scoring. With ``confirm=True`` it asks before making the billed calls.
    With ``log=True`` each case is logged as one ``"batch"`` entry, all sharing a ``batch_id``.

    Returns ``{"batch_id", "summary" (one row per model), "results" (one per case and model)}``, or ``None`` if
    cancelled.
    """
    cases = load_cases(cases)
    models = list(dict.fromkeys(models or default_models(providers)))
    if scorer is not None:
        for case in cases:
            try:
                check_scoring(scorer, case["expected"], case["rubric"])
            except ValueError as e:
                raise ValueError(f"case {case['id']}: {e}") from None

    calls = len(cases) * len(models)
    grading = f" plus {calls} grading calls to {judge_model}" if scorer == "llm_judge" else ""
    print(f"Batch: {len(cases)} cases x {len(models)} models = {calls} billed calls{grading}")
    if confirm and input("Run it? [y/N] ").strip().lower() not in ("y", "yes"):
        print("Cancelled; nothing was run")
        return None

    batch_id = datetime.now(UTC).strftime("%Y%m%d-%H%M%S-") + os.urandom(2).hex()

    def run_one(case: dict, model: str) -> tuple[str, Result]:
        result = run_model_safe(model, [{"role": "user", "content": case["prompt"]}], system=system)
        if scorer:
            score_response(
                result,
                case["prompt"],
                expected=case["expected"],
                rubric=case["rubric"],
                scorer=scorer,
                judge_model=judge_model,
            )
        return case["id"], result

    by_case: dict[str, dict[str, Result]] = {case["id"]: {} for case in cases}
    with ThreadPoolExecutor(max_workers=max(1, min(max_workers, calls))) as pool:
        futures = [pool.submit(run_one, case, model) for case in cases for model in models]
        for done, future in enumerate(as_completed(futures), 1):
            case_id, result = future.result()
            by_case[case_id][result["model"]] = result
            status = "ERROR" if result["error"] else format_score(result) if scorer else f"{result['seconds']:.1f}s"
            print(f"[{done}/{calls}] {case_id} {result['model']}: {status}")

    # In case then model order, whichever call finished first
    results = [{"case_id": case["id"], **by_case[case["id"]][m]} for case in cases for m in models]
    summary = summarise_batch(models, results)
    _print_summary(batch_id, cases, summary, results, scorer, judge_model)

    if log:
        new_entries = []
        for case in cases:
            case_results = [
                {k: v for k, v in r.items() if k != "case_id"} for r in results if r["case_id"] == case["id"]
            ]
            extra = {"batch_id": batch_id, "case_id": case["id"]}
            if scorer:
                extra.update(scoring_fields(case["expected"], case["rubric"], scorer, judge_model))
            entry = make_entry("batch", case["prompt"], system, case_results, extra)
            entry["results"] = case_results
            new_entries.append(entry)
        append_entries(new_entries, log_file)
        print(f"Logged {len(new_entries)} cases to {log_file} (batch {batch_id})")

    return {"batch_id": batch_id, "summary": summary, "results": results}
