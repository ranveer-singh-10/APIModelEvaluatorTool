import json

import pytest

from model_evaluator import batch
from model_evaluator.batch import evaluate_batch, load_cases, summarise_batch
from model_evaluator.logstore import read_log

from .conftest import make_result


def test_load_cases_fills_in_ids():
    cases = load_cases([{"prompt": "a", "expected": "1"}, {"id": "named", "prompt": "b", "rubric": "r"}])
    assert [c["id"] for c in cases] == ["case-1", "named"]
    assert cases[0]["rubric"] is None


def test_load_cases_from_jsonl(tmp_path):
    path = tmp_path / "cases.jsonl"
    path.write_text('{"prompt": "a"}\n\n{"prompt": "b"}\n', encoding="utf-8")
    assert len(load_cases(path)) == 2


@pytest.mark.parametrize("dataset", ["datasets/example.json", "datasets/hard.json"])
def test_bundled_datasets_are_valid(dataset):
    cases = load_cases(dataset)
    assert all(c["expected"] or c["rubric"] for c in cases)


@pytest.mark.parametrize(
    "cases",
    [[], [{"prompt": ""}], [{"prompt": "a", "unknown": 1}], [{"prompt": "a", "expected": 5}],
     [{"id": "x", "prompt": "a"}, {"id": "x", "prompt": "b"}]],
)  # fmt: skip
def test_load_cases_rejects(cases):
    with pytest.raises(ValueError):
        load_cases(cases)


def scored(model, value, cost):
    return {**make_result(model=model, cost=cost), "score": {"value": value, "judge_usage": None}}


def test_summarise_batch():
    results = [
        scored("m1", 5, 0.01),
        scored("m1", 2, 0.03),
        make_result(model="m1", error="boom"),
        scored("m2", 4, 0.02),
    ]
    summary = {row["model"]: row for row in summarise_batch(["m1", "m2"], results)}

    m1 = summary["m1"]
    assert (m1["cases"], m1["errors"], m1["scored"]) == (3, 1, 2)
    assert m1["avg_score"] == 3.5
    assert m1["pass_rate"] == 0.5
    assert m1["total_cost"] == pytest.approx(0.04)
    assert m1["cost_per_pass"] == pytest.approx(0.04)  # both scored calls' cost, one pass
    assert summary["m2"]["pass_rate"] == 1.0


def test_evaluate_batch_runs_scores_and_logs(monkeypatch, log_file):
    def fake_run(model, messages, system=None, output_config=None):
        answer = "Canberra" if model == "good" else "Sydney"
        return make_result(model=model, response=answer, cost=0.01)

    monkeypatch.setattr(batch, "run_model_safe", fake_run)
    cases = [
        {"id": "capital", "prompt": "Capital of Australia?", "expected": "Canberra"},
        {"prompt": "x", "expected": "y"},
    ]

    out = evaluate_batch(cases, models=["good", "bad"], scorer="exact", confirm=False, log=True, log_file=log_file)

    summary = {row["model"]: row for row in out["summary"]}
    assert summary["good"]["pass_rate"] == 0.5
    assert summary["bad"]["pass_rate"] == 0.0
    # One entry per case, sharing the batch id, results in model order
    entries = read_log(log_file)
    assert [e["case_id"] for e in entries] == ["capital", "case-2"]
    assert {e["batch_id"] for e in entries} == {out["batch_id"]}
    assert [r["model"] for r in entries[0]["results"]] == ["good", "bad"]
    assert entries[0]["scorer"] == "exact" and "judge_model" not in entries[0]
    assert json.dumps(entries)  # serialisable


def test_evaluate_batch_checks_cases_before_running(monkeypatch):
    monkeypatch.setattr(batch, "run_model_safe", pytest.fail)
    with pytest.raises(ValueError, match="case-1"):
        evaluate_batch([{"prompt": "no expected answer"}], models=["m"], scorer="exact", confirm=False)


def test_evaluate_batch_can_be_cancelled(monkeypatch):
    monkeypatch.setattr(batch, "run_model_safe", pytest.fail)
    monkeypatch.setattr("builtins.input", lambda prompt: "n")
    assert evaluate_batch([{"prompt": "a", "expected": "b"}], models=["m"], scorer="exact") is None
