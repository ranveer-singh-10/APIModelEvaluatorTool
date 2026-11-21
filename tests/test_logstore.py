import json

import pytest

from model_evaluator.logstore import log_evaluation, read_log, set_manual_score

from .conftest import make_result


def test_read_log_missing_or_empty(log_file):
    assert read_log(log_file) == []
    log_file.parent.mkdir(parents=True)
    log_file.write_text("  ", encoding="utf-8")
    assert read_log(log_file) == []


def test_read_log_rejects_non_list(log_file):
    log_file.parent.mkdir(parents=True)
    log_file.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError):
        read_log(log_file)


def test_log_evaluation_totals_and_session(log_file):
    results = [make_result(cost=0.01), make_result(model="gpt-6-luna", cost=0.02), make_result(error="Boom")]
    first = log_evaluation("chat", "hi", None, results, log_file=log_file, session_id="abc")
    second = log_evaluation("chat", "again", None, [make_result(cost=0.03)], log_file=log_file, session_id="abc")

    assert first["total_cost"] == pytest.approx(0.03)
    assert (first["turn"], second["turn"]) == (1, 2)
    assert second["session_total_cost"] == pytest.approx(0.06)
    assert "judge_cost" not in first
    assert len(json.loads(log_file.read_text(encoding="utf-8"))) == 2


def test_log_evaluation_leaves_out_unset_extras(log_file):
    entry = log_evaluation("evaluate", "q", None, [make_result()], log_file=log_file, extra={"rubric": None, "x": 1})
    assert "rubric" not in entry and entry["x"] == 1


@pytest.fixture
def graded_log(log_file):
    auto = {"value": 3, "scorer": "exact", "reasoning": None, "judge_model": None, "judge_usage": None, "error": None}
    result = {**make_result(), "score": auto}
    entry = log_evaluation("evaluate", "q", None, [result, make_result(model="gpt-6-luna", error="E")],
                           log_file=log_file)  # fmt: skip
    return log_file, entry["timestamp"], auto


def test_manual_grade_keeps_and_restores_the_automatic_score(graded_log):
    log_file, ts, auto = graded_log

    graded = set_manual_score(0, ts, "claude-haiku-5-5", 5, note="  spot on ", log_file=log_file)
    assert graded["score"]["scorer"] == "manual"
    assert graded["score"]["reasoning"] == "spot on"
    assert graded["score"]["previous"] == auto

    regraded = set_manual_score(0, ts, "claude-haiku-5-5", 2, log_file=log_file)
    assert regraded["score"]["previous"] == auto  # still the original, not the first manual grade

    cleared = set_manual_score(0, ts, "claude-haiku-5-5", None, log_file=log_file)
    assert cleared["score"] == auto
    assert read_log(log_file)[0]["results"][0]["score"] == auto


def test_clearing_an_unscored_result_removes_the_score(log_file):
    ts = log_evaluation("evaluate", "q", None, [make_result()], log_file=log_file)["timestamp"]
    set_manual_score(0, ts, "claude-haiku-5-5", 4, log_file=log_file)
    assert "score" not in set_manual_score(0, ts, "claude-haiku-5-5", None, log_file=log_file)


@pytest.mark.parametrize(
    ("index", "timestamp", "model", "value", "error"),
    [
        (0, "wrong-timestamp", "claude-haiku-5-5", 4, LookupError),
        (5, None, "claude-haiku-5-5", 4, LookupError),
        (0, None, "not-in-run", 4, LookupError),
        (0, None, "gpt-6-luna", 4, ValueError),  # a failed call can't be graded
        (0, None, "claude-haiku-5-5", 6, ValueError),
        (0, None, "claude-haiku-5-5", True, ValueError),
    ],
)
def test_manual_grade_refuses(graded_log, index, timestamp, model, value, error):
    log_file, ts, _ = graded_log
    with pytest.raises(error):
        set_manual_score(index, timestamp or ts, model, value, log_file=log_file)


def test_manual_grade_note_limit(graded_log):
    log_file, ts, _ = graded_log
    with pytest.raises(ValueError):
        set_manual_score(0, ts, "claude-haiku-5-5", 4, note="x" * 2001, log_file=log_file)
