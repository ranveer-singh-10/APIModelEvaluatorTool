import json

import pytest

from model_evaluator import scoring
from model_evaluator.scoring import (
    check_scoring,
    format_score,
    judge_cost,
    normalise,
    resolve_scorer,
    score_response,
)

from .conftest import make_result


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("  Canberra. ", "canberra"), ("New   York", "new york"), ('"Yes!"', "yes"), (42, "42")],
)
def test_normalise(raw, expected):
    assert normalise(raw) == expected


def test_exact_scorer():
    assert score_response(make_result(response="canberra."), "q", "Canberra", scorer="exact")["score"]["value"] == 5
    assert score_response(make_result(response="Sydney"), "q", "Canberra", scorer="exact")["score"]["value"] == 1


def test_contains_scorer():
    result = make_result(response="The capital is Canberra.")
    assert score_response(result, "q", "canberra", scorer="contains")["score"]["value"] == 5


def test_failed_call_is_not_scored():
    result = score_response(make_result(error="APIError: down"), "q", "Canberra", scorer="exact")
    assert result["score"] is None
    assert format_score(result) == "-"


@pytest.mark.parametrize(
    ("scorer", "expected", "rubric", "message"),
    [
        ("magic", "x", None, "scorer must be one of"),
        ("exact", None, "rubric", "needs an expected answer"),
        ("llm_judge", None, None, "expected answer or a rubric"),
    ],
)
def test_check_scoring_rejects(scorer, expected, rubric, message):
    with pytest.raises(ValueError, match=message):
        check_scoring(scorer, expected, rubric)


def test_resolve_scorer_defaults_to_judge_only_when_needed():
    assert resolve_scorer(None, None, None) is None
    assert resolve_scorer(None, "x", None) == "llm_judge"
    assert resolve_scorer("exact", "x", None) == "exact"


def fake_judge(response_text, cost=0.002, error=None):
    def run(model, messages, system=None, output_config=None):
        result = make_result(model=model, response=response_text, cost=cost, error=error)
        run.calls.append({"model": model, "messages": messages, "system": system, "output_config": output_config})
        return result

    run.calls = []
    return run


def test_llm_judge_reads_the_grade(monkeypatch):
    judge = fake_judge(json.dumps({"reasoning": "Correct city.", "score": 5}))
    monkeypatch.setattr(scoring, "run_model_safe", judge)

    result = score_response(make_result(response="Canberra"), "Capital of Australia?", expected="Canberra")

    score = result["score"]
    assert score["value"] == 5
    assert score["reasoning"] == "Correct city."
    assert score["judge_model"] == scoring.JUDGE_MODEL
    # The grader's cost is kept apart from the model's
    assert result["usage"]["cost"]["total_cost"] == 0.001
    assert judge_cost([result]) == pytest.approx(0.002)
    # The prompt, expected answer and response each sit in their own tag, with structured output requested
    content = judge.calls[0]["messages"][0]["content"]
    assert "<expected_answer>\nCanberra\n</expected_answer>" in content
    assert judge.calls[0]["output_config"] == scoring.JUDGE_OUTPUT


@pytest.mark.parametrize("bad", ["not json", json.dumps({"score": 9, "reasoning": ""}), json.dumps({"reasoning": "x"})])
def test_llm_judge_unreadable_grade(monkeypatch, bad):
    monkeypatch.setattr(scoring, "run_model_safe", fake_judge(bad))
    score = score_response(make_result(), "q", expected="Canberra")["score"]
    assert score["value"] is None
    assert score["error"].startswith("Unreadable grade")


def test_llm_judge_call_failure(monkeypatch):
    monkeypatch.setattr(scoring, "run_model_safe", fake_judge("", error="RateLimitError: slow down"))
    result = score_response(make_result(), "q", rubric="Names the capital")
    assert result["score"]["error"] == "RateLimitError: slow down"
    assert format_score(result) == "err"
