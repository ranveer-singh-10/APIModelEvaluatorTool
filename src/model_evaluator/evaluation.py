"""Evaluating a prompt against one model, or comparing it across many."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from .config import LOG_FILE
from .logstore import PathLike, log_evaluation
from .providers import default_models
from .runner import Result, run_model, run_model_safe
from .scoring import (
    JUDGE_MODEL,
    format_score,
    is_pass,
    judge_cost,
    resolve_scorer,
    score_response,
    scoring_fields,
)

SPACER = "-" * 26


def print_table(headers: Sequence[str], rows: list[Sequence[str]], numeric: set[int]) -> None:
    """Print a plain-text table; columns in ``numeric`` are right-aligned."""
    widths = [max(len(row[i]) for row in [headers, *rows]) for i in range(len(headers))]

    def format_row(row: Sequence[str]) -> str:
        return " | ".join(
            cell.rjust(widths[i]) if i in numeric else cell.ljust(widths[i]) for i, cell in enumerate(row)
        )

    print(format_row(headers))
    print("-+-".join("-" * w for w in widths))
    for row in rows:
        print(format_row(row))


def _truncate(text: str, width: int) -> str:
    return text if len(text) <= width else text[: width - 3] + "..."


def evaluate(
    model: str,
    request: str | list,
    system: str | None = None,
    expected: str | None = None,
    rubric: str | None = None,
    scorer: str | None = None,
    judge_model: str = JUDGE_MODEL,
    log: bool = False,
    log_file: PathLike = LOG_FILE,
) -> Result:
    """Send one prompt (or a messages list) to one model and print the response, tokens and estimated cost.

    With ``expected`` and/or ``rubric`` the response is also scored (see :func:`score_response`).
    """
    messages = [{"role": "user", "content": request}] if isinstance(request, str) else request
    scorer = resolve_scorer(scorer, expected, rubric)

    result = run_model(model, messages, system=system)
    if scorer:
        score_response(result, request, expected=expected, rubric=rubric, scorer=scorer, judge_model=judge_model)
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
    score = result.get("score")
    if score:
        print(SPACER)
        print(f"Score ({score['scorer']}): {format_score(result)}" + (" - pass" if is_pass(result) else ""))
        if score["reasoning"]:
            print(f"Grader: {score['reasoning']}")
        if score["error"]:
            print(f"Grading failed: {score['error']}")
        if score["judge_usage"] and score["judge_usage"]["cost"]:
            grader_cost = score["judge_usage"]["cost"]["total_cost"]
            print(f"Grader cost:        ${grader_cost:.6f} ({score['judge_model']}, not in the total above)")
    print(SPACER)

    if log:
        log_evaluation(
            "evaluate",
            request,
            system,
            [result],
            log_file=log_file,
            extra=scoring_fields(expected, rubric, scorer, judge_model) if scorer else None,
        )
    return result


def evaluate_all_models(
    prompt: str,
    models: Sequence[str] | None = None,
    providers: Iterable[str] | None = None,
    system: str | None = None,
    expected: str | None = None,
    rubric: str | None = None,
    scorer: str | None = None,
    judge_model: str = JUDGE_MODEL,
    response_width: int = 60,
    log: bool = False,
    log_file: PathLike = LOG_FILE,
) -> list[Result]:
    """Send one prompt to many models in turn and print a comparison table.

    ``models`` wins over ``providers``; with neither, every priced model of every configured provider runs. With
    ``expected`` and/or ``rubric``, each response is also scored and the table gains a Score column.
    """
    models = list(models or default_models(providers))
    scorer = resolve_scorer(scorer, expected, rubric)
    results = []

    for model in models:
        print(f"Running {model}...")
        result = run_model_safe(model, [{"role": "user", "content": prompt}], system=system)
        if scorer:
            score_response(result, prompt, expected=expected, rubric=rubric, scorer=scorer, judge_model=judge_model)
        results.append(result)

    score_header = ["Score"] if scorer else []
    headers = ["Model", "Provider", "Time (s)", "Input", "Output", "Total", "Cost ($)", *score_header, "Response"]
    rows = []
    for r in results:
        usage = r["usage"]
        score_cell = [format_score(r)] if scorer else []
        if usage:
            cost = usage["cost"]["total_cost"] if usage["cost"] else None
            rows.append(
                [
                    r["model"],
                    r["provider"],
                    f"{r['seconds']:.2f}",
                    str(usage["input_tokens"]),
                    str(usage["output_tokens"]),
                    str(usage["total_tokens"]),
                    f"{cost:.6f}" if cost is not None else "n/a",
                    *score_cell,
                    _truncate(" ".join(r["response"].split()), response_width),
                ]
            )
        else:
            rows.append(
                [
                    r["model"],
                    r["provider"] or "?",
                    f"{r['seconds']:.2f}",
                    "-",
                    "-",
                    "-",
                    "-",
                    *score_cell,
                    "ERROR: " + _truncate(r["error"], response_width),
                ]
            )

    print()
    print(f"PROMPT: {prompt}")
    print_table(headers, rows, numeric={2, 3, 4, 5, 6, 7} if scorer else {2, 3, 4, 5, 6})
    if scorer:
        grader = f" ({judge_model}), grader cost ${judge_cost(results):.6f}" if scorer == "llm_judge" else ""
        print(f"Scored by {scorer}{grader}")
    print()

    if log:
        # Logs every table row in full: responses and errors are not truncated
        log_evaluation(
            "evaluate_all_models",
            prompt,
            system,
            results,
            log_file=log_file,
            extra=scoring_fields(expected, rubric, scorer, judge_model) if scorer else None,
        )
    return results
