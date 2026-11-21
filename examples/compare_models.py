"""Compare models on one prompt, then on a small dataset, scoring each answer.

Makes real (billed) API calls, including grading calls to the judge model. Run with:  python examples/compare_models.py
"""

from model_evaluator import evaluate_all_models, evaluate_batch

# One prompt, two cheap models, scored against an expected answer by the LLM judge
evaluate_all_models(
    "What is the capital of Australia? Answer with just the city name.",
    models=["claude-haiku-5-5", "gpt-6-luna"],
    expected="Canberra",
)

# A whole dataset in parallel: prints average score, pass rate and cost per pass for each model.
# Asks for confirmation before it runs.
evaluate_batch("datasets/example.json", models=["claude-haiku-5-5", "gpt-6-luna"])
