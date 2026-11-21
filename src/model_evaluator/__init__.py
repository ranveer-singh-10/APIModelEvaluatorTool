"""Compare LLMs from several providers on cost and accuracy.

Typical use::

    from model_evaluator import evaluate, evaluate_all_models, evaluate_batch, chat_ui, view_logs

    evaluate("claude-haiku-5-5", "What is quantum computing? Answer in one sentence")
    evaluate_batch("datasets/example.json", models=["claude-haiku-5-5", "gpt-6-luna"], log=True)

Every evaluation makes real, billed API calls.
"""

from .config import load_env

load_env()

from .attachments import attachment_summary, with_context_files  # noqa: E402
from .batch import evaluate_batch, load_cases, summarise_batch  # noqa: E402
from .config import LOG_FILE  # noqa: E402
from .evaluation import evaluate, evaluate_all_models  # noqa: E402
from .logstore import log_evaluation, read_log, set_manual_score  # noqa: E402
from .providers import PROVIDERS, default_models, provider_for  # noqa: E402
from .runner import run_model  # noqa: E402
from .scoring import (  # noqa: E402
    JUDGE_MODEL,
    PASS_SCORE,
    SCORE_MAX,
    SCORE_MIN,
    SCORERS,
    judge_cost,
    score_response,
)
from .server import chat_ui, view_logs  # noqa: E402

__version__ = "0.1.0"

__all__ = [
    "JUDGE_MODEL",
    "LOG_FILE",
    "PASS_SCORE",
    "PROVIDERS",
    "SCORERS",
    "SCORE_MAX",
    "SCORE_MIN",
    "attachment_summary",
    "chat_ui",
    "default_models",
    "evaluate",
    "evaluate_all_models",
    "evaluate_batch",
    "judge_cost",
    "load_cases",
    "log_evaluation",
    "provider_for",
    "read_log",
    "run_model",
    "score_response",
    "set_manual_score",
    "summarise_batch",
    "view_logs",
    "with_context_files",
]
