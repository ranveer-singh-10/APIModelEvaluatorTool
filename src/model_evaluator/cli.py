"""Command line interface: ``model-eval <command>`` or ``python -m model_evaluator <command>``."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from . import LOG_FILE, PROVIDERS, SCORERS, default_models, evaluate, evaluate_all_models, evaluate_batch, provider_for
from .server import DEFAULT_PORT, chat_ui, view_logs


def _add_scoring(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--expected", help="the correct answer; turns scoring on")
    parser.add_argument("--rubric", help="what a good answer must do; turns scoring on")
    parser.add_argument("--scorer", choices=SCORERS, help="default: llm_judge when --expected or --rubric is given")


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--system", help="system prompt")
    parser.add_argument("--log", action="store_true", help="append the run to the log file")
    parser.add_argument("--log-file", type=Path, default=LOG_FILE, help=f"default: {LOG_FILE}")


def _add_model_choice(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--models", nargs="+", metavar="MODEL", help="models to run (default: every priced model)")
    parser.add_argument("--providers", nargs="+", choices=list(PROVIDERS), help="run every priced model of these")


def _add_server(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--log-file", type=Path, default=LOG_FILE, help=f"default: {LOG_FILE}")
    parser.add_argument("--no-browser", action="store_true", help="don't open the page in a browser")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="model-eval",
        description="Compare LLMs on cost and accuracy. Commands that run models make real, billed API calls.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("models", help="list the priced models and their providers")
    p.add_argument("--providers", nargs="+", choices=list(PROVIDERS))

    p = sub.add_parser("evaluate", help="send one prompt to one model")
    p.add_argument("model")
    p.add_argument("prompt")
    _add_scoring(p)
    _add_common(p)

    p = sub.add_parser("compare", help="send one prompt to many models and print a comparison table")
    p.add_argument("prompt")
    _add_model_choice(p)
    _add_scoring(p)
    _add_common(p)
    p.add_argument("--width", type=int, default=60, help="response column width")

    p = sub.add_parser("batch", help="run a dataset of cases against many models in parallel")
    p.add_argument("cases", help="a .json or .jsonl file of {prompt, expected?, rubric?, id?}")
    _add_model_choice(p)
    p.add_argument("--scorer", choices=[*SCORERS, "none"], default="llm_judge")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("-y", "--yes", action="store_true", help="don't ask before making the billed calls")
    _add_common(p)

    p = sub.add_parser("chat", help="open the multi-model chat page")
    _add_server(p)

    p = sub.add_parser("logs", help="open the log viewer")
    _add_server(p)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "models":
        for model in default_models(args.providers):
            print(f"{provider_for(model):<11} {model}")
    elif args.command == "evaluate":
        evaluate(
            args.model,
            args.prompt,
            system=args.system,
            expected=args.expected,
            rubric=args.rubric,
            scorer=args.scorer,
            log=args.log,
            log_file=args.log_file,
        )
    elif args.command == "compare":
        evaluate_all_models(
            args.prompt,
            models=args.models,
            providers=args.providers,
            system=args.system,
            expected=args.expected,
            rubric=args.rubric,
            scorer=args.scorer,
            response_width=args.width,
            log=args.log,
            log_file=args.log_file,
        )
    elif args.command == "batch":
        evaluate_batch(
            args.cases,
            models=args.models,
            providers=args.providers,
            system=args.system,
            scorer=None if args.scorer == "none" else args.scorer,
            max_workers=args.workers,
            confirm=not args.yes,
            log=args.log,
            log_file=args.log_file,
        )
    elif args.command == "chat":
        chat_ui(log_file=args.log_file, port=args.port, open_browser=not args.no_browser, block=True)
    elif args.command == "logs":
        view_logs(log_file=args.log_file, port=args.port, open_browser=not args.no_browser, block=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
