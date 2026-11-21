"""Paths and environment settings shared across the package."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

LOG_FILE_ENV = "MODEL_EVAL_LOG_FILE"


def load_env() -> None:
    """Load API keys from the nearest ``.env``, searching up from the working directory.

    Variables already set in the environment win over the file.
    """
    load_dotenv(find_dotenv(usecwd=True))


def find_project_root(start: Path | None = None) -> Path:
    """The nearest folder at or above ``start`` (default: the working directory) holding ``pyproject.toml``.

    Falls back to ``start`` itself, so scripts and the notebook (run from ``notebook/``) share one log.
    """
    start = (start or Path.cwd()).resolve()
    for folder in (start, *start.parents):
        if (folder / "pyproject.toml").is_file():
            return folder
    return start


PROJECT_ROOT = find_project_root()
# Overridable with MODEL_EVAL_LOG_FILE; otherwise logs/eval_log.json in the project folder
LOG_FILE = Path(os.environ.get(LOG_FILE_ENV) or PROJECT_ROOT / "logs" / "eval_log.json")
