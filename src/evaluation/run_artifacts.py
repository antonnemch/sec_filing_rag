"""Paths and naming rules for isolated evaluation-run artifacts."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path


RESULTS_FILENAME = "eval_results.csv"
_RUN_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def runs_root(project_root: Path) -> Path:
    return project_root / "outputs" / "eval_results" / "runs"


def validate_run_name(run_name: str) -> str:
    """Validate a filesystem-safe, single-directory run name."""

    if not _RUN_NAME_RE.fullmatch(run_name) or run_name in {".", ".."}:
        raise ValueError(
            "Run names must start with a letter or number and contain only "
            "letters, numbers, dots, underscores, and hyphens."
        )
    return run_name


def new_run_name(now: datetime | None = None) -> str:
    """Create a sortable UTC run name with microseconds to avoid collisions."""

    instant = now or datetime.now(timezone.utc)
    return instant.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")


def output_for_run(project_root: Path, run_name: str) -> Path:
    return runs_root(project_root) / validate_run_name(run_name) / RESULTS_FILENAME


def newest_run_output(project_root: Path) -> Path:
    """Return the most recently modified completed/materialized run output."""

    candidates = list(runs_root(project_root).glob(f"*/{RESULTS_FILENAME}"))
    if not candidates:
        raise FileNotFoundError(
            "No evaluation runs were found. Run src.evaluation.run_eval first or "
            "pass --input explicitly."
        )
    return max(candidates, key=lambda path: path.stat().st_mtime_ns)
