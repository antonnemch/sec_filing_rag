"""Load and filter ground-truth answers from the evaluation CSV."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

_DEFAULT_EVAL_CSV = (
    Path(__file__).resolve().parents[2] / "eval_sets" / "faang_eval_set_dummy.csv"
)


def load_eval_set(path: Path = _DEFAULT_EVAL_CSV) -> pd.DataFrame:
    """Return the evaluation DataFrame. Raises FileNotFoundError if missing."""
    if not path.exists():
        raise FileNotFoundError(f"Eval set not found: {path}")
    return pd.read_csv(path, encoding="latin-1")


def filter_by_ticker(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Return rows for a single ticker (case-insensitive)."""
    return df[df["ticker"].str.upper() == ticker.upper()].copy()
