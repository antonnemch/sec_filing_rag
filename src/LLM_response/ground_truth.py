"""Load and filter ground-truth answers from the evaluation CSV."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.utils import PROJECT_ROOT, ticker_slug

DEFAULT_EVAL_CSV = (
    Path(__file__).resolve().parents[2] / "eval_sets" / "faang_eval_set_milestone.csv"
)
REQUIRED_COLUMNS = {
    "qa_id", "question_id", "question", "answer", "ticker", "category",
    "answerable", "source_doc_id",
}
ALLOWED_CATEGORIES = {1, 2, 3, 4}


def load_eval_set(path: Path = DEFAULT_EVAL_CSV) -> pd.DataFrame:
    """Load and validate an evaluation CSV."""
    if not path.exists():
        raise FileNotFoundError(f"Eval set not found: {path}")
    try:
        df = pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="latin-1")

    missing = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing:
        raise ValueError(
            f"Eval set {path} is missing required column(s): {', '.join(missing)}"
        )
    if df["qa_id"].isna().any() or df["qa_id"].astype(str).str.strip().eq("").any():
        raise ValueError(f"Eval set {path} contains a missing or blank qa_id.")
    duplicates = sorted(
        df.loc[df["qa_id"].duplicated(keep=False), "qa_id"].astype(str).unique()
    )
    if duplicates:
        raise ValueError(
            f"Eval set {path} contains duplicate qa_id value(s): "
            + ", ".join(duplicates)
        )
    if df["question"].isna().any() or df["question"].astype(str).str.strip().eq("").any():
        raise ValueError(f"Eval set {path} contains a missing or blank question.")
    if df["source_doc_id"].isna().any() or df["source_doc_id"].astype(str).str.strip().eq("").any():
        raise ValueError(f"Eval set {path} contains a missing or blank source_doc_id.")
    numeric_categories = pd.to_numeric(df["category"], errors="coerce")
    if numeric_categories.isna().any() or not set(numeric_categories.astype(int)).issubset(
        ALLOWED_CATEGORIES
    ):
        raise ValueError(
            f"Eval set {path} contains invalid category values; expected 1, 2, 3, or 4."
        )
    df["category"] = numeric_categories.astype(int)

    def parse_answerable(value: object) -> bool:
        if isinstance(value, bool):
            return value
        normalized = str(value).strip().lower()
        if normalized in {"true", "1", "yes"}:
            return True
        if normalized in {"false", "0", "no"}:
            return False
        raise ValueError

    try:
        df["answerable"] = df["answerable"].apply(parse_answerable)
    except ValueError as exc:
        raise ValueError(
            f"Eval set {path} contains invalid answerable values; expected true or false."
        ) from exc
    normalized_tickers = df["ticker"].astype(str).str.strip().str.upper()
    if normalized_tickers.eq("").any():
        raise ValueError(f"Eval set {path} contains a missing or blank ticker.")
    df["ticker"] = normalized_tickers

    return df


def filter_completed_questions(df: pd.DataFrame) -> pd.DataFrame:
    """Return rows with a non-empty reference answer."""
    completed = df["answer"].notna() & df["answer"].astype(str).str.strip().ne("")
    return df[completed].copy()


def validate_source_doc_ids(
    df: pd.DataFrame,
    project_root: Path = PROJECT_ROOT,
) -> None:
    """Require every gold document ID to exist in its ticker chunk database."""
    mismatches: list[str] = []
    for ticker, ticker_rows in df.groupby("ticker", sort=True):
        normalized_ticker = str(ticker).strip().upper()
        slug = ticker_slug(normalized_ticker)
        chunks_path = (
            project_root / "data" / "processed" / slug / f"{slug}_filing_chunks.csv"
        )
        if not chunks_path.exists():
            raise FileNotFoundError(
                f"Cannot validate source_doc_id values because the chunk database "
                f"is missing for {normalized_ticker}: {chunks_path}"
            )
        try:
            chunks = pd.read_csv(
                chunks_path,
                usecols=["ticker", "filing_type", "filing_date"],
            ).drop_duplicates()
        except (KeyError, ValueError) as exc:
            raise ValueError(
                f"Chunk database {chunks_path} lacks document metadata required "
                "for source_doc_id validation."
            ) from exc

        available = {
            f"{str(row.ticker).strip().upper()}_{str(row.filing_type).strip()}_"
            f"{str(row.filing_date).strip()}"
            for row in chunks.itertuples(index=False)
        }
        has_plural_ids = "source_doc_ids" in ticker_rows.columns
        for row in ticker_rows.itertuples(index=False):
            marker = str(row.source_doc_id).strip()
            plural_value = getattr(row, "source_doc_ids", "") if has_plural_ids else ""
            expected_ids = (
                []
                if pd.isna(plural_value)
                else [item.strip() for item in str(plural_value).split("|") if item.strip()]
            )
            multi_marker = f"{normalized_ticker}_MULTI"
            if marker == multi_marker:
                # MULTI is valid without labels while the question is unfinished.
                # Once exact IDs are supplied, validate every contributing document.
                expected = expected_ids
            else:
                expected = expected_ids or [marker]

            missing_ids = [doc_id for doc_id in expected if doc_id not in available]
            if missing_ids:
                mismatches.append(
                    f"{row.qa_id}: expected {', '.join(missing_ids)}; available: "
                    + (", ".join(sorted(available)) if available else "none")
                )

    if mismatches:
        raise ValueError(
            "Evaluation source_doc_id validation failed:\n  "
            + "\n  ".join(mismatches)
        )
