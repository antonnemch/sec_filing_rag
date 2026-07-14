"""Create aggregate data descriptions for processed SEC filing chunks."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from .chunk_filings import CHUNK_COLUMNS
from .utils import (
    DEFAULT_TICKERS,
    PROJECT_ROOT,
    dataset_slug_for_tickers,
    normalize_tickers,
    project_relative,
    read_json,
    read_jsonl,
    ticker_slug,
    write_json,
)


def _as_percent(numerator: int | float, denominator: int | float) -> float:
    """Return a rounded percentage with zero-denominator protection."""

    return round(100 * numerator / denominator, 2) if denominator else 0.0


def _missing_count(series: pd.Series) -> int:
    """Count missing or empty-string values in a pandas Series."""

    return int(series.isna().sum() + (series.astype(str).str.strip() == "").sum())


def _stats(series: pd.Series) -> dict[str, float | int]:
    """Return compact descriptive statistics for a numeric series."""

    if series.empty:
        return {
            "min": 0,
            "p25": 0.0,
            "median": 0.0,
            "mean": 0.0,
            "p75": 0.0,
            "max": 0,
        }
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return {
            "min": 0,
            "p25": 0.0,
            "median": 0.0,
            "mean": 0.0,
            "p75": 0.0,
            "max": 0,
        }
    return {
        "min": int(numeric.min()),
        "p25": round(float(numeric.quantile(0.25)), 2),
        "median": round(float(numeric.median()), 2),
        "mean": round(float(numeric.mean()), 2),
        "p75": round(float(numeric.quantile(0.75)), 2),
        "max": int(numeric.max()),
    }


def _markdown_table(rows: Iterable[dict[str, Any]], columns: list[str]) -> str:
    """Render rows as a small GitHub-flavored Markdown table."""

    rows = list(rows)
    if not rows:
        return "_No rows._\n"

    def clean(value: Any) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")

    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(clean(row.get(col, "")) for col in columns) + " |")
    return "\n".join(lines) + "\n"


def _load_dataset_frames(
    tickers: Iterable[str],
    project_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load per-ticker manifests and chunks into DataFrames."""

    raw_rows: list[dict[str, Any]] = []
    cleaning_rows: list[dict[str, Any]] = []
    chunk_rows: list[dict[str, Any]] = []

    for ticker in normalize_tickers(tickers):
        slug = ticker_slug(ticker)
        raw_manifest = read_json(
            project_root / "data" / "raw" / slug / "filing_metadata.json"
        )
        cleaning_manifest = read_json(
            project_root / "data" / "processed" / slug / "cleaning_manifest.json"
        )
        chunks = read_jsonl(
            project_root / "data" / "processed" / slug / f"{slug}_filing_chunks.jsonl"
        )
        raw_rows.extend(raw_manifest)
        cleaning_rows.extend(cleaning_manifest)
        chunk_rows.extend(chunks)

    raw_df = pd.DataFrame(raw_rows)
    cleaning_df = pd.DataFrame(cleaning_rows)
    chunk_df = pd.DataFrame(chunk_rows, columns=CHUNK_COLUMNS)
    if not cleaning_df.empty:
        cleaning_df["retained_ratio"] = (
            pd.to_numeric(cleaning_df["cleaned_length"], errors="coerce")
            / pd.to_numeric(cleaning_df["raw_length"], errors="coerce")
        )
    return raw_df, cleaning_df, chunk_df


def _write_aggregate_tables(
    raw_df: pd.DataFrame,
    cleaning_df: pd.DataFrame,
    chunk_df: pd.DataFrame,
    dataset_slug: str,
    project_root: Path,
    chunk_size: int,
) -> dict[str, Path]:
    """Write aggregate CSV summary tables and return their paths."""

    summary_directory = project_root / "outputs" / "data_summary"
    summary_directory.mkdir(parents=True, exist_ok=True)

    paths = {
        "filing_inventory": summary_directory / f"{dataset_slug}_filing_inventory.csv",
        "cleaning_summary": summary_directory / f"{dataset_slug}_cleaning_summary.csv",
        "chunk_summary": (
            summary_directory / f"{dataset_slug}_chunk_summary_by_ticker_form.csv"
        ),
        "missingness_summary": (
            summary_directory / f"{dataset_slug}_missingness_summary.csv"
        ),
        "outlier_summary": summary_directory / f"{dataset_slug}_outlier_summary.csv",
        "section_summary": summary_directory / f"{dataset_slug}_section_summary.csv",
    }

    raw_df.to_csv(paths["filing_inventory"], index=False)

    cleaning_output = cleaning_df.copy()
    if not cleaning_output.empty and "retained_ratio" in cleaning_output:
        cleaning_output["retained_pct"] = (
            cleaning_output["retained_ratio"].fillna(0) * 100
        ).round(2)
    cleaning_output.to_csv(paths["cleaning_summary"], index=False)

    chunk_summary = (
        chunk_df.groupby(["ticker", "filing_type"], dropna=False)
        .agg(
            chunks=("chunk_id", "count"),
            filings=("accession_number", "nunique"),
            total_words=("word_count", "sum"),
            mean_words=("word_count", "mean"),
            median_words=("word_count", "median"),
            min_words=("word_count", "min"),
            max_words=("word_count", "max"),
            unknown_heading_chunks=(
                "section_heading",
                lambda values: int((values == "unknown").sum()),
            ),
        )
        .reset_index()
    )
    for column in ["mean_words", "median_words"]:
        chunk_summary[column] = chunk_summary[column].round(2)
    chunk_summary.to_csv(paths["chunk_summary"], index=False)

    missing_rows: list[dict[str, Any]] = []
    for dataset_name, frame, fields in (
        ("filing_manifest", raw_df, list(raw_df.columns)),
        ("cleaning_manifest", cleaning_df, list(cleaning_df.columns)),
        ("chunks", chunk_df, CHUNK_COLUMNS),
    ):
        for field in fields:
            missing = _missing_count(frame[field]) if field in frame else len(frame)
            missing_rows.append(
                {
                    "dataset": dataset_name,
                    "field": field,
                    "rows": len(frame),
                    "missing_count": missing,
                    "missing_pct": _as_percent(missing, len(frame)),
                }
            )
    pd.DataFrame(missing_rows).to_csv(paths["missingness_summary"], index=False)

    duplicate_chunk_ids = (
        int(chunk_df["chunk_id"].duplicated().sum()) if "chunk_id" in chunk_df else 0
    )
    source_missing = (
        sum(not (project_root / str(path)).exists() for path in chunk_df["source_file"])
        if "source_file" in chunk_df
        else 0
    )
    raw_missing = (
        sum(not (project_root / str(path)).exists() for path in raw_df["local_raw_path"])
        if "local_raw_path" in raw_df
        else 0
    )
    human_missing = (
        sum(
            not (project_root / str(path)).exists()
            for path in raw_df.get("local_markdown_path", pd.Series(dtype=str))
            if str(path).strip()
        )
        if not raw_df.empty
        else 0
    )
    fallback_text = (
        int((raw_df.get("human_readable_format", pd.Series(dtype=str)) == "text").sum())
        if not raw_df.empty
        else 0
    )
    low_retention = (
        int((cleaning_df.get("retained_ratio", pd.Series(dtype=float)) < 0.15).sum())
        if not cleaning_df.empty
        else 0
    )
    outlier_rows = [
        {
            "metric": "duplicate_chunk_ids",
            "count": duplicate_chunk_ids,
            "handling": "Validation flag; rows are not dropped automatically.",
        },
        {
            "metric": "empty_chunk_text_rows",
            "count": int((chunk_df["text"].astype(str).str.strip() == "").sum()),
            "handling": "Validation flag; rows are not dropped automatically.",
        },
        {
            "metric": "missing_chunk_source_files",
            "count": source_missing,
            "handling": "Validation flag; required source files should exist.",
        },
        {
            "metric": "oversized_chunks",
            "count": int((pd.to_numeric(chunk_df["word_count"]) > chunk_size).sum()),
            "handling": "Validation failure if nonzero.",
        },
        {
            "metric": "short_chunks_under_100_words",
            "count": int((pd.to_numeric(chunk_df["word_count"]) < 100).sum()),
            "handling": "Kept because short sections can be meaningful.",
        },
        {
            "metric": "low_retention_filings_under_15_pct",
            "count": low_retention,
            "handling": "Flagged because SEC HTML includes substantial markup/tables.",
        },
        {
            "metric": "missing_raw_files",
            "count": raw_missing,
            "handling": "Validation flag; raw files should exist.",
        },
        {
            "metric": "missing_human_readable_files",
            "count": human_missing,
            "handling": "Validation flag; human-readable files should exist.",
        },
        {
            "metric": "human_readable_text_fallbacks",
            "count": fallback_text,
            "handling": "Kept as text when native Markdown rendering is unavailable.",
        },
    ]
    pd.DataFrame(outlier_rows).to_csv(paths["outlier_summary"], index=False)

    section_summary = (
        chunk_df.groupby("section_heading", dropna=False)
        .agg(
            chunks=("chunk_id", "count"),
            tickers=("ticker", lambda values: ", ".join(sorted(set(values)))),
            filing_types=(
                "filing_type",
                lambda values: ", ".join(sorted(set(values))),
            ),
        )
        .reset_index()
        .sort_values("chunks", ascending=False)
    )
    section_summary["chunk_pct"] = (
        section_summary["chunks"] / max(len(chunk_df), 1) * 100
    ).round(2)
    section_summary.to_csv(paths["section_summary"], index=False)

    return paths


def _write_markdown_report(
    raw_df: pd.DataFrame,
    cleaning_df: pd.DataFrame,
    chunk_df: pd.DataFrame,
    dataset_slug: str,
    project_root: Path,
    output_paths: dict[str, Path],
) -> Path:
    """Write a human-readable Markdown data description."""

    summary_directory = project_root / "outputs" / "data_summary"
    report_path = summary_directory / f"{dataset_slug}_data_description.md"

    filing_counts = (
        raw_df.groupby(["ticker", "filing_type"])
        .size()
        .reset_index(name="filings")
        .sort_values(["ticker", "filing_type"])
    )
    filing_dates = raw_df[
        ["ticker", "filing_type", "filing_date", "accession_number"]
    ].sort_values(["ticker", "filing_type", "filing_date"])
    file_counts = {
        "raw_files": len(raw_df),
        "cleaned_files": len(cleaning_df),
        "human_readable_files": int(
            raw_df.get("local_markdown_path", pd.Series(dtype=str))
            .fillna("")
            .astype(str)
            .str.strip()
            .ne("")
            .sum()
        ),
    }
    character_stats = [
        {"text_version": "raw", **_stats(cleaning_df["raw_length"])},
        {"text_version": "cleaned", **_stats(cleaning_df["cleaned_length"])},
    ]
    word_stats = _stats(chunk_df["word_count"])
    known_headings = int((chunk_df["section_heading"] != "unknown").sum())
    unknown_headings = int((chunk_df["section_heading"] == "unknown").sum())
    section_top = (
        chunk_df["section_heading"]
        .value_counts()
        .head(10)
        .rename_axis("section_heading")
        .reset_index(name="chunks")
    )
    retained_stats = _stats((cleaning_df["retained_ratio"] * 100).round(2))
    human_counts = (
        raw_df.get("human_readable_format", pd.Series(dtype=str))
        .value_counts()
        .rename_axis("format")
        .reset_index(name="files")
    )
    missingness = pd.read_csv(output_paths["missingness_summary"])
    outliers = pd.read_csv(output_paths["outlier_summary"])
    chunk_counts = pd.read_csv(output_paths["chunk_summary"])

    lines = [
        f"# {dataset_slug.upper()} SEC Filing Data Description",
        "",
        f"Generated at: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Companies included",
        "",
        (
            f"The dataset contains {len(raw_df)} filings for "
            f"{', '.join(sorted(raw_df['ticker'].unique()))}. Each ticker includes "
            "the latest unamended 10-K, latest unamended 10-Q, and recent 8-K filings."
        ),
        "",
        "## Filing counts by ticker and form",
        "",
        _markdown_table(
            filing_counts.to_dict("records"),
            ["ticker", "filing_type", "filings"],
        ),
        "",
        "## Filing dates and accession numbers",
        "",
        _markdown_table(
            filing_dates.to_dict("records"),
            ["ticker", "filing_type", "filing_date", "accession_number"],
        ),
        "",
        "## Raw, cleaned, and human-readable file counts",
        "",
        _markdown_table(
            [file_counts],
            ["raw_files", "cleaned_files", "human_readable_files"],
        ),
        "",
        "Human-readable format counts:",
        "",
        _markdown_table(human_counts.to_dict("records"), ["format", "files"]),
        "",
        "## Raw and cleaned character statistics",
        "",
        _markdown_table(
            character_stats,
            ["text_version", "min", "p25", "median", "mean", "p75", "max"],
        ),
        "",
        "## Retained-text ratios",
        "",
        _markdown_table([retained_stats], ["min", "p25", "median", "mean", "p75", "max"]),
        "",
        "## Chunk counts by ticker and form",
        "",
        _markdown_table(
            chunk_counts.to_dict("records"),
            [
                "ticker",
                "filing_type",
                "chunks",
                "filings",
                "total_words",
                "mean_words",
                "median_words",
                "min_words",
                "max_words",
                "unknown_heading_chunks",
            ],
        ),
        "",
        "## Chunk word-count statistics",
        "",
        f"Total chunks: {len(chunk_df)}.",
        "",
        _markdown_table([word_stats], ["min", "p25", "median", "mean", "p75", "max"]),
        "",
        "## Section-heading coverage and most common sections",
        "",
        (
            f"Known-heading chunks: {known_headings} "
            f"({_as_percent(known_headings, len(chunk_df))}%). "
            f"Unknown-heading chunks: {unknown_headings} "
            f"({_as_percent(unknown_headings, len(chunk_df))}%)."
        ),
        "",
        _markdown_table(section_top.to_dict("records"), ["section_heading", "chunks"]),
        "",
        "## Missing metadata counts",
        "",
        (
            "Missing metadata is counted, not imputed. Rows with fully present "
            "metadata are omitted from this compact table."
        ),
        "",
        _markdown_table(
            missingness[missingness["missing_count"] > 0].to_dict("records"),
            ["dataset", "field", "rows", "missing_count", "missing_pct"],
        ),
        "",
        "## Missing data policy",
        "",
        (
            "No filing text is imputed. Required manifests and source files are hard "
            "failures when missing. Undetected section headings remain `unknown`."
        ),
        "",
        "## Data quality and outlier checks",
        "",
        (
            "The following checks quantify duplicate chunk IDs, empty chunk text, "
            "missing files, oversized chunks, short chunks, low retained-text ratios, "
            "and human-readable fallback behavior."
        ),
        "",
        _markdown_table(
            outliers.to_dict("records"),
            ["metric", "count", "handling"],
        ),
        "",
        "## Outlier policy",
        "",
        (
            "No chunks are dropped for being short. Chunks above the configured "
            "chunk size are validation failures. Low cleaning-retention ratios, "
            "very short chunks, and unusual file sizes are flagged in summaries, "
            "not removed."
        ),
        "",
        "## Generated summary tables",
        "",
        "\n".join(
            f"- `{project_relative(path, project_root)}`"
            for path in output_paths.values()
        ),
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def describe_dataset(
    tickers: Iterable[str] = DEFAULT_TICKERS,
    dataset_slug: str | None = None,
    chunk_size: int = 400,
    project_root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    """Generate aggregate summary tables and a Markdown data description."""

    normalized_tickers = normalize_tickers(tickers)
    slug = dataset_slug or dataset_slug_for_tickers(normalized_tickers)
    raw_df, cleaning_df, chunk_df = _load_dataset_frames(
        normalized_tickers,
        project_root,
    )
    output_paths = _write_aggregate_tables(
        raw_df,
        cleaning_df,
        chunk_df,
        slug,
        project_root,
        chunk_size,
    )
    report_path = _write_markdown_report(
        raw_df,
        cleaning_df,
        chunk_df,
        slug,
        project_root,
        output_paths,
    )

    summary_path = project_root / "outputs" / "data_summary" / f"{slug}_dataset_summary.json"
    summary = {
        "dataset": slug,
        "tickers": list(normalized_tickers),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "filings_downloaded": len(raw_df),
        "filing_type_counts": dict(Counter(raw_df["filing_type"])),
        "cleaned_files": len(cleaning_df),
        "human_readable_files": int(
            raw_df.get("local_markdown_path", pd.Series(dtype=str))
            .fillna("")
            .astype(str)
            .str.strip()
            .ne("")
            .sum()
        ),
        "total_chunks": len(chunk_df),
        "average_words_per_chunk": round(float(chunk_df["word_count"].mean()), 2)
        if not chunk_df.empty
        else 0.0,
        "word_count_stats": _stats(chunk_df["word_count"]),
        "section_heading_coverage_pct": _as_percent(
            int((chunk_df["section_heading"] != "unknown").sum()),
            len(chunk_df),
        ),
        "output_paths": {
            "combined_chunks_csv": f"data/processed/{slug}/{slug}_filing_chunks.csv",
            "combined_chunks_jsonl": f"data/processed/{slug}/{slug}_filing_chunks.jsonl",
            **{
                key: project_relative(path, project_root)
                for key, path in output_paths.items()
            },
            "data_description": project_relative(report_path, project_root),
            "dataset_summary": f"outputs/data_summary/{slug}_dataset_summary.json",
        },
    }
    write_json(summary_path, summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    """Create the data-description command-line parser."""

    parser = argparse.ArgumentParser(
        description="Describe a processed SEC filing dataset."
    )
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=list(DEFAULT_TICKERS),
        help="Tickers to describe. Defaults to FAANG.",
    )
    parser.add_argument(
        "--dataset-slug",
        default=None,
        help="Output prefix. Defaults to faang for the default ticker set.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=400,
        help="Expected maximum chunk size for outlier checks.",
    )
    return parser


def main() -> None:
    """Run the data-description CLI."""

    parser = build_parser()
    args = parser.parse_args()
    try:
        summary = describe_dataset(
            tickers=args.tickers,
            dataset_slug=args.dataset_slug,
            chunk_size=args.chunk_size,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        parser.exit(1, f"Error: {exc}\n")

    print(f"Data description complete: {summary['output_paths']['data_description']}")


if __name__ == "__main__":
    main()
