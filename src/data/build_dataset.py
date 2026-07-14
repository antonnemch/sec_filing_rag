"""Run SEC filing download, cleaning, chunking, and reporting pipelines."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from .chunk_filings import CHUNK_COLUMNS, chunk_filings
from .clean_filings import clean_filings
from .describe_dataset import describe_dataset
from .download_filings import download_filings
from .utils import (
    DEFAULT_MULTI_COMPANY_SLUG,
    DEFAULT_TICKERS,
    PROJECT_ROOT,
    dataset_slug_for_tickers,
    normalize_ticker,
    normalize_tickers,
    project_relative,
    read_jsonl,
    ticker_slug,
    validate_chunk_settings,
    validate_num_8k,
    write_json,
    write_jsonl,
)


def resolve_requested_tickers(
    ticker: str | None = None,
    tickers: Iterable[str] | None = None,
) -> tuple[str, ...]:
    """Resolve mutually exclusive single- and multi-ticker requests."""

    if ticker and tickers:
        raise ValueError("Use either --ticker or --tickers, not both.")
    if ticker:
        return (normalize_ticker(ticker),)
    if tickers is not None:
        return normalize_tickers(tickers)
    return DEFAULT_TICKERS


def build_single_dataset(
    ticker: str = "AMZN",
    num_8k: int = 5,
    chunk_size: int = 400,
    chunk_overlap: int = 75,
    project_root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    """Run every data stage for one ticker and write its compact summary."""

    normalized_ticker = normalize_ticker(ticker)
    validate_num_8k(num_8k)
    validate_chunk_settings(chunk_size, chunk_overlap)

    records = download_filings(
        ticker=normalized_ticker,
        num_8k=num_8k,
        project_root=project_root,
    )
    cleaned_records = clean_filings(
        ticker=normalized_ticker,
        project_root=project_root,
    )
    chunks = chunk_filings(
        ticker=normalized_ticker,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        project_root=project_root,
    )

    slug = ticker_slug(normalized_ticker)
    processed_directory = project_root / "data" / "processed" / slug
    summary_directory = project_root / "outputs" / "data_summary"
    summary_path = summary_directory / f"{slug}_dataset_summary.json"
    filing_type_counts = dict(Counter(record.filing_type for record in records))
    average_words = (
        round(sum(chunk["word_count"] for chunk in chunks) / len(chunks), 2)
        if chunks
        else 0.0
    )
    summary = {
        "ticker": normalized_ticker,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "filings_downloaded": len(records),
        "filing_type_counts": filing_type_counts,
        "filings": [
            {
                "filing_type": record.filing_type,
                "filing_date": record.filing_date,
                "accession_number": record.accession_number,
                "human_readable_path": record.local_markdown_path,
            }
            for record in records
        ],
        "cleaned_files": len(cleaned_records),
        "human_readable_files": sum(
            1 for record in records if record.local_markdown_path
        ),
        "total_chunks": len(chunks),
        "average_words_per_chunk": average_words,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "output_paths": {
            "raw_manifest": f"data/raw/{slug}/filing_metadata.json",
            "human_readable_dir": f"data/human_readable/{slug}",
            "cleaning_manifest": (
                f"data/processed/{slug}/cleaning_manifest.json"
            ),
            "chunks_csv": f"data/processed/{slug}/{slug}_filing_chunks.csv",
            "chunks_jsonl": f"data/processed/{slug}/{slug}_filing_chunks.jsonl",
            "filing_inventory": (
                f"outputs/data_summary/{slug}_filing_inventory.csv"
            ),
            "cleaning_summary": (
                f"outputs/data_summary/{slug}_cleaning_summary.csv"
            ),
            "dataset_summary": (
                f"outputs/data_summary/{slug}_dataset_summary.json"
            ),
        },
    }
    write_json(summary_path, summary)

    print("\nDataset build complete")
    print(f"Ticker: {normalized_ticker}")
    print(f"Filings downloaded: {len(records)}")
    for record in records:
        print(
            f"  {record.filing_type:<4} {record.filing_date} "
            f"{record.accession_number}"
        )
    print(f"Cleaned files: {len(cleaned_records)}")
    print(f"Human-readable files: {summary['human_readable_files']}")
    print(f"Total chunks: {len(chunks)}")
    print(f"Average words per chunk: {average_words}")
    print(f"Processed data: {project_relative(processed_directory, project_root)}")
    print(f"Summary: {project_relative(summary_path, project_root)}")
    return summary


def combine_chunk_outputs(
    tickers: Iterable[str],
    dataset_slug: str | None = None,
    project_root: Path = PROJECT_ROOT,
) -> list[dict[str, Any]]:
    """Concatenate per-ticker chunk outputs into a combined dataset."""

    normalized_tickers = normalize_tickers(tickers)
    slug = dataset_slug or dataset_slug_for_tickers(normalized_tickers)
    combined_directory = project_root / "data" / "processed" / slug
    combined_directory.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for ticker in normalized_tickers:
        ticker_slug_value = ticker_slug(ticker)
        jsonl_path = (
            project_root
            / "data"
            / "processed"
            / ticker_slug_value
            / f"{ticker_slug_value}_filing_chunks.jsonl"
        )
        rows.extend(read_jsonl(jsonl_path))

    duplicate_count = len(rows) - len({row["chunk_id"] for row in rows})
    if duplicate_count:
        raise RuntimeError(
            f"Combined chunk output has {duplicate_count} duplicate chunk ID(s)."
        )

    csv_path = combined_directory / f"{slug}_filing_chunks.csv"
    jsonl_path = combined_directory / f"{slug}_filing_chunks.jsonl"
    pd.DataFrame(rows, columns=CHUNK_COLUMNS).to_csv(csv_path, index=False)
    write_jsonl(jsonl_path, rows)
    print(f"Combined chunks: {project_relative(csv_path, project_root)}")
    print(f"Combined JSONL: {project_relative(jsonl_path, project_root)}")
    return rows


def build_dataset(
    ticker: str | None = None,
    tickers: Iterable[str] | None = None,
    num_8k: int = 5,
    chunk_size: int = 400,
    chunk_overlap: int = 75,
    project_root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    """Run the requested single- or multi-ticker data build."""

    selected_tickers = resolve_requested_tickers(ticker=ticker, tickers=tickers)
    validate_num_8k(num_8k)
    validate_chunk_settings(chunk_size, chunk_overlap)

    if len(selected_tickers) == 1:
        return build_single_dataset(
            ticker=selected_tickers[0],
            num_8k=num_8k,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            project_root=project_root,
        )

    print(
        "Starting multi-company build for: "
        + ", ".join(selected_tickers)
    )
    ticker_summaries = [
        build_single_dataset(
            ticker=single_ticker,
            num_8k=num_8k,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            project_root=project_root,
        )
        for single_ticker in selected_tickers
    ]
    dataset_slug = (
        DEFAULT_MULTI_COMPANY_SLUG
        if selected_tickers == DEFAULT_TICKERS
        else dataset_slug_for_tickers(selected_tickers)
    )
    combined_chunks = combine_chunk_outputs(
        selected_tickers,
        dataset_slug=dataset_slug,
        project_root=project_root,
    )
    aggregate_summary = describe_dataset(
        tickers=selected_tickers,
        dataset_slug=dataset_slug,
        chunk_size=chunk_size,
        project_root=project_root,
    )
    aggregate_summary["ticker_summaries"] = ticker_summaries
    aggregate_summary["combined_chunks"] = len(combined_chunks)
    write_json(
        project_root / "outputs" / "data_summary" / f"{dataset_slug}_dataset_summary.json",
        aggregate_summary,
    )
    print("\nMulti-company build complete")
    print(f"Dataset: {dataset_slug.upper()}")
    print(f"Tickers: {', '.join(selected_tickers)}")
    print(f"Filings downloaded: {aggregate_summary['filings_downloaded']}")
    print(f"Combined chunks: {len(combined_chunks)}")
    print(f"Data description: {aggregate_summary['output_paths']['data_description']}")
    return aggregate_summary


def build_parser() -> argparse.ArgumentParser:
    """Create the complete pipeline command-line parser."""

    parser = argparse.ArgumentParser(
        description=(
            "Build a cleaned, chunked SEC filing dataset. Defaults to FAANG."
        )
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--ticker", help="Single SEC company ticker.")
    group.add_argument(
        "--tickers",
        nargs="+",
        help="One or more SEC company tickers. Defaults to FAANG when omitted.",
    )
    parser.add_argument(
        "--num-8k",
        type=int,
        default=5,
        help="Number of recent unamended 8-K filings to download per ticker.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=400,
        help="Maximum number of words per chunk.",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=75,
        help="Number of overlapping words within each section.",
    )
    return parser


def main() -> None:
    """Run the complete pipeline CLI."""

    parser = build_parser()
    args = parser.parse_args()
    try:
        build_dataset(
            ticker=args.ticker,
            tickers=args.tickers,
            num_8k=args.num_8k,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
