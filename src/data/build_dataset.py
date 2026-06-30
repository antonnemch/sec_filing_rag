"""Run the complete SEC filing download, cleaning, and chunking pipeline."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .chunk_filings import chunk_filings
from .clean_filings import clean_filings
from .download_filings import download_filings
from .utils import (
    PROJECT_ROOT,
    normalize_ticker,
    project_relative,
    ticker_slug,
    validate_chunk_settings,
    validate_num_8k,
    write_json,
)


def build_dataset(
    ticker: str = "AMZN",
    num_8k: int = 5,
    chunk_size: int = 400,
    chunk_overlap: int = 75,
    project_root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    """Run every data stage and write a compact dataset summary."""

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
            }
            for record in records
        ],
        "cleaned_files": len(cleaned_records),
        "total_chunks": len(chunks),
        "average_words_per_chunk": average_words,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "output_paths": {
            "raw_manifest": f"data/raw/{slug}/filing_metadata.json",
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
    print(f"Total chunks: {len(chunks)}")
    print(f"Average words per chunk: {average_words}")
    print(f"Processed data: {project_relative(processed_directory, project_root)}")
    print(f"Summary: {project_relative(summary_path, project_root)}")
    return summary


def build_parser() -> argparse.ArgumentParser:
    """Create the complete pipeline command-line parser."""

    parser = argparse.ArgumentParser(
        description="Build a cleaned, chunked SEC filing dataset."
    )
    parser.add_argument("--ticker", default="AMZN", help="SEC company ticker.")
    parser.add_argument(
        "--num-8k",
        type=int,
        default=5,
        help="Number of recent unamended 8-K filings to download.",
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
            num_8k=args.num_8k,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
