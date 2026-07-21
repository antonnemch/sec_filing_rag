"""Split cleaned SEC filings into metadata-rich word chunks."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import CHUNK_SCHEMA_VERSION

from .clean_filings import HEADING_PREFIX
from .utils import (
    PROJECT_ROOT,
    normalize_ticker,
    project_relative,
    read_json,
    resolve_project_path,
    ticker_slug,
    validate_chunk_settings,
    write_jsonl,
)


CHUNK_COLUMNS = [
    "chunk_schema_version",
    "company",
    "ticker",
    "cik",
    "filing_type",
    "filing_date",
    "accession_number",
    "source_url",
    "source_file",
    "chunk_id",
    "chunk_index",
    "section_heading",
    "text",
    "word_count",
]


def split_sections(cleaned_text: str) -> list[tuple[str, str]]:
    """Split cleaned text into heading-labelled sections."""

    sections: list[tuple[str, str]] = []
    current_heading = "unknown"
    body_lines: list[str] = []

    def flush() -> None:
        body = "\n\n".join(line for line in body_lines if line).strip()
        # Page numbers and table-navigation fragments must not become apparent
        # substantive sections merely because they followed an item heading.
        if body and re.search(r"[A-Za-z]{2,}", body):
            sections.append((current_heading, body))

    for raw_line in cleaned_text.splitlines():
        line = raw_line.strip()
        if line.startswith(HEADING_PREFIX):
            flush()
            body_lines = []
            current_heading = line[len(HEADING_PREFIX) :].strip() or "unknown"
        elif line:
            body_lines.append(line)

    flush()
    return sections


def chunk_cleaned_text(
    cleaned_text: str,
    chunk_size: int = 400,
    chunk_overlap: int = 75,
) -> list[tuple[str, str, int]]:
    """Create overlapping word windows without crossing section boundaries."""

    validate_chunk_settings(chunk_size, chunk_overlap)
    chunks: list[tuple[str, str, int]] = []
    step = chunk_size - chunk_overlap

    for section_heading, section_text in split_sections(cleaned_text):
        words = section_text.split()
        start = 0
        while start < len(words):
            chunk_words = words[start : start + chunk_size]
            chunks.append(
                (section_heading, " ".join(chunk_words), len(chunk_words))
            )
            if start + chunk_size >= len(words):
                break
            start += step

    return chunks


def _deterministic_chunk_id(
    ticker: str, accession_number: str, chunk_index: int
) -> str:
    """Build a stable chunk identifier from filing metadata."""

    accession = re.sub(r"[^A-Za-z0-9]+", "", accession_number)
    return f"{ticker_slug(ticker)}_{accession}_{chunk_index:04d}"


def chunk_filings(
    ticker: str = "AMZN",
    chunk_size: int = 400,
    chunk_overlap: int = 75,
    project_root: Path = PROJECT_ROOT,
) -> list[dict[str, Any]]:
    """Chunk only cleaned files referenced by the current cleaning manifest."""

    normalized_ticker = normalize_ticker(ticker)
    validate_chunk_settings(chunk_size, chunk_overlap)
    slug = ticker_slug(normalized_ticker)
    processed_directory = project_root / "data" / "processed" / slug
    manifest_path = processed_directory / "cleaning_manifest.json"
    manifest = read_json(manifest_path)
    if not isinstance(manifest, list) or not manifest:
        raise ValueError(f"Cleaning manifest is empty or invalid: {manifest_path}")

    rows: list[dict[str, Any]] = []
    required_metadata = {
        "company",
        "ticker",
        "cik",
        "filing_type",
        "filing_date",
        "accession_number",
        "source_url",
        "local_cleaned_path",
    }
    for item in manifest:
        missing = sorted(required_metadata.difference(item))
        if missing:
            raise ValueError(
                "Cleaning manifest is missing required field(s): " + ", ".join(missing)
            )
        if item["ticker"] != normalized_ticker:
            raise ValueError(
                f"Manifest ticker {item['ticker']} does not match {normalized_ticker}."
            )

        cleaned_path = resolve_project_path(item["local_cleaned_path"], project_root)
        if not cleaned_path.exists():
            raise FileNotFoundError(f"Cleaned filing does not exist: {cleaned_path}")
        cleaned_text = cleaned_path.read_text(encoding="utf-8")
        filing_chunks = chunk_cleaned_text(
            cleaned_text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        if not filing_chunks:
            raise RuntimeError(
                f"No chunks were created for filing {item['accession_number']}."
            )

        for chunk_index, (section_heading, text, word_count) in enumerate(
            filing_chunks
        ):
            rows.append(
                {
                    "chunk_schema_version": CHUNK_SCHEMA_VERSION,
                    "company": item["company"],
                    "ticker": item["ticker"],
                    "cik": str(item["cik"]),
                    "filing_type": item["filing_type"],
                    "filing_date": item["filing_date"],
                    "accession_number": item["accession_number"],
                    "source_url": item["source_url"],
                    "source_file": item["local_cleaned_path"],
                    "chunk_id": _deterministic_chunk_id(
                        normalized_ticker,
                        item["accession_number"],
                        chunk_index,
                    ),
                    "chunk_index": chunk_index,
                    "section_heading": section_heading,
                    "text": text,
                    "word_count": word_count,
                }
            )

    csv_path = processed_directory / f"{slug}_filing_chunks.csv"
    jsonl_path = processed_directory / f"{slug}_filing_chunks.jsonl"
    pd.DataFrame(rows, columns=CHUNK_COLUMNS).to_csv(csv_path, index=False)
    write_jsonl(jsonl_path, rows)

    print(
        f"Created {len(rows)} chunk(s) in "
        f"{project_relative(processed_directory, project_root)}"
    )
    print(f"CSV: {project_relative(csv_path, project_root)}")
    print(f"JSONL: {project_relative(jsonl_path, project_root)}")
    return rows


def build_parser() -> argparse.ArgumentParser:
    """Create the chunking command-line parser."""

    parser = argparse.ArgumentParser(
        description="Create retrieval-ready word chunks from cleaned SEC filings."
    )
    parser.add_argument("--ticker", default="AMZN", help="SEC company ticker.")
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
    """Run the chunking CLI."""

    parser = build_parser()
    args = parser.parse_args()
    try:
        chunk_filings(
            ticker=args.ticker,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
