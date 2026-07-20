"""Run SEC filing download, cleaning, chunking, and reporting pipelines."""

from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from src.config import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_MULTI_COMPANY_SLUG,
    DEFAULT_NUM_8K,
    DEFAULT_TICKERS,
    MAX_FILING_DATE,
)

from .chunk_filings import CHUNK_COLUMNS, chunk_filings
from .clean_filings import clean_filings
from .describe_dataset import describe_dataset
from .download_filings import download_filings
from .utils import (
    PROJECT_ROOT,
    FilingRecord,
    dataset_slug_for_tickers,
    normalize_ticker,
    normalize_tickers,
    project_relative,
    read_jsonl,
    ticker_slug,
    validate_chunk_settings,
    validate_num_8k,
    load_project_env,
    read_json,
    sha256_file,
    resolve_project_path,
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


def _write_ticker_summary(
    ticker: str,
    records: list[FilingRecord],
    cleaned_records: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
    chunk_size: int,
    chunk_overlap: int,
    project_root: Path,
) -> dict[str, Any]:
    """Write the canonical compact summary for current ticker artifacts."""

    slug = ticker_slug(ticker)
    chunks_path = (
        project_root / "data" / "processed" / slug / f"{slug}_filing_chunks.csv"
    )
    summary_path = (
        project_root / "outputs" / "data_summary" / f"{slug}_dataset_summary.json"
    )
    average_words = (
        round(sum(int(chunk["word_count"]) for chunk in chunks) / len(chunks), 2)
        if chunks
        else 0.0
    )
    summary = {
        "ticker": ticker,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "filings_as_of": MAX_FILING_DATE.isoformat(),
        "filings_downloaded": len(records),
        "filing_type_counts": dict(Counter(record.filing_type for record in records)),
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
        "human_readable_files": sum(1 for record in records if record.local_markdown_path),
        "total_chunks": len(chunks),
        "average_words_per_chunk": average_words,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "chunks_sha256": sha256_file(chunks_path),
        "output_paths": {
            "raw_manifest": f"data/raw/{slug}/filing_metadata.json",
            "human_readable_dir": f"data/human_readable/{slug}",
            "cleaning_manifest": f"data/processed/{slug}/cleaning_manifest.json",
            "chunks_csv": f"data/processed/{slug}/{slug}_filing_chunks.csv",
            "chunks_jsonl": f"data/processed/{slug}/{slug}_filing_chunks.jsonl",
            "filing_inventory": f"outputs/data_summary/{slug}_filing_inventory.csv",
            "cleaning_summary": f"outputs/data_summary/{slug}_cleaning_summary.csv",
            "dataset_summary": f"outputs/data_summary/{slug}_dataset_summary.json",
        },
    }
    write_json(summary_path, summary)
    return summary


def _build_single_dataset_in_place(
    ticker: str = "AMZN",
    num_8k: int = DEFAULT_NUM_8K,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
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
    summary_path = project_root / "outputs" / "data_summary" / f"{slug}_dataset_summary.json"
    summary = _write_ticker_summary(
        normalized_ticker,
        records,
        cleaned_records,
        chunks,
        chunk_size,
        chunk_overlap,
        project_root,
    )

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
    print(f"Average words per chunk: {summary['average_words_per_chunk']}")
    print(f"Processed data: {project_relative(processed_directory, project_root)}")
    print(f"Summary: {project_relative(summary_path, project_root)}")
    return summary


def refresh_dataset_summary(
    ticker: str,
    project_root: Path = PROJECT_ROOT,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> dict[str, Any]:
    """Regenerate one ticker summary entirely from its current manifests and chunks."""

    normalized_ticker = normalize_ticker(ticker)
    slug = ticker_slug(normalized_ticker)
    raw_manifest_path = project_root / "data" / "raw" / slug / "filing_metadata.json"
    cleaning_manifest_path = (
        project_root / "data" / "processed" / slug / "cleaning_manifest.json"
    )
    chunks_path = (
        project_root / "data" / "processed" / slug / f"{slug}_filing_chunks.csv"
    )
    records = [FilingRecord.from_dict(item) for item in read_json(raw_manifest_path)]
    cleaned_records = read_json(cleaning_manifest_path)
    chunks = pd.read_csv(chunks_path)
    if not records or not isinstance(cleaned_records, list) or chunks.empty:
        raise ValueError(f"Cannot summarize incomplete dataset artifacts for {normalized_ticker}.")

    existing_path = (
        project_root / "outputs" / "data_summary" / f"{slug}_dataset_summary.json"
    )
    existing: dict[str, Any] = {}
    if existing_path.exists():
        try:
            value = read_json(existing_path)
            existing = value if isinstance(value, dict) else {}
        except (OSError, ValueError):
            existing = {}
    chunk_size = int(
        chunk_size
        if chunk_size is not None
        else existing.get("chunk_size", DEFAULT_CHUNK_SIZE)
    )
    chunk_overlap = int(
        chunk_overlap
        if chunk_overlap is not None
        else existing.get("chunk_overlap", DEFAULT_CHUNK_OVERLAP)
    )
    summary_directory = project_root / "outputs" / "data_summary"
    summary_directory.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([record.to_dict() for record in records]).to_csv(
        summary_directory / f"{slug}_filing_inventory.csv", index=False
    )
    cleaning_rows = [
        {
            "ticker": item["ticker"],
            "filing_type": item["filing_type"],
            "filing_date": item["filing_date"],
            "accession_number": item["accession_number"],
            "raw_length": item["raw_length"],
            "cleaned_length": item["cleaned_length"],
            "retained_ratio": round(item["cleaned_length"] / item["raw_length"], 4)
            if item["raw_length"]
            else 0.0,
            "warnings": "; ".join(item.get("warnings", [])),
        }
        for item in cleaned_records
    ]
    pd.DataFrame(cleaning_rows).to_csv(
        summary_directory / f"{slug}_cleaning_summary.csv", index=False
    )
    return _write_ticker_summary(
        normalized_ticker,
        records,
        cleaned_records,
        chunks.to_dict(orient="records"),
        chunk_size,
        chunk_overlap,
        project_root,
    )


def _prune_children(directory: Path, allowed: set[Path]) -> int:
    """Remove direct children that are neither allowed nor parents of allowed paths."""

    if not directory.exists():
        return 0
    directory_root = directory.resolve()
    normalized_allowed = {path.resolve() for path in allowed}
    removed = 0
    for child in directory.iterdir():
        resolved = child.resolve()
        resolved.relative_to(directory_root)
        keep = resolved in normalized_allowed or any(
            resolved in candidate.parents for candidate in normalized_allowed
        )
        if not keep:
            _remove_artifact(child)
            removed += 1
    return removed


def prune_stale_dataset_artifacts(
    ticker: str, project_root: Path = PROJECT_ROOT
) -> int:
    """Prune files not referenced by current ticker manifests from scoped directories."""

    normalized_ticker = normalize_ticker(ticker)
    slug = ticker_slug(normalized_ticker)
    raw_directory = project_root / "data" / "raw" / slug
    human_directory = project_root / "data" / "human_readable" / slug
    processed_directory = project_root / "data" / "processed" / slug
    raw_manifest_path = raw_directory / "filing_metadata.json"
    cleaning_manifest_path = processed_directory / "cleaning_manifest.json"
    raw_manifest = read_json(raw_manifest_path)
    cleaning_manifest = read_json(cleaning_manifest_path)
    if not isinstance(raw_manifest, list) or not isinstance(cleaning_manifest, list):
        raise ValueError(f"Cannot prune invalid manifests for {normalized_ticker}.")

    raw_allowed = {raw_manifest_path.resolve()}
    human_allowed: set[Path] = set()
    for item in raw_manifest:
        raw_path = resolve_project_path(str(item["local_raw_path"]), project_root)
        try:
            raw_path.resolve().relative_to(raw_directory.resolve())
        except ValueError as exc:
            raise ValueError(
                f"Raw manifest path is outside {normalized_ticker}'s raw directory: {raw_path}"
            ) from exc
        raw_allowed.add(raw_path)
        human_path = str(item.get("local_markdown_path", "")).strip()
        if human_path:
            resolved_human_path = resolve_project_path(human_path, project_root)
            try:
                resolved_human_path.resolve().relative_to(human_directory.resolve())
            except ValueError as exc:
                raise ValueError(
                    f"Human-readable manifest path is outside {normalized_ticker}'s directory: "
                    f"{resolved_human_path}"
                ) from exc
            human_allowed.add(resolved_human_path)

    cleaned_directory = processed_directory / "cleaned_filings"
    cleaned_allowed: set[Path] = set()
    for item in cleaning_manifest:
        cleaned_path = resolve_project_path(str(item["local_cleaned_path"]), project_root)
        try:
            cleaned_path.resolve().relative_to(cleaned_directory.resolve())
        except ValueError as exc:
            raise ValueError(
                f"Cleaning manifest path is outside {normalized_ticker}'s cleaned directory: "
                f"{cleaned_path}"
            ) from exc
        cleaned_allowed.add(cleaned_path)
    recognized_index_files = {
        "index_chunks.pkl",
        "embeddings.faiss",
        "faiss_index_manifest.json",
        "bm25_index.pkl",
        "bm25_index_manifest.json",
    }
    processed_allowed = {
        cleaning_manifest_path.resolve(),
        (processed_directory / f"{slug}_filing_chunks.csv").resolve(),
        (processed_directory / f"{slug}_filing_chunks.jsonl").resolve(),
        cleaned_directory.resolve(),
        *{
            (processed_directory / filename).resolve()
            for filename in recognized_index_files
        },
    }
    removed = _prune_children(raw_directory, raw_allowed)
    removed += _prune_children(human_directory, human_allowed)
    removed += _prune_children(cleaned_directory, cleaned_allowed)
    removed += _prune_children(processed_directory, processed_allowed)

    allowed_summaries = {
        f"{slug}_filing_inventory.csv",
        f"{slug}_cleaning_summary.csv",
        f"{slug}_dataset_summary.json",
    }
    summary_directory = project_root / "outputs" / "data_summary"
    for candidate in summary_directory.glob(f"{slug}_*"):
        candidate.resolve().relative_to(summary_directory.resolve())
        if candidate.name not in allowed_summaries:
            _remove_artifact(candidate)
            removed += 1
    return removed


def _validate_staged_dataset(
    ticker: str,
    num_8k: int,
    chunk_size: int,
    project_root: Path,
) -> None:
    """Validate a complete staged ticker build before replacing live artifacts."""

    slug = ticker_slug(ticker)
    raw_manifest_path = project_root / "data" / "raw" / slug / "filing_metadata.json"
    chunks_path = (
        project_root / "data" / "processed" / slug / f"{slug}_filing_chunks.csv"
    )
    chunks_jsonl_path = (
        project_root / "data" / "processed" / slug / f"{slug}_filing_chunks.jsonl"
    )
    summary_path = project_root / "outputs" / "data_summary" / f"{slug}_dataset_summary.json"
    raw_manifest = read_json(raw_manifest_path)
    summary = read_json(summary_path)
    if not isinstance(raw_manifest, list) or len(raw_manifest) != num_8k + 2:
        raise RuntimeError(
            f"Staged {ticker} manifest does not contain the expected {num_8k + 2} filings."
        )

    chunks = pd.read_csv(chunks_path)
    required = {
        "chunk_id",
        "ticker",
        "filing_type",
        "filing_date",
        "accession_number",
        "text",
        "word_count",
    }
    missing = sorted(required.difference(chunks.columns))
    if missing:
        raise RuntimeError(
            "Staged chunk database is missing required column(s): " + ", ".join(missing)
        )
    if chunks.empty or chunks["chunk_id"].isna().any() or chunks["chunk_id"].duplicated().any():
        raise RuntimeError("Staged chunk database is empty or contains invalid chunk IDs.")
    if chunks["text"].isna().any() or chunks["text"].astype(str).str.strip().eq("").any():
        raise RuntimeError("Staged chunk database contains blank chunk text.")
    if set(chunks["ticker"].astype(str).str.strip().str.upper()) != {ticker}:
        raise RuntimeError(f"Staged chunk database contains rows outside ticker {ticker}.")
    jsonl_rows = read_jsonl(chunks_jsonl_path)
    if [str(row.get("chunk_id", "")) for row in jsonl_rows] != chunks[
        "chunk_id"
    ].astype(str).tolist():
        raise RuntimeError("Staged CSV and JSONL chunk rows do not match in order.")
    word_counts = pd.to_numeric(chunks["word_count"], errors="coerce")
    if word_counts.isna().any() or (word_counts < 1).any() or (word_counts > chunk_size).any():
        raise RuntimeError("Staged chunk database violates the configured chunk-size limit.")

    filings = chunks.drop_duplicates("accession_number")
    counts = filings["filing_type"].value_counts().to_dict()
    dates = pd.to_datetime(filings["filing_date"], errors="raise").dt.date
    expected = {"10-K": 1, "10-Q": 1, "8-K": num_8k}
    if any(counts.get(form, 0) != count for form, count in expected.items()):
        raise RuntimeError(f"Staged {ticker} dataset violates the filing-count policy.")
    if not bool((dates <= MAX_FILING_DATE).all()):
        raise RuntimeError(f"Staged {ticker} dataset contains a filing after the cutoff.")
    if not isinstance(summary, dict) or summary.get("total_chunks") != len(chunks):
        raise RuntimeError("Staged dataset summary does not match the chunk database.")


def _remove_artifact(path: Path) -> None:
    """Remove one already-validated file or directory during commit rollback."""

    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def _commit_staged_dataset(ticker: str, staged_root: Path, project_root: Path) -> None:
    """Replace only one ticker's generated artifacts, rolling back on failure."""

    slug = ticker_slug(ticker)
    relative_targets = [
        Path("data") / "raw" / slug,
        Path("data") / "human_readable" / slug,
        Path("data") / "processed" / slug,
        Path("outputs") / "data_summary" / f"{slug}_filing_inventory.csv",
        Path("outputs") / "data_summary" / f"{slug}_cleaning_summary.csv",
        Path("outputs") / "data_summary" / f"{slug}_dataset_summary.json",
    ]
    root = project_root.resolve()
    backup_root = staged_root / ".commit_backup"
    backups: dict[Path, Path] = {}
    committed: list[Path] = []

    try:
        for relative in relative_targets:
            source = (staged_root / relative).resolve()
            target = (project_root / relative).resolve()
            source.relative_to(staged_root.resolve())
            target.relative_to(root)
            if not source.exists():
                raise RuntimeError(f"Staged artifact is missing: {relative.as_posix()}")
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                backup = backup_root / relative
                backup.parent.mkdir(parents=True, exist_ok=True)
                os.replace(target, backup)
                backups[target] = backup
            os.replace(source, target)
            committed.append(target)
    except Exception:
        for target in reversed(committed):
            _remove_artifact(target)
        for target, backup in reversed(list(backups.items())):
            if backup.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                os.replace(backup, target)
        raise

    allowed_summary_files = {
        f"{slug}_filing_inventory.csv",
        f"{slug}_cleaning_summary.csv",
        f"{slug}_dataset_summary.json",
    }
    summary_directory = project_root / "outputs" / "data_summary"
    for stale in summary_directory.glob(f"{slug}_*"):
        stale.resolve().relative_to(summary_directory.resolve())
        if stale.name not in allowed_summary_files:
            try:
                _remove_artifact(stale)
            except OSError as exc:
                print(f"Warning: could not prune stale artifact {stale}: {exc}")


def build_single_dataset(
    ticker: str = "AMZN",
    num_8k: int = DEFAULT_NUM_8K,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    project_root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    """Transactionally build and replace all generated artifacts for one ticker."""

    normalized_ticker = normalize_ticker(ticker)
    validate_num_8k(num_8k)
    validate_chunk_settings(chunk_size, chunk_overlap)
    project_root.mkdir(parents=True, exist_ok=True)
    load_project_env(project_root)
    os.environ.setdefault(
        "EDGAR_LOCAL_DATA_DIR",
        str((project_root / "data" / "raw" / ".edgar_cache").resolve()),
    )
    staging_parent = project_root / ".cache" / "dataset_builds"
    staging_parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix=f"{ticker_slug(ticker)}-", dir=staging_parent) as name:
        staged_root = Path(name)
        summary = _build_single_dataset_in_place(
            ticker=normalized_ticker,
            num_8k=num_8k,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            project_root=staged_root,
        )
        _validate_staged_dataset(
            normalized_ticker, num_8k, chunk_size, staged_root
        )
        _commit_staged_dataset(normalized_ticker, staged_root, project_root)
    print(f"[{normalized_ticker}] Transactional dataset replacement complete.")
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
    num_8k: int = DEFAULT_NUM_8K,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
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
        default=DEFAULT_NUM_8K,
        help="Number of recent unamended 8-K filings to download per ticker.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help="Maximum number of words per chunk.",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=DEFAULT_CHUNK_OVERLAP,
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
