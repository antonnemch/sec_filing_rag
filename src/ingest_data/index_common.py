"""Shared chunk snapshot and manifest handling for retrieval indexes."""

from __future__ import annotations

import os
import pickle
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import INDEX_SCHEMA_VERSION
from src.data.utils import (
    PROJECT_ROOT,
    read_json,
    sha256_file,
    stable_json_hash,
    ticker_slug,
    write_json,
)


CHUNKS_METADATA_FILE = "index_chunks.pkl"
LEGACY_INDEX_FILES = ("embeddings_chunks.pkl", "bm25_chunks.pkl")


class IndexValidationError(RuntimeError):
    """Raised when an index exists but no longer matches its source chunks."""


def select_ranked_indices(
    ranked_indices: list[int],
    chunks: list[dict[str, Any]],
    k: int,
    filing_types: tuple[str, ...] = (),
    require_each_filing_type: bool = False,
) -> list[int]:
    """Filter ranked chunks by form and optionally reserve one slot per form."""

    if k < 1:
        raise ValueError("k must be at least 1.")
    normalized_types = tuple(
        dict.fromkeys(value.strip().upper() for value in filing_types if value.strip())
    )
    allowed = set(normalized_types)
    eligible = [
        index
        for index in ranked_indices
        if not allowed
        or str(chunks[index].get("filing_type", "")).strip().upper() in allowed
    ]
    if not require_each_filing_type or len(normalized_types) < 2:
        return eligible[:k]

    selected: set[int] = set()
    for filing_type in normalized_types:
        match = next(
            (
                index
                for index in eligible
                if str(chunks[index].get("filing_type", "")).strip().upper()
                == filing_type
            ),
            None,
        )
        if match is not None and len(selected) < k:
            selected.add(match)
    for index in eligible:
        if len(selected) >= k:
            break
        selected.add(index)
    return [index for index in eligible if index in selected]


def index_dir(ticker: str, project_root: Path = PROJECT_ROOT) -> Path:
    return project_root / "data" / "processed" / ticker_slug(ticker)


def chunks_csv_path(ticker: str, project_root: Path = PROJECT_ROOT) -> Path:
    slug = ticker_slug(ticker)
    return index_dir(ticker, project_root) / f"{slug}_filing_chunks.csv"


def index_manifest_path(
    ticker: str, retriever: str, project_root: Path = PROJECT_ROOT
) -> Path:
    return index_dir(ticker, project_root) / f"{retriever}_index_manifest.json"


def load_chunk_rows(
    ticker: str, project_root: Path = PROJECT_ROOT
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load and validate source chunks, returning rows and stable fingerprint fields."""

    path = chunks_csv_path(ticker, project_root)
    if not path.exists():
        raise FileNotFoundError(f"Chunks CSV not found: {path}. Run build_dataset first.")
    frame = pd.read_csv(path)
    required = {"chunk_id", "text"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(
            f"Chunks CSV {path} is missing required column(s): {', '.join(missing)}"
        )
    if frame.empty:
        raise ValueError(f"Chunks CSV is empty: {path}")
    if frame["chunk_id"].isna().any() or frame["chunk_id"].duplicated().any():
        raise ValueError(f"Chunks CSV has missing or duplicate chunk IDs: {path}")
    if frame["text"].isna().any() or frame["text"].astype(str).str.strip().eq("").any():
        raise ValueError(f"Chunks CSV has blank chunk text: {path}")
    rows = frame.to_dict(orient="records")
    fingerprint = {
        "chunks_sha256": sha256_file(path),
        "chunks_row_count": len(rows),
        "chunk_ids_sha256": stable_json_hash(frame["chunk_id"].astype(str).tolist()),
    }
    return rows, fingerprint


def write_pickle_atomic(path: Path, value: Any) -> None:
    """Atomically write a pickle beside its final destination."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            pickle.dump(value, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def write_shared_chunks(
    ticker: str, chunks: list[dict[str, Any]], project_root: Path = PROJECT_ROOT
) -> Path:
    path = index_dir(ticker, project_root) / CHUNKS_METADATA_FILE
    write_pickle_atomic(path, chunks)
    return path


def prune_legacy_index_files(ticker: str, project_root: Path = PROJECT_ROOT) -> None:
    """Remove only known superseded per-retriever chunk metadata files."""

    directory = index_dir(ticker, project_root).resolve()
    for filename in LEGACY_INDEX_FILES:
        candidate = (directory / filename).resolve()
        candidate.relative_to(directory)
        if candidate.exists():
            candidate.unlink()


def build_index_manifest(
    ticker: str,
    retriever: str,
    source_fingerprint: dict[str, Any],
    project_root: Path = PROJECT_ROOT,
    **settings: Any,
) -> dict[str, Any]:
    manifest = {
        "schema_version": INDEX_SCHEMA_VERSION,
        "retriever": retriever,
        "ticker": ticker.strip().upper(),
        **source_fingerprint,
        **settings,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    manifest["index_fingerprint"] = stable_json_hash(
        {key: value for key, value in manifest.items() if key != "created_at_utc"}
    )
    write_json(index_manifest_path(ticker, retriever, project_root), manifest)
    return manifest


def load_validated_index_metadata(
    ticker: str,
    retriever: str,
    project_root: Path = PROJECT_ROOT,
    **expected_settings: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Validate an index manifest against current chunks and load shared metadata."""

    manifest_path = index_manifest_path(ticker, retriever, project_root)
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"{retriever.upper()} index manifest not found: {manifest_path}. Rebuild the index."
        )
    try:
        manifest = read_json(manifest_path)
    except (OSError, ValueError) as exc:
        raise IndexValidationError(f"Cannot read index manifest: {manifest_path}") from exc
    if not isinstance(manifest, dict):
        raise IndexValidationError(f"Invalid index manifest: {manifest_path}")
    fingerprint_payload = {
        key: value
        for key, value in manifest.items()
        if key not in {"created_at_utc", "index_fingerprint"}
    }
    if manifest.get("index_fingerprint") != stable_json_hash(fingerprint_payload):
        raise IndexValidationError(f"Index manifest fingerprint is invalid: {manifest_path}")
    _, current = load_chunk_rows(ticker, project_root)
    expected = {
        "schema_version": INDEX_SCHEMA_VERSION,
        "retriever": retriever,
        "ticker": ticker.strip().upper(),
        **current,
        **expected_settings,
    }
    mismatches = [
        key for key, value in expected.items() if manifest.get(key) != value
    ]
    if mismatches:
        raise IndexValidationError(
            f"Stale {retriever} index for {ticker}: manifest mismatch in "
            + ", ".join(sorted(mismatches))
        )

    chunks_path = index_dir(ticker, project_root) / CHUNKS_METADATA_FILE
    if not chunks_path.exists():
        raise IndexValidationError(
            f"Shared index chunk metadata is missing: {chunks_path}"
        )
    try:
        with chunks_path.open("rb") as handle:
            chunks = pickle.load(handle)
    except (OSError, pickle.PickleError, EOFError) as exc:
        raise IndexValidationError(
            f"Cannot read shared index chunk metadata: {chunks_path}"
        ) from exc
    if not isinstance(chunks, list) or len(chunks) != current["chunks_row_count"]:
        raise IndexValidationError(
            f"Shared index chunk metadata does not match current chunks for {ticker}."
        )
    if stable_json_hash([str(row.get("chunk_id", "")) for row in chunks]) != current[
        "chunk_ids_sha256"
    ]:
        raise IndexValidationError(
            f"Shared index chunk order does not match current chunks for {ticker}."
        )
    return chunks, manifest


def validated_index_fingerprint(
    ticker: str,
    retriever: str,
    project_root: Path = PROJECT_ROOT,
    **expected_settings: Any,
) -> str:
    _, manifest = load_validated_index_metadata(
        ticker, retriever, project_root, **expected_settings
    )
    return str(manifest["index_fingerprint"])
