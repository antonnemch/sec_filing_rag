"""Build and query a fingerprinted BM25 index for SEC filing chunks."""

from __future__ import annotations

from pathlib import Path

from rank_bm25 import BM25Okapi

from src.config import DEFAULT_RETRIEVAL_K
from src.data.utils import PROJECT_ROOT
from src.ingest_data.index_common import (
    IndexValidationError,
    build_index_manifest,
    index_dir,
    index_manifest_path,
    load_chunk_rows,
    load_validated_index_metadata,
    prune_legacy_index_files,
    select_ranked_indices,
    write_pickle_atomic,
    write_shared_chunks,
)


_INDEX_FILE = "bm25_index.pkl"
_TOKENIZER_VERSION = "lower-whitespace-v1"


def build_bm25_index(ticker: str, project_root: Path = PROJECT_ROOT) -> Path:
    chunks, source_fingerprint = load_chunk_rows(ticker, project_root)
    texts = [str(chunk["text"]) for chunk in chunks]
    print(f"Building BM25 index for {ticker} ({len(texts)} chunks)...")
    index = BM25Okapi([text.lower().split() for text in texts])
    out_dir = index_dir(ticker, project_root)
    manifest_path = index_manifest_path(ticker, "bm25", project_root)
    if manifest_path.exists():
        manifest_path.unlink()
    write_pickle_atomic(out_dir / _INDEX_FILE, index)
    write_shared_chunks(ticker, chunks, project_root)
    build_index_manifest(
        ticker,
        "bm25",
        source_fingerprint,
        project_root,
        tokenizer_version=_TOKENIZER_VERSION,
    )
    prune_legacy_index_files(ticker, project_root)
    print(f"  Saved {_INDEX_FILE}, shared chunks, and manifest to {out_dir}")
    return out_dir


def load_bm25_index(
    ticker: str, project_root: Path = PROJECT_ROOT
) -> tuple[BM25Okapi, list[dict]]:
    import pickle

    out_dir = index_dir(ticker, project_root)
    index_path = out_dir / _INDEX_FILE
    if not index_path.exists():
        raise FileNotFoundError(f"BM25 index not found: {index_path}. Rebuild the index.")
    chunks, _ = load_validated_index_metadata(
        ticker,
        "bm25",
        project_root,
        tokenizer_version=_TOKENIZER_VERSION,
    )
    try:
        with index_path.open("rb") as handle:
            index = pickle.load(handle)
    except (OSError, pickle.PickleError, EOFError) as exc:
        raise IndexValidationError(f"Cannot read BM25 index: {index_path}") from exc
    if len(index.doc_freqs) != len(chunks):
        raise IndexValidationError(f"BM25 document count does not match chunks for {ticker}.")
    return index, chunks


def search_bm25(
    query: str,
    index: BM25Okapi,
    chunks: list[dict],
    k: int = DEFAULT_RETRIEVAL_K,
    filing_types: tuple[str, ...] = (),
    require_each_filing_type: bool = False,
) -> list[dict]:
    if k < 1:
        raise ValueError("k must be at least 1.")
    scores = index.get_scores(query.lower().split())
    ranked_indices = sorted(
        range(len(scores)), key=lambda item: scores[item], reverse=True
    )
    top_indices = select_ranked_indices(
        ranked_indices,
        chunks,
        min(k, len(chunks)),
        filing_types,
        require_each_filing_type,
    )
    return [
        {**chunks[index_position], "retrieval_score": float(scores[index_position])}
        for index_position in top_indices
    ]
