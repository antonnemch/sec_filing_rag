"""Build and query a BM25 index for SEC filing chunks."""

from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd
from rank_bm25 import BM25Okapi

from src.data.utils import PROJECT_ROOT, ticker_slug

_INDEX_FILE = "bm25_index.pkl"
_CHUNKS_FILE = "bm25_chunks.pkl"


def _index_dir(ticker: str, project_root: Path) -> Path:
    return project_root / "data" / "processed" / ticker_slug(ticker)


def build_bm25_index(ticker: str, project_root: Path = PROJECT_ROOT) -> Path:
    """Build a BM25 index over chunk texts for a ticker. Returns the index dir."""
    slug = ticker_slug(ticker)
    chunks_csv = project_root / "data" / "processed" / slug / f"{slug}_filing_chunks.csv"
    if not chunks_csv.exists():
        raise FileNotFoundError(
            f"Chunks CSV not found: {chunks_csv}. Run build_dataset first."
        )

    df = pd.read_csv(chunks_csv)
    texts = df["text"].fillna("").astype(str).tolist()
    chunks = df.to_dict(orient="records")

    print(f"Building BM25 index for {ticker} ({len(texts)} chunks)...")
    tokenized = [text.lower().split() for text in texts]
    index = BM25Okapi(tokenized)

    out_dir = _index_dir(ticker, project_root)
    with open(out_dir / _INDEX_FILE, "wb") as fh:
        pickle.dump(index, fh)
    with open(out_dir / _CHUNKS_FILE, "wb") as fh:
        pickle.dump(chunks, fh)

    print(f"  Saved {_INDEX_FILE} and {_CHUNKS_FILE} to {out_dir}")
    return out_dir


def load_bm25_index(
    ticker: str, project_root: Path = PROJECT_ROOT
) -> tuple[BM25Okapi, list[dict]]:
    """Load an existing BM25 index and its chunk metadata."""
    out_dir = _index_dir(ticker, project_root)
    index_path = out_dir / _INDEX_FILE
    chunks_path = out_dir / _CHUNKS_FILE

    if not index_path.exists():
        raise FileNotFoundError(
            f"BM25 index not found: {index_path}. Run build_bm25_index first."
        )

    with open(index_path, "rb") as fh:
        index = pickle.load(fh)
    with open(chunks_path, "rb") as fh:
        chunks = pickle.load(fh)
    return index, chunks


def search_bm25(
    query: str,
    index: BM25Okapi,
    chunks: list[dict],
    k: int = 5,
) -> list[dict]:
    """Return the top-k most relevant chunks for a query."""
    scores = index.get_scores(query.lower().split())
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [
        {**chunks[idx], "retrieval_score": float(scores[idx])}
        for idx in top_indices
    ]
