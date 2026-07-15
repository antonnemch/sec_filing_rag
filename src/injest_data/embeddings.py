"""Build and query an OpenAI-backed FAISS embeddings index for filing chunks."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import pandas as pd
from openai import OpenAI

from src.data.utils import PROJECT_ROOT, ticker_slug

EMBEDDINGS_MODEL = "text-embedding-3-small"
DEFAULT_BATCH_SIZE = 50
_INDEX_FILE = "embeddings.faiss"
_CHUNKS_FILE = "embeddings_chunks.pkl"


def _index_dir(ticker: str, project_root: Path) -> Path:
    return project_root / "data" / "processed" / ticker_slug(ticker)


def _embed_texts(
    texts: list[str],
    model: str = EMBEDDINGS_MODEL,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> list[list[float]]:
    client = OpenAI()
    all_embeddings: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        response = client.embeddings.create(model=model, input=batch)
        all_embeddings.extend(item.embedding for item in response.data)
        print(f"  Embedded {min(i + batch_size, len(texts))}/{len(texts)}")
    return all_embeddings


def build_embeddings_index(
    ticker: str,
    project_root: Path = PROJECT_ROOT,
    model: str = EMBEDDINGS_MODEL,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> Path:
    """Embed all chunks for a ticker and save a FAISS index. Returns the index dir."""
    slug = ticker_slug(ticker)
    chunks_csv = project_root / "data" / "processed" / slug / f"{slug}_filing_chunks.csv"
    if not chunks_csv.exists():
        raise FileNotFoundError(
            f"Chunks CSV not found: {chunks_csv}. Run build_dataset first."
        )

    df = pd.read_csv(chunks_csv)
    texts = df["text"].fillna("").astype(str).tolist()
    chunks = df.to_dict(orient="records")

    print(f"Building embeddings index for {ticker} ({len(texts)} chunks)...")
    raw = _embed_texts(texts, model=model, batch_size=batch_size)
    vectors = np.array(raw, dtype="float32")
    faiss.normalize_L2(vectors)

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    out_dir = _index_dir(ticker, project_root)
    faiss.write_index(index, str(out_dir / _INDEX_FILE))
    with open(out_dir / _CHUNKS_FILE, "wb") as fh:
        pickle.dump(chunks, fh)

    print(f"  Saved {_INDEX_FILE} and {_CHUNKS_FILE} to {out_dir}")
    return out_dir


def load_embeddings_index(
    ticker: str, project_root: Path = PROJECT_ROOT
) -> tuple[Any, list[dict]]:
    """Load an existing FAISS index and its chunk metadata."""
    out_dir = _index_dir(ticker, project_root)
    index_path = out_dir / _INDEX_FILE
    chunks_path = out_dir / _CHUNKS_FILE

    if not index_path.exists():
        raise FileNotFoundError(
            f"Embeddings index not found: {index_path}. "
            "Run build_embeddings_index first."
        )

    index = faiss.read_index(str(index_path))
    with open(chunks_path, "rb") as fh:
        chunks = pickle.load(fh)
    return index, chunks


def search_embeddings(
    query: str,
    index: Any,
    chunks: list[dict],
    model: str = EMBEDDINGS_MODEL,
    k: int = 5,
) -> list[dict]:
    """Embed a query and return the top-k most similar chunks with scores."""
    client = OpenAI()
    response = client.embeddings.create(model=model, input=[query])
    q_vec = np.array([response.data[0].embedding], dtype="float32")
    faiss.normalize_L2(q_vec)

    scores, indices = index.search(q_vec, k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        chunk = dict(chunks[idx])
        chunk["retrieval_score"] = float(score)
        results.append(chunk)
    return results
