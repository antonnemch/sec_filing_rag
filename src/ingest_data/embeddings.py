"""Build and query a fingerprinted OpenAI-backed FAISS index."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from openai import OpenAI

from src.config import (
    API_MAX_RETRIES,
    API_TIMEOUT_SECONDS,
    DEFAULT_EMBEDDING_BATCH_SIZE,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_RETRIEVAL_K,
)
from src.data.utils import PROJECT_ROOT, load_project_env
from src.ingest_data.index_common import (
    IndexValidationError,
    build_index_manifest,
    index_dir,
    index_manifest_path,
    load_chunk_rows,
    load_validated_index_metadata,
    prune_legacy_index_files,
    write_shared_chunks,
)


_INDEX_FILE = "embeddings.faiss"


def _client() -> OpenAI:
    load_project_env()
    return OpenAI(timeout=API_TIMEOUT_SECONDS, max_retries=API_MAX_RETRIES)


def _embed_texts(
    texts: list[str],
    model: str = DEFAULT_EMBEDDING_MODEL,
    batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE,
) -> list[list[float]]:
    client = _client()
    all_embeddings: list[list[float]] = []
    for offset in range(0, len(texts), batch_size):
        batch = texts[offset : offset + batch_size]
        response = client.embeddings.create(model=model, input=batch)
        all_embeddings.extend(item.embedding for item in response.data)
        print(f"  Embedded {min(offset + batch_size, len(texts))}/{len(texts)}")
    return all_embeddings


def build_embeddings_index(
    ticker: str,
    project_root: Path = PROJECT_ROOT,
    model: str = DEFAULT_EMBEDDING_MODEL,
    batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE,
) -> Path:
    """Embed all chunks and atomically save a validated FAISS index."""

    chunks, source_fingerprint = load_chunk_rows(ticker, project_root)
    texts = [str(chunk["text"]) for chunk in chunks]
    print(f"Building embeddings index for {ticker} ({len(texts)} chunks)...")
    vectors = np.asarray(
        _embed_texts(texts, model=model, batch_size=batch_size), dtype="float32"
    )
    if vectors.ndim != 2 or len(vectors) != len(chunks) or vectors.shape[1] < 1:
        raise RuntimeError("OpenAI returned an invalid embedding matrix.")
    faiss.normalize_L2(vectors)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    out_dir = index_dir(ticker, project_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{_INDEX_FILE}.", suffix=".tmp", dir=out_dir
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        faiss.write_index(index, str(temporary_path))
        manifest_path = index_manifest_path(ticker, "faiss", project_root)
        if manifest_path.exists():
            manifest_path.unlink()
        os.replace(temporary_path, out_dir / _INDEX_FILE)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    write_shared_chunks(ticker, chunks, project_root)
    build_index_manifest(
        ticker,
        "faiss",
        source_fingerprint,
        project_root,
        embedding_model=model,
        vector_dimension=int(vectors.shape[1]),
    )
    prune_legacy_index_files(ticker, project_root)
    print(f"  Saved {_INDEX_FILE}, shared chunks, and manifest to {out_dir}")
    return out_dir


def load_embeddings_index(
    ticker: str,
    project_root: Path = PROJECT_ROOT,
    model: str = DEFAULT_EMBEDDING_MODEL,
) -> tuple[Any, list[dict]]:
    out_dir = index_dir(ticker, project_root)
    index_path = out_dir / _INDEX_FILE
    if not index_path.exists():
        raise FileNotFoundError(f"Embeddings index not found: {index_path}. Rebuild the index.")
    try:
        index = faiss.read_index(str(index_path))
    except RuntimeError as exc:
        raise IndexValidationError(f"Cannot read FAISS index: {index_path}") from exc
    chunks, _ = load_validated_index_metadata(
        ticker,
        "faiss",
        project_root,
        embedding_model=model,
        vector_dimension=int(index.d),
    )
    if index.ntotal != len(chunks):
        raise IndexValidationError(f"FAISS vector count does not match chunks for {ticker}.")
    return index, chunks


def search_embeddings(
    query: str,
    index: Any,
    chunks: list[dict],
    model: str = DEFAULT_EMBEDDING_MODEL,
    k: int = DEFAULT_RETRIEVAL_K,
) -> list[dict]:
    if k < 1:
        raise ValueError("k must be at least 1.")
    response = _client().embeddings.create(model=model, input=[query])
    q_vec = np.asarray([response.data[0].embedding], dtype="float32")
    if q_vec.shape[1] != index.d:
        raise RuntimeError("Query embedding dimension does not match the FAISS index.")
    faiss.normalize_L2(q_vec)
    scores, indices = index.search(q_vec, min(k, len(chunks)))
    results: list[dict] = []
    for score, index_position in zip(scores[0], indices[0]):
        if index_position < 0:
            continue
        chunk = dict(chunks[index_position])
        chunk["retrieval_score"] = float(score)
        results.append(chunk)
    return results
