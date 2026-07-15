"""Retrieve relevant filing chunks for a question using FAISS or BM25."""

from __future__ import annotations

from pathlib import Path

from src.data.utils import PROJECT_ROOT
from src.injest_data.bm25 import build_bm25_index, load_bm25_index, search_bm25
from src.injest_data.embeddings import (
    build_embeddings_index,
    load_embeddings_index,
    search_embeddings,
)


def retrieve_chunks(
    question: str,
    ticker: str,
    retriever: str = "faiss",
    k: int = 5,
    project_root: Path = PROJECT_ROOT,
    build_if_missing: bool = True,
) -> list[dict]:
    """Return the top-k chunks most relevant to a question for the given ticker.

    Args:
        retriever: "faiss" (dense, OpenAI embeddings) or "bm25" (sparse, keyword).
        build_if_missing: automatically build the index if not found on disk.
    """
    if retriever == "faiss":
        try:
            index, chunks = load_embeddings_index(ticker, project_root)
        except FileNotFoundError:
            if not build_if_missing:
                raise
            build_embeddings_index(ticker, project_root)
            index, chunks = load_embeddings_index(ticker, project_root)
        return search_embeddings(question, index, chunks, k=k)

    if retriever == "bm25":
        try:
            index, chunks = load_bm25_index(ticker, project_root)
        except FileNotFoundError:
            if not build_if_missing:
                raise
            build_bm25_index(ticker, project_root)
            index, chunks = load_bm25_index(ticker, project_root)
        return search_bm25(question, index, chunks, k=k)

    raise ValueError(f"Unknown retriever {retriever!r}. Choose 'faiss' or 'bm25'.")
