"""Retrieve relevant filing chunks for a question using FAISS or BM25."""

from __future__ import annotations

import re
from pathlib import Path

from src.config import DEFAULT_RETRIEVAL_K
from src.data.utils import PROJECT_ROOT
from src.ingest_data.bm25 import build_bm25_index, load_bm25_index, search_bm25
from src.ingest_data.embeddings import (
    build_embeddings_index,
    load_embeddings_index,
    search_embeddings,
)
from src.ingest_data.index_common import IndexValidationError


_FILING_TYPE_PATTERN = re.compile(
    r"\b(?:10\s*[-\u2013\u2014]?\s*[kq]|8\s*[-\u2013\u2014]?\s*k)\b",
    re.IGNORECASE,
)
_COMPARISON_PATTERN = re.compile(
    r"\b(?:compar(?:e|ed|ing|ison)|versus|vs\.?|relative\s+to)\b",
    re.IGNORECASE,
)


def infer_filing_constraints(question: str) -> tuple[tuple[str, ...], bool]:
    """Infer explicit SEC form filters and whether every named form is required."""

    normalized: list[str] = []
    for match in _FILING_TYPE_PATTERN.finditer(question):
        compact = re.sub(r"[\s\u2013\u2014-]+", "", match.group(0)).upper()
        filing_type = {"10K": "10-K", "10Q": "10-Q", "8K": "8-K"}[compact]
        if filing_type not in normalized:
            normalized.append(filing_type)
    require_each = len(normalized) > 1 and bool(_COMPARISON_PATTERN.search(question))
    return tuple(normalized), require_each


def retrieve_chunks(
    question: str,
    ticker: str,
    retriever: str = "faiss",
    k: int = DEFAULT_RETRIEVAL_K,
    project_root: Path = PROJECT_ROOT,
    build_if_missing: bool = True,
) -> list[dict]:
    """Return the top-k chunks most relevant to a question for the given ticker.

    Args:
        retriever: "faiss" (dense, OpenAI embeddings) or "bm25" (sparse, keyword).
        build_if_missing: automatically build the index if not found on disk.
    """
    filing_types, require_each_filing_type = infer_filing_constraints(question)
    if retriever == "faiss":
        try:
            index, chunks = load_embeddings_index(ticker, project_root)
        except (FileNotFoundError, IndexValidationError):
            if not build_if_missing:
                raise
            build_embeddings_index(ticker, project_root)
            index, chunks = load_embeddings_index(ticker, project_root)
        return search_embeddings(
            question,
            index,
            chunks,
            k=k,
            filing_types=filing_types,
            require_each_filing_type=require_each_filing_type,
        )

    if retriever == "bm25":
        try:
            index, chunks = load_bm25_index(ticker, project_root)
        except (FileNotFoundError, IndexValidationError):
            if not build_if_missing:
                raise
            build_bm25_index(ticker, project_root)
            index, chunks = load_bm25_index(ticker, project_root)
        return search_bm25(
            question,
            index,
            chunks,
            k=k,
            filing_types=filing_types,
            require_each_filing_type=require_each_filing_type,
        )

    raise ValueError(f"Unknown retriever {retriever!r}. Choose 'faiss' or 'bm25'.")
