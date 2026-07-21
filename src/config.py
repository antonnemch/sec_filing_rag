"""Canonical runtime configuration for the SEC filing RAG project."""

from __future__ import annotations

from datetime import date


DEFAULT_TICKERS = ("META", "AMZN", "AAPL", "NFLX", "GOOG")
DEFAULT_MULTI_COMPANY_SLUG = "faang"

MAX_FILING_DATE = date(2026, 7, 14)
DEFAULT_NUM_8K = 1
DEFAULT_CHUNK_SIZE = 400
DEFAULT_CHUNK_OVERLAP = 75
CHUNK_SCHEMA_VERSION = 2

DEFAULT_RETRIEVER = "faiss"
DEFAULT_RETRIEVAL_K = 5
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_EMBEDDING_BATCH_SIZE = 50
DENSE_DOCUMENT_FORMAT_VERSION = "filing-metadata-v1"
RETRIEVAL_POLICY_VERSION = "filing-type-routing-v1"
DEFAULT_LLM_MODEL = "claude-sonnet-4-6"
DEFAULT_JUDGE_MODEL = "claude-haiku-4-5-20251001"
DEFAULT_LLM_MAX_TOKENS = 1024

CATEGORY_LABELS = {
    1: "Business overview",
    2: "Financial/operations",
    3: "Risk",
    4: "Recent developments",
}

API_TIMEOUT_SECONDS = 60.0
API_MAX_RETRIES = 3

INDEX_SCHEMA_VERSION = 1
EVALUATION_SCHEMA_VERSION = 2
