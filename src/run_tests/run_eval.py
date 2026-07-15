"""
Run the FAANG SEC filing RAG evaluation end-to-end.

This script:
  1. Optionally builds filing datasets (download → clean → chunk) for each ticker.
  2. Builds or loads an embeddings (FAISS) or BM25 index per ticker.
  3. For every question in eval_sets/faang_eval_set_dummy.csv, retrieves the
     top-k most relevant chunks and calls the LLM for an answer.
  4. Writes results to outputs/eval_results/eval_results.csv.

Usage:
    python -m src.run_tests.run_eval
    python -m src.run_tests.run_eval --tickers AMZN AAPL --retriever bm25 --k 3
    python -m src.run_tests.run_eval --skip-build --retriever faiss
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.data.build_dataset import build_dataset
from src.data.utils import PROJECT_ROOT, ticker_slug
from src.injest_data.bm25 import build_bm25_index, load_bm25_index
from src.injest_data.embeddings import build_embeddings_index, load_embeddings_index
from src.LLM_response.batch_eval import FAANG_TICKERS, run_batch_eval
from src.LLM_response.LLM import DEFAULT_MODEL

FAANG = ["META", "AMZN", "AAPL", "NFLX", "GOOG"]


def _chunks_csv(ticker: str, project_root: Path) -> Path:
    slug = ticker_slug(ticker)
    return project_root / "data" / "processed" / slug / f"{slug}_filing_chunks.csv"


def ensure_datasets(
    tickers: list[str], project_root: Path, num_8k: int
) -> None:
    """Build the data pipeline for any ticker missing a chunks CSV."""
    for ticker in tickers:
        if _chunks_csv(ticker, project_root).exists():
            print(f"[{ticker}] Chunks CSV found — skipping download.")
        else:
            print(f"[{ticker}] Building dataset (download → clean → chunk)...")
            build_dataset(ticker=ticker, num_8k=num_8k, project_root=project_root)


def ensure_indexes(
    tickers: list[str], retriever: str, project_root: Path
) -> None:
    """Build retrieval indexes for tickers that don't have one yet."""
    retrievers = ["faiss", "bm25"] if retriever == "both" else [retriever]
    for r in retrievers:
        for ticker in tickers:
            if r == "faiss":
                try:
                    load_embeddings_index(ticker, project_root)
                    print(f"[{ticker}] FAISS index found — skipping build.")
                except FileNotFoundError:
                    build_embeddings_index(ticker, project_root)
            else:
                try:
                    load_bm25_index(ticker, project_root)
                    print(f"[{ticker}] BM25 index found — skipping build.")
                except FileNotFoundError:
                    build_bm25_index(ticker, project_root)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the FAANG SEC filing RAG evaluation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--tickers", nargs="+", default=FAANG,
        metavar="TICKER",
        help="Tickers to evaluate.",
    )
    parser.add_argument(
        "--retriever", choices=["faiss", "bm25", "both"], default="faiss",
        help="Retrieval method. 'both' runs faiss then bm25 into the same file.",
    )
    parser.add_argument("--k", type=int, default=5, help="Top-k chunks to retrieve.")
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help="Claude model for answering questions.",
    )
    parser.add_argument(
        "--num-8k", type=int, default=5,
        help="Number of 8-K filings to download per ticker (if building).",
    )
    parser.add_argument(
        "--skip-build", action="store_true",
        help="Skip dataset download even if the chunks CSV is missing.",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output CSV path (default: outputs/eval_results/eval_results.csv).",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Re-run only rows with status=error in the existing results CSV.",
    )
    args = parser.parse_args()

    tickers = [t.upper() for t in args.tickers]

    if not args.skip_build:
        ensure_datasets(tickers, PROJECT_ROOT, num_8k=args.num_8k)

    ensure_indexes(tickers, args.retriever, PROJECT_ROOT)

    if args.retriever == "both":
        # Run faiss first (creates/overwrites the file), then bm25 (appends)
        run_batch_eval(
            tickers=tickers, retriever="faiss", k=args.k, llm_model=args.model,
            project_root=PROJECT_ROOT, output_csv=args.output, resume=args.resume,
        )
        run_batch_eval(
            tickers=tickers, retriever="bm25", k=args.k, llm_model=args.model,
            project_root=PROJECT_ROOT, output_csv=args.output, resume=args.resume,
            append=True,
        )
    else:
        run_batch_eval(
            tickers=tickers,
            retriever=args.retriever,
            k=args.k,
            llm_model=args.model,
            project_root=PROJECT_ROOT,
            output_csv=args.output,
            resume=args.resume,
        )


if __name__ == "__main__":
    main()
