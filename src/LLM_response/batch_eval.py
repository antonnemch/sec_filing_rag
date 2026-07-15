"""Batch evaluation: retrieve chunks, call LLM, and save results."""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from src.data.utils import PROJECT_ROOT
from src.LLM_response.LLM import DEFAULT_MODEL, answer_question
from src.LLM_response.ground_truth import load_eval_set
from src.LLM_response.retrieve_context import retrieve_chunks

FAANG_TICKERS = ["META", "AMZN", "AAPL", "NFLX", "GOOG"]


def run_batch_eval(
    tickers: list[str] = FAANG_TICKERS,
    retriever: str = "faiss",
    k: int = 5,
    llm_model: str = DEFAULT_MODEL,
    project_root: Path = PROJECT_ROOT,
    eval_csv: Path | None = None,
    output_csv: Path | None = None,
    resume: bool = False,
    append: bool = False,
) -> pd.DataFrame:
    """Run the RAG eval loop for one retriever and write results to CSV.

    resume=True  — re-run only (qa_id, retriever) pairs with status='error'.
    append=True  — add rows for this retriever without touching existing rows
                   for other retrievers (used when running 'both').
    """
    if output_csv is None:
        output_csv = project_root / "outputs" / "eval_results" / "eval_results.csv"
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    eval_df = load_eval_set(eval_csv) if eval_csv else load_eval_set()
    upper_tickers = [t.upper() for t in tickers]
    eval_df = eval_df[eval_df["ticker"].isin(upper_tickers)].copy()

    existing: pd.DataFrame | None = None
    if (resume or append) and output_csv.exists():
        existing = pd.read_csv(output_csv)

        if resume:
            # Re-run only rows where this retriever failed
            failed = set(
                existing.loc[
                    (existing["status"] == "error") & (existing["retriever"] == retriever),
                    "qa_id",
                ]
            )
            if not failed:
                print(f"No failed rows for retriever={retriever} — nothing to resume.")
                return existing
            eval_df = eval_df[eval_df["qa_id"].isin(failed)].copy()
            print(f"Resuming {len(eval_df)} failed row(s) for {retriever}: {sorted(failed)}")

        elif append:
            # Skip rows already completed for this retriever
            done = set(
                existing.loc[existing["retriever"] == retriever, "qa_id"]
            )
            eval_df = eval_df[~eval_df["qa_id"].isin(done)].copy()
            if eval_df.empty:
                print(f"All rows already present for retriever={retriever} — skipping.")
                return existing
            print(f"Appending {len(eval_df)} new row(s) for retriever={retriever}.")

    results: list[dict] = []
    total = len(eval_df)
    for i, row in enumerate(eval_df.itertuples(index=False), 1):
        ticker = row.ticker
        question = row.question
        print(f"[{i}/{total}] {ticker} Q{row.question_id}: {question[:80]}...")

        try:
            retrieved = retrieve_chunks(
                question, ticker, retriever=retriever, k=k, project_root=project_root
            )
            llm_answer = answer_question(question, retrieved, model=llm_model)
            chunk_ids = "|".join(c.get("chunk_id", "") for c in retrieved)
            scores = "|".join(f"{c.get('retrieval_score', 0):.4f}" for c in retrieved)
            status, error = "ok", ""
        except Exception as exc:
            llm_answer, chunk_ids, scores = "", "", ""
            status, error = "error", str(exc)
            print(f"  ERROR: {exc}")

        results.append(
            {
                "qa_id": row.qa_id,
                "ticker": ticker,
                "question_id": row.question_id,
                "question": question,
                "ground_truth": row.answer,
                "llm_answer": llm_answer,
                "retrieved_chunk_ids": chunk_ids,
                "retrieval_scores": scores,
                "retriever": retriever,
                "llm_model": llm_model,
                "status": status,
                "error": error,
            }
        )
        time.sleep(0.3)

    results_df = pd.DataFrame(results)

    if existing is not None:
        # Replace any rows that match on (qa_id, retriever) then re-append
        mask = ~(
            existing["qa_id"].isin(results_df["qa_id"])
            & (existing["retriever"] == retriever)
        )
        results_df = (
            pd.concat([existing[mask], results_df], ignore_index=True)
            .sort_values(["retriever", "qa_id"])
            .reset_index(drop=True)
        )

    results_df.to_csv(output_csv, index=False)
    ok = (results_df["status"] == "ok").sum()
    print(f"\nEval complete: {ok}/{len(results_df)} ok — results saved to {output_csv}")
    return results_df
