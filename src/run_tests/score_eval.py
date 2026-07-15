"""
Score eval results and compare retrievers.

Primary metric — semantic cosine similarity (OpenAI embeddings):
  Embeds both the LLM answer and the ground truth, then computes cosine
  similarity between them. 1.0 = identical meaning, ~0.8+ = strong match.
  All texts are embedded in a single batched API call.

Secondary metric — word overlap F1 (ROUGE-1, no API needed):
  Harmonic mean of word-set precision and recall after stripping dummy markers.

When the results file contains both 'faiss' and 'bm25' rows, a side-by-side
comparison table is printed so you can see which retriever produced better
LLM answers.

Usage:
    python -m src.run_tests.score_eval                  # cosine + word overlap
    python -m src.run_tests.score_eval --no-embeddings  # word overlap only
    python -m src.run_tests.score_eval --llm-judge      # also add LLM 1-5 scores
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.utils import PROJECT_ROOT

_DUMMY_RE = re.compile(r"\[DUMMY[^\]]*\]\s*", re.IGNORECASE)
_RESULTS_CSV = PROJECT_ROOT / "outputs" / "eval_results" / "eval_results.csv"
_EMBED_MODEL = "text-embedding-3-small"
_BATCH = 100

CATEGORY_LABELS = {1: "Business overview", 2: "Financial/ops", 3: "Risk", 4: "Recent dev"}


# ── helpers ──────────────────────────────────────────────────────────────────

def _strip_dummy(text: str) -> str:
    return _DUMMY_RE.sub("", str(text)).strip()


def _word_f1(pred: str, ref: str) -> float:
    p = set(_strip_dummy(pred).lower().split())
    r = set(_strip_dummy(ref).lower().split())
    if not p or not r:
        return 0.0
    overlap = len(p & r)
    prec = overlap / len(p)
    rec = overlap / len(r)
    return round(2 * prec * rec / (prec + rec), 4) if (prec + rec) else 0.0


def _embed(texts: list[str]) -> np.ndarray:
    from openai import OpenAI
    client = OpenAI()
    all_vecs: list[list[float]] = []
    for i in range(0, len(texts), _BATCH):
        resp = client.embeddings.create(model=_EMBED_MODEL, input=texts[i : i + _BATCH])
        all_vecs.extend(item.embedding for item in resp.data)
        print(f"  Embedded {min(i + _BATCH, len(texts))}/{len(texts)} texts")
    vecs = np.array(all_vecs, dtype="float32")
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / np.where(norms == 0, 1, norms)


def _cosine_scores(df: pd.DataFrame) -> list[float]:
    """Embed all LLM answers + ground truths in one batch, return per-row cosine sim."""
    answers = [_strip_dummy(t) for t in df["llm_answer"]]
    truths = [_strip_dummy(t) for t in df["ground_truth"]]

    print(f"Embedding {len(answers)} answers and {len(truths)} ground truths...")
    combined = _embed(answers + truths)
    ans_vecs = combined[: len(answers)]
    gt_vecs = combined[len(answers) :]
    return [float(np.dot(a, g)) for a, g in zip(ans_vecs, gt_vecs)]


def _llm_judge(question: str, ground_truth: str, llm_answer: str) -> int:
    import anthropic
    prompt = (
        f"Question: {question}\n\n"
        f"Reference answer: {ground_truth}\n\n"
        f"Model answer: {llm_answer}\n\n"
        "Rate the model answer 1-5 for factual alignment with the reference "
        "(1=wrong/missing, 5=fully correct). Reply with one integer only."
    )
    msg = anthropic.Anthropic().messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=8,
        messages=[{"role": "user", "content": prompt}],
    )
    try:
        return int(msg.content[0].text.strip())
    except ValueError:
        return 0


# ── scoring ───────────────────────────────────────────────────────────────────

def score(
    df: pd.DataFrame,
    use_embeddings: bool = True,
    use_llm_judge: bool = False,
) -> pd.DataFrame:
    ok = df[df["status"] == "ok"].copy()

    ok["word_overlap_f1"] = ok.apply(
        lambda r: _word_f1(r["llm_answer"], r["ground_truth"]), axis=1
    )

    if use_embeddings:
        ok["cosine_sim"] = _cosine_scores(ok)

    if use_llm_judge:
        print("Running LLM-as-judge...")
        judges = []
        for i, row in enumerate(ok.itertuples(index=False), 1):
            print(f"  Judging [{i}/{len(ok)}] {row.qa_id} ({row.retriever})...")
            judges.append(_llm_judge(row.question, row.ground_truth, row.llm_answer))
        ok["llm_score"] = judges

    return ok


# ── reporting ─────────────────────────────────────────────────────────────────

def _score_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in ["cosine_sim", "word_overlap_f1", "llm_score"] if c in df.columns]


def print_report(df: pd.DataFrame) -> None:
    retrievers = sorted(df["retriever"].unique())
    score_cols = _score_cols(df)

    # Per-row detail
    print("\n── Per-row results ──────────────────────────────────────────────────────")
    for row in df.sort_values(["retriever", "qa_id"]).itertuples(index=False):
        metrics = "  ".join(
            f"{c}={getattr(row, c):.3f}" for c in score_cols
        )
        print(f"{row.qa_id:<12} [{row.retriever:<5}]  {metrics}")
        print(f"  Q  : {row.question[:90]}")
        print(f"  GT : {_strip_dummy(str(row.ground_truth))[:90]}")
        print(f"  LLM: {str(row.llm_answer)[:90]}")
        print()

    # Side-by-side retriever comparison (only when both are present)
    if len(retrievers) > 1:
        print("── Retriever comparison (mean scores) ───────────────────────────────────")
        pivot = df.groupby("retriever")[score_cols].mean().round(4)
        print(pivot.to_string())
        print()

        for col in score_cols:
            wide = df.pivot_table(index="qa_id", columns="retriever", values=col)
            wide["delta (faiss-bm25)"] = wide.get("faiss", 0) - wide.get("bm25", 0)
            print(f"── {col} per question ──────────────────────────────────────────────────")
            print(wide.round(4).to_string())
            print()

    # By ticker
    print("── By ticker ────────────────────────────────────────────────────────────")
    if len(retrievers) > 1:
        print(df.groupby(["ticker", "retriever"])[score_cols].mean().round(4).to_string())
    else:
        print(df.groupby("ticker")[score_cols].mean().round(4).to_string())

    # By category
    df["category"] = df["question_id"].apply(
        lambda q: CATEGORY_LABELS.get((int(q) - 1) // 3 + 1, "Other")
    )
    print("\n── By category ──────────────────────────────────────────────────────────")
    if len(retrievers) > 1:
        print(df.groupby(["category", "retriever"])[score_cols].mean().round(4).to_string())
    else:
        print(df.groupby("category")[score_cols].mean().round(4).to_string())

    # Overall
    print("\n── Overall ──────────────────────────────────────────────────────────────")
    print(df.groupby("retriever")[score_cols].mean().round(4).to_string())


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score RAG eval results vs ground truth.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input", type=Path, default=_RESULTS_CSV,
        help="Results CSV from run_eval.py.",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Path to save scored CSV (default: <input>_scored.csv).",
    )
    parser.add_argument(
        "--no-embeddings", action="store_true",
        help="Skip cosine similarity (word overlap only, no API calls).",
    )
    parser.add_argument(
        "--llm-judge", action="store_true",
        help="Add LLM-as-judge scores 1-5 per row (uses claude-haiku).",
    )
    args = parser.parse_args()

    if not args.input.exists():
        raise SystemExit(f"Results file not found: {args.input}")

    df = pd.read_csv(args.input)
    scored = score(df, use_embeddings=not args.no_embeddings, use_llm_judge=args.llm_judge)
    print_report(scored)

    out = args.output or args.input.parent / (args.input.stem + "_scored.csv")
    scored.to_csv(out, index=False)
    print(f"\nScored results saved to {out}")


if __name__ == "__main__":
    main()
