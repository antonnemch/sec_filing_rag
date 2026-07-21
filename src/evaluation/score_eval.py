"""Score durable RAG evaluation results while preserving failed rows."""

from __future__ import annotations

import argparse
import io
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import (
    API_MAX_RETRIES,
    API_TIMEOUT_SECONDS,
    CATEGORY_LABELS,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_JUDGE_MODEL,
    EVALUATION_SCHEMA_VERSION,
)
from src.data.utils import PROJECT_ROOT, atomic_write_text, load_project_env
from src.LLM_response.LLM import _extract_text_content
from src.evaluation.run_artifacts import newest_run_output, output_for_run
from src.visualizations import generate_evaluation_figures


_EMBED_BATCH_SIZE = 100
_TOKEN_RE = re.compile(r"\b\w+(?:[.-]\w+)*\b", re.UNICODE)
_JUDGE_SCORE_RE = re.compile(r"[1-5]\.?")
_REQUIRED_RESULT_COLUMNS = {
    "schema_version",
    "run_fingerprint",
    "qa_id",
    "ticker",
    "question",
    "ground_truth",
    "source_doc_id",
    "retrieved_doc_ids",
    "retrieved_chunk_ids",
    "retrieval_k",
    "retriever",
    "category",
    "answerable",
    "llm_answer",
    "retrieval_status",
    "generation_status",
    "status",
}

def _text(value: object) -> str:
    return "" if pd.isna(value) else str(value).strip()


def _tokens(value: object) -> list[str]:
    return _TOKEN_RE.findall(_text(value).lower())


def _word_f1(prediction: object, reference: object) -> float:
    """Return frequency-aware unigram precision/recall F1."""

    predicted = Counter(_tokens(prediction))
    expected = Counter(_tokens(reference))
    if not predicted or not expected:
        return 0.0
    overlap = sum((predicted & expected).values())
    precision = overlap / sum(predicted.values())
    recall = overlap / sum(expected.values())
    return round(2 * precision * recall / (precision + recall), 4) if overlap else 0.0


def _split_ids(value: object) -> list[str]:
    return [item.strip() for item in _text(value).split("|") if item.strip()]


def validate_results_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize the clean-break result schema."""

    missing = sorted(_REQUIRED_RESULT_COLUMNS.difference(df.columns))
    if missing:
        raise ValueError(
            "Results CSV is not compatible with evaluation schema v2; missing column(s): "
            + ", ".join(missing)
            + ". Rerun the evaluation to rebuild it."
        )
    if df.empty:
        raise ValueError("Results CSV contains no evaluation rows.")
    versions = pd.to_numeric(df["schema_version"], errors="coerce")
    if versions.isna().any() or set(versions.astype(int)) != {EVALUATION_SCHEMA_VERSION}:
        raise ValueError(
            f"Results CSV must contain only schema version {EVALUATION_SCHEMA_VERSION}."
        )
    if df["run_fingerprint"].isna().any() or df["run_fingerprint"].astype(str).str.strip().eq("").any():
        raise ValueError("Results CSV contains a missing run fingerprint.")
    if len(set(df["run_fingerprint"].astype(str))) != 1:
        raise ValueError("Results CSV mixes records from incompatible evaluation runs.")
    if df.duplicated(["qa_id", "retriever"]).any():
        raise ValueError("Results CSV contains duplicate (qa_id, retriever) rows.")
    if df["qa_id"].isna().any() or df["qa_id"].astype(str).str.strip().eq("").any():
        raise ValueError("Results CSV contains a missing or blank qa_id.")
    if not set(df["retriever"].astype(str)).issubset({"faiss", "bm25"}):
        raise ValueError("Results CSV contains an invalid retriever.")
    if not set(df["retrieval_status"].astype(str)).issubset({"ok", "error"}):
        raise ValueError("Results CSV contains an invalid retrieval_status.")
    if not set(df["generation_status"].astype(str)).issubset({"ok", "error", "not_run"}):
        raise ValueError("Results CSV contains an invalid generation_status.")
    if not set(df["status"].astype(str)).issubset(
        {"ok", "retrieval_error", "generation_error"}
    ):
        raise ValueError("Results CSV contains an invalid overall status.")
    expected_status = {
        ("ok", "ok"): "ok",
        ("ok", "error"): "generation_error",
        ("error", "not_run"): "retrieval_error",
    }
    inconsistent = df.apply(
        lambda row: expected_status.get(
            (str(row["retrieval_status"]), str(row["generation_status"]))
        )
        != str(row["status"]),
        axis=1,
    )
    if inconsistent.any():
        raise ValueError("Results CSV contains inconsistent stage and overall statuses.")
    retrieval_k = pd.to_numeric(df["retrieval_k"], errors="coerce")
    if retrieval_k.isna().any() or (retrieval_k < 1).any():
        raise ValueError("Results CSV contains an invalid retrieval_k; expected at least 1.")
    categories = pd.to_numeric(df["category"], errors="coerce")
    if categories.isna().any() or not set(categories.astype(int)).issubset(CATEGORY_LABELS):
        raise ValueError("Results CSV contains an invalid category.")
    normalized = df.copy()
    normalized["category"] = categories.astype(int)
    normalized["retrieval_k"] = retrieval_k.astype(int)
    answerable = normalized["answerable"].astype(str).str.strip().str.lower()
    if not set(answerable).issubset({"true", "false", "1", "0"}):
        raise ValueError("Results CSV contains an invalid answerable value.")
    normalized["answerable"] = answerable.isin({"true", "1"})
    return normalized


def _retrieval_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Add document metrics and optional chunk metrics to supplied rows."""

    scored = df.copy()
    if "source_doc_ids" not in scored.columns:
        scored["source_doc_ids"] = scored["source_doc_id"].apply(
            lambda value: "" if _text(value).endswith("_MULTI") else _text(value)
        )
    if "source_chunk_ids" not in scored.columns:
        scored["source_chunk_ids"] = ""

    def metrics(row: pd.Series, gold_column: str, retrieved_column: str) -> tuple[float, float]:
        gold = set(_split_ids(row.get(gold_column, "")))
        retrieved = _split_ids(row.get(retrieved_column, ""))
        if not gold:
            return np.nan, np.nan
        relevant_ranks = [
            rank for rank, item in enumerate(retrieved, start=1) if item in gold
        ]
        recall = len(gold.intersection(retrieved)) / len(gold)
        reciprocal_rank = 1.0 / relevant_ranks[0] if relevant_ranks else 0.0
        return recall, reciprocal_rank

    document = scored.apply(
        lambda row: metrics(row, "source_doc_ids", "retrieved_doc_ids"), axis=1
    )
    scored[["document_recall_at_k", "document_reciprocal_rank"]] = pd.DataFrame(
        document.tolist(), index=scored.index
    )
    chunks = scored.apply(
        lambda row: metrics(row, "source_chunk_ids", "retrieved_chunk_ids"), axis=1
    )
    scored[["chunk_recall_at_k", "chunk_reciprocal_rank"]] = pd.DataFrame(
        chunks.tolist(), index=scored.index
    )
    return scored


def _openai_client():
    from openai import OpenAI

    load_project_env()
    return OpenAI(timeout=API_TIMEOUT_SECONDS, max_retries=API_MAX_RETRIES)


def _embed(texts: list[str]) -> np.ndarray:
    if not texts:
        return np.empty((0, 0), dtype="float32")
    client = _openai_client()
    all_vectors: list[list[float]] = []
    for offset in range(0, len(texts), _EMBED_BATCH_SIZE):
        response = client.embeddings.create(
            model=DEFAULT_EMBEDDING_MODEL,
            input=texts[offset : offset + _EMBED_BATCH_SIZE],
        )
        all_vectors.extend(item.embedding for item in response.data)
        print(f"  Embedded {min(offset + _EMBED_BATCH_SIZE, len(texts))}/{len(texts)} texts")
    vectors = np.asarray(all_vectors, dtype="float32")
    if vectors.ndim != 2 or len(vectors) != len(texts):
        raise RuntimeError("OpenAI returned an invalid embedding matrix while scoring.")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.where(norms == 0, 1, norms)


def _cosine_scores(df: pd.DataFrame) -> list[float]:
    answers = [_text(value) for value in df["llm_answer"]]
    truths = [_text(value) for value in df["ground_truth"]]
    if not answers:
        return []
    print(f"Embedding {len(answers)} answers and {len(truths)} ground truths...")
    combined = _embed(answers + truths)
    midpoint = len(answers)
    return [
        float(np.dot(answer, truth))
        for answer, truth in zip(combined[:midpoint], combined[midpoint:])
    ]


def _parse_judge_score(value: str) -> float:
    normalized = value.strip()
    if not _JUDGE_SCORE_RE.fullmatch(normalized):
        return np.nan
    return float(int(normalized.rstrip(".")))


def _llm_judge(question: str, ground_truth: str, llm_answer: str) -> tuple[float, str]:
    import anthropic

    load_project_env()
    prompt = (
        f"Question: {question}\n\n"
        f"Reference answer: {ground_truth}\n\n"
        f"Model answer: {llm_answer}\n\n"
        "Rate factual alignment from 1 to 5. Reply with one integer only, where "
        "1 means wrong or missing and 5 means fully correct."
    )
    try:
        message = anthropic.Anthropic(
            timeout=API_TIMEOUT_SECONDS,
            max_retries=API_MAX_RETRIES,
        ).messages.create(
            model=DEFAULT_JUDGE_MODEL,
            max_tokens=8,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = _extract_text_content(message)
        score = _parse_judge_score(raw)
        if pd.isna(score):
            return np.nan, f"Invalid judge response: {raw!r}"
        return score, ""
    except Exception as exc:
        return np.nan, str(exc)


def score(
    df: pd.DataFrame,
    use_embeddings: bool = True,
    use_llm_judge: bool = False,
) -> pd.DataFrame:
    """Add eligible metrics while retaining every evaluation row."""

    scored = validate_results_schema(df)
    metric_columns = [
        "document_recall_at_k",
        "document_reciprocal_rank",
        "chunk_recall_at_k",
        "chunk_reciprocal_rank",
        "word_overlap_f1",
        "cosine_sim",
        "llm_score",
    ]
    for column in metric_columns:
        scored[column] = np.nan
    scored["llm_judge_error"] = ""
    scored["retrieval_success"] = (scored["retrieval_status"] == "ok").astype(float)
    scored["generation_success"] = (scored["generation_status"] == "ok").astype(float)

    retrieval_mask = scored["retrieval_status"] == "ok"
    if retrieval_mask.any():
        retrieval = _retrieval_scores(scored.loc[retrieval_mask].copy())
        for column in (
            "document_recall_at_k",
            "document_reciprocal_rank",
            "chunk_recall_at_k",
            "chunk_reciprocal_rank",
        ):
            scored.loc[retrieval.index, column] = retrieval[column]

    answer_mask = (
        (scored["generation_status"] == "ok")
        & scored["ground_truth"].apply(lambda value: bool(_text(value)))
    )
    if answer_mask.any():
        scored.loc[answer_mask, "word_overlap_f1"] = scored.loc[answer_mask].apply(
            lambda row: _word_f1(row["llm_answer"], row["ground_truth"]), axis=1
        )
        if use_embeddings:
            scored.loc[answer_mask, "cosine_sim"] = _cosine_scores(
                scored.loc[answer_mask]
            )
        if use_llm_judge:
            eligible = scored.loc[answer_mask]
            print(f"Running LLM-as-judge for {len(eligible)} eligible answer(s)...")
            for position, (index, row) in enumerate(eligible.iterrows(), 1):
                print(f"  Judging [{position}/{len(eligible)}] {row.qa_id} ({row.retriever})...")
                judge_score, error = _llm_judge(
                    _text(row.question), _text(row.ground_truth), _text(row.llm_answer)
                )
                scored.at[index, "llm_score"] = judge_score
                scored.at[index, "llm_judge_error"] = error
    else:
        print("No generated answers with nonblank references; answer metrics are unavailable.")
    return scored


def _score_columns(df: pd.DataFrame) -> list[str]:
    candidates = [
        "retrieval_success",
        "generation_success",
        "document_recall_at_k",
        "document_reciprocal_rank",
        "chunk_recall_at_k",
        "chunk_reciprocal_rank",
        "cosine_sim",
        "word_overlap_f1",
        "llm_score",
    ]
    return [column for column in candidates if column in df and df[column].notna().any()]


def _print_group(df: pd.DataFrame, groups: list[str], score_columns: list[str]) -> None:
    if not score_columns:
        print("No available metrics.")
        return
    print(df.groupby(groups, dropna=False)[score_columns].mean().round(4).to_string())


def print_report(df: pd.DataFrame) -> None:
    """Print status-aware metric coverage and aggregate comparisons."""

    score_columns = _score_columns(df)
    total = len(df)
    retrieval_ok = int((df["retrieval_status"] == "ok").sum())
    generation_ok = int((df["generation_status"] == "ok").sum())
    print("\nEvaluation coverage")
    print(f"  Rows: {total}")
    print(f"  Retrieval successful: {retrieval_ok}/{total}")
    print(f"  Generation successful: {generation_ok}/{total}")
    for column in score_columns:
        print(f"  {column}: {int(df[column].notna().sum())}/{total} eligible")

    print("\nPer-row results")
    for row in df.sort_values(["retriever", "qa_id"]).itertuples(index=False):
        metrics = "  ".join(
            f"{column}={getattr(row, column):.3f}"
            for column in score_columns
            if not pd.isna(getattr(row, column))
        )
        print(f"{row.qa_id} [{row.retriever}] status={row.status}  {metrics}".rstrip())
        if row.status != "ok":
            error = _text(getattr(row, "retrieval_error", "")) or _text(
                getattr(row, "generation_error", "")
            )
            if error:
                print(f"  Error: {error[:160]}")

    report = df.copy()
    report["category_label"] = report["category"].map(CATEGORY_LABELS)
    print("\nBy retriever")
    _print_group(report, ["retriever"], score_columns)
    print("\nBy ticker and retriever")
    _print_group(report, ["ticker", "retriever"], score_columns)
    print("\nBy category and retriever")
    _print_group(report, ["category_label", "retriever"], score_columns)
    print("\nBy answerable and retriever")
    _print_group(report, ["answerable", "retriever"], score_columns)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Score RAG evaluation results against ground truth.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Results CSV to score; defaults to the newest isolated run.",
    )
    input_group.add_argument(
        "--run-name",
        default=None,
        help="Run directory name under outputs/eval_results/runs.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Scored CSV; defaults to <input>_scored.csv.",
    )
    parser.add_argument(
        "--no-embeddings",
        action="store_true",
        help="Skip semantic cosine similarity and its OpenAI calls.",
    )
    parser.add_argument(
        "--llm-judge",
        action="store_true",
        help="Add optional Claude 1-5 factual-alignment scores.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Explicitly replace an existing scored CSV and performance figures.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        input_path = (
            output_for_run(PROJECT_ROOT, args.run_name)
            if args.run_name
            else args.input or newest_run_output(PROJECT_ROOT)
        )
        if not input_path.exists():
            raise FileNotFoundError(f"Results file not found: {input_path}")
        output = args.output or input_path.parent / f"{input_path.stem}_scored.csv"
        figures_dir = output.parent / "figures"
        existing_performance_figures = list(figures_dir.glob("0[3-9]_*.png")) + list(
            figures_dir.glob("1[01]_*.png")
        )
        if not args.overwrite and (output.exists() or existing_performance_figures):
            existing = [output] if output.exists() else []
            existing.extend(existing_performance_figures)
            raise FileExistsError(
                "Refusing to overwrite existing scored artifacts: "
                + ", ".join(str(path) for path in existing)
                + ". Use --overwrite or select a different --output."
            )
        scored = score(
            pd.read_csv(input_path),
            use_embeddings=not args.no_embeddings,
            use_llm_judge=args.llm_judge,
        )
        print_report(scored)
        buffer = io.StringIO()
        scored.to_csv(buffer, index=False)
        atomic_write_text(output, buffer.getvalue())
        try:
            generate_evaluation_figures(
                scored,
                figures_dir,
            )
        except Exception as exc:
            print(f"Warning: visualization generation skipped: {exc}")
        print(f"\nScored results saved to {output}")
    except (FileExistsError, FileNotFoundError, RuntimeError, ValueError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
