"""Durable batch evaluation: retrieve, answer, checkpoint, and materialize CSV."""

from __future__ import annotations

import io
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import (
    DEFAULT_LLM_MODEL,
    DEFAULT_RETRIEVAL_K,
    DEFAULT_TICKERS,
    EVALUATION_SCHEMA_VERSION,
)
from src.data.utils import (
    PROJECT_ROOT,
    atomic_write_text,
    normalize_tickers,
    read_json,
    sha256_file,
    stable_json_hash,
    write_json,
)
from src.ingest_data.bm25 import load_bm25_index
from src.ingest_data.embeddings import load_embeddings_index
from src.ingest_data.index_common import index_manifest_path
from src.LLM_response.ground_truth import (
    DEFAULT_EVAL_CSV,
    filter_completed_questions,
    load_eval_set,
    validate_source_doc_ids,
)
from src.LLM_response.LLM import SYSTEM_PROMPT, answer_question
from src.LLM_response.retrieve_context import retrieve_chunks


RESULT_COLUMNS = [
    "schema_version",
    "run_fingerprint",
    "qa_id",
    "ticker",
    "question_id",
    "category",
    "answerable",
    "question",
    "ground_truth",
    "source_doc_id",
    "source_doc_ids",
    "source_chunk_ids",
    "llm_answer",
    "retrieved_chunk_ids",
    "retrieved_doc_ids",
    "retrieval_scores",
    "retrieved_sections",
    "retrieved_filing_dates",
    "retrieval_k",
    "retriever",
    "llm_model",
    "retrieval_status",
    "retrieval_error",
    "generation_status",
    "generation_error",
    "status",
]


def _document_id(chunk: dict[str, Any]) -> str:
    return "_".join(
        [
            str(chunk.get("ticker", "")).upper(),
            str(chunk.get("filing_type", "")),
            str(chunk.get("filing_date", "")),
        ]
    )


def _checkpoint_path(output_csv: Path) -> Path:
    return output_csv.with_name(output_csv.name + ".checkpoint.jsonl")


def _manifest_path(output_csv: Path) -> Path:
    return output_csv.with_name(output_csv.name + ".manifest.json")


def _index_fingerprint(ticker: str, retriever: str, project_root: Path) -> str:
    if retriever == "faiss":
        load_embeddings_index(ticker, project_root)
    else:
        load_bm25_index(ticker, project_root)
    manifest = read_json(index_manifest_path(ticker, retriever, project_root))
    return str(manifest["index_fingerprint"])


def _build_run_manifest(
    eval_df: pd.DataFrame,
    eval_path: Path,
    tickers: tuple[str, ...],
    retrievers: tuple[str, ...],
    k: int,
    llm_model: str,
    include_incomplete: bool,
    project_root: Path,
) -> dict[str, Any]:
    selected_rows = json.loads(eval_df.fillna("").to_json(orient="records"))
    configuration = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "tickers": list(tickers),
        "retrievers": list(retrievers),
        "retrieval_k": k,
        "llm_model": llm_model,
        "include_incomplete": include_incomplete,
        "eval_csv": str(eval_path.resolve()),
        "eval_csv_sha256": sha256_file(eval_path),
        "selected_rows_sha256": stable_json_hash(selected_rows),
        "system_prompt_sha256": stable_json_hash(SYSTEM_PROMPT),
        "indexes": {
            f"{ticker}:{retriever}": _index_fingerprint(
                ticker, retriever, project_root
            )
            for ticker in tickers
            for retriever in retrievers
        },
    }
    return {
        **configuration,
        "run_fingerprint": stable_json_hash(configuration),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def _read_checkpoint(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    """Read last-write-wins records, tolerating only a malformed final line."""

    if not path.exists():
        return {}
    raw_checkpoint = path.read_text(encoding="utf-8")
    lines = raw_checkpoint.splitlines()
    nonempty = [(index, line) for index, line in enumerate(lines) if line.strip()]
    records: dict[tuple[str, str], dict[str, Any]] = {}
    truncated = False
    valid_lines: list[str] = []
    for position, (line_index, line) in enumerate(nonempty):
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            if position == len(nonempty) - 1:
                print(
                    f"Warning: ignoring truncated final checkpoint line {line_index + 1}."
                )
                truncated = True
                break
            raise ValueError(
                f"Checkpoint contains malformed JSON at line {line_index + 1}: {path}"
            ) from exc
        if not isinstance(record, dict) or not record.get("qa_id") or not record.get(
            "retriever"
        ):
            raise ValueError(f"Checkpoint contains an invalid record at line {line_index + 1}.")
        records[(str(record["qa_id"]), str(record["retriever"]))] = record
        valid_lines.append(line)
    if truncated or (raw_checkpoint and not raw_checkpoint.endswith(("\n", "\r"))):
        atomic_write_text(path, "".join(valid_line + "\n" for valid_line in valid_lines))
    return records


def _append_checkpoint(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _materialize_results(
    records: dict[tuple[str, str], dict[str, Any]], output_csv: Path
) -> pd.DataFrame:
    frame = pd.DataFrame(records.values(), columns=RESULT_COLUMNS)
    if not frame.empty:
        frame = frame.sort_values(["retriever", "qa_id"]).reset_index(drop=True)
    buffer = io.StringIO()
    frame.to_csv(buffer, index=False)
    atomic_write_text(output_csv, buffer.getvalue())
    return frame


def _clean_scalar(value: object) -> object:
    if pd.isna(value):
        return ""
    return value.item() if hasattr(value, "item") else value


def _result_for_question(
    row: object,
    retriever: str,
    k: int,
    llm_model: str,
    run_fingerprint: str,
    project_root: Path,
    source_chunk_column: str | None,
    has_source_doc_ids: bool,
) -> dict[str, Any]:
    retrieved: list[dict[str, Any]] = []
    retrieval_error = ""
    generation_error = ""
    llm_answer = ""
    try:
        retrieved = retrieve_chunks(
            str(row.question),
            str(row.ticker),
            retriever=retriever,
            k=k,
            project_root=project_root,
            build_if_missing=False,
        )
        if not retrieved:
            raise RuntimeError("Retriever returned no chunks.")
        retrieval_status = "ok"
    except Exception as exc:
        retrieval_status = "error"
        retrieval_error = str(exc)

    if retrieval_status == "ok":
        try:
            llm_answer = answer_question(str(row.question), retrieved, model=llm_model)
            generation_status = "ok"
        except Exception as exc:
            generation_status = "error"
            generation_error = str(exc)
    else:
        generation_status = "not_run"

    status = (
        "ok"
        if generation_status == "ok"
        else "retrieval_error"
        if retrieval_status == "error"
        else "generation_error"
    )
    source_doc_id = str(row.source_doc_id).strip()
    plural_value = getattr(row, "source_doc_ids", "") if has_source_doc_ids else ""
    source_doc_ids = (
        ""
        if pd.isna(plural_value)
        else str(plural_value).strip()
    )
    if not source_doc_ids and not source_doc_id.endswith("_MULTI"):
        source_doc_ids = source_doc_id
    source_chunk_ids = (
        _clean_scalar(getattr(row, source_chunk_column))
        if source_chunk_column is not None
        else ""
    )

    return {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "run_fingerprint": run_fingerprint,
        "qa_id": str(row.qa_id),
        "ticker": str(row.ticker),
        "question_id": _clean_scalar(row.question_id),
        "category": int(row.category),
        "answerable": bool(row.answerable),
        "question": str(row.question),
        "ground_truth": _clean_scalar(row.answer),
        "source_doc_id": source_doc_id,
        "source_doc_ids": source_doc_ids,
        "source_chunk_ids": source_chunk_ids,
        "llm_answer": llm_answer,
        "retrieved_chunk_ids": "|".join(str(chunk.get("chunk_id", "")) for chunk in retrieved),
        "retrieved_doc_ids": "|".join(_document_id(chunk) for chunk in retrieved),
        "retrieval_scores": "|".join(
            f"{float(chunk.get('retrieval_score', 0)):.6f}" for chunk in retrieved
        ),
        "retrieved_sections": "|".join(
            str(chunk.get("section_heading", "")) for chunk in retrieved
        ),
        "retrieved_filing_dates": "|".join(
            str(chunk.get("filing_date", "")) for chunk in retrieved
        ),
        "retrieval_k": k,
        "retriever": retriever,
        "llm_model": llm_model,
        "retrieval_status": retrieval_status,
        "retrieval_error": retrieval_error,
        "generation_status": generation_status,
        "generation_error": generation_error,
        "status": status,
    }


def run_batch_eval(
    tickers: list[str] | tuple[str, ...] = DEFAULT_TICKERS,
    retriever: str = "faiss",
    k: int = DEFAULT_RETRIEVAL_K,
    llm_model: str = DEFAULT_LLM_MODEL,
    project_root: Path = PROJECT_ROOT,
    eval_csv: Path | None = None,
    include_incomplete: bool = False,
    output_csv: Path | None = None,
    resume: bool = False,
) -> pd.DataFrame:
    """Run one fingerprinted evaluation, checkpointing every attempted answer."""

    if k < 1:
        raise ValueError("k must be at least 1.")
    normalized_tickers = normalize_tickers(tickers)
    if retriever not in {"faiss", "bm25", "both"}:
        raise ValueError("retriever must be 'faiss', 'bm25', or 'both'.")
    retrievers = ("faiss", "bm25") if retriever == "both" else (retriever,)
    eval_path = (eval_csv or DEFAULT_EVAL_CSV).resolve()
    output_csv = output_csv or (
        project_root / "outputs" / "eval_results" / "eval_results.csv"
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    eval_df = load_eval_set(eval_path)
    eval_df = eval_df[eval_df["ticker"].isin(normalized_tickers)].copy()
    available = len(eval_df)
    if not include_incomplete:
        eval_df = filter_completed_questions(eval_df)
    print(
        f"Selected {len(eval_df)}/{available} question(s) for the requested tickers"
        + (" including incomplete rows." if include_incomplete else " with completed answers.")
    )
    if eval_df.empty:
        raise ValueError(
            "No evaluation questions were selected. Populate reference answers or use "
            "--include-incomplete."
        )
    validate_source_doc_ids(eval_df, project_root)
    source_chunk_column = next(
        (name for name in ("source_chunk_ids", "source_chunk_id") if name in eval_df.columns),
        None,
    )
    if source_chunk_column is None:
        print("Gold source chunk IDs are absent; chunk retrieval metrics will be unavailable.")

    manifest = _build_run_manifest(
        eval_df,
        eval_path,
        normalized_tickers,
        retrievers,
        k,
        llm_model,
        include_incomplete,
        project_root,
    )
    checkpoint_path = _checkpoint_path(output_csv)
    manifest_path = _manifest_path(output_csv)
    if resume:
        if not manifest_path.exists() or not checkpoint_path.exists():
            raise FileNotFoundError(
                "Cannot resume: the run manifest or checkpoint is missing. Start a fresh run."
            )
        saved_manifest = read_json(manifest_path)
        if saved_manifest.get("run_fingerprint") != manifest["run_fingerprint"]:
            raise ValueError(
                "Cannot resume because the evaluation data, prompt, model, retrieval "
                "configuration, chunks, or indexes changed. Start a fresh run or use a new output path."
            )
        records = _read_checkpoint(checkpoint_path)
    else:
        if checkpoint_path.exists():
            checkpoint_path.unlink()
        write_json(manifest_path, manifest)
        records = {}

    work = [
        (row, method)
        for method in retrievers
        for row in eval_df.itertuples(index=False)
        if records.get((str(row.qa_id), method), {}).get("status") != "ok"
    ]
    if not work:
        print("All selected evaluation records are already successful.")
        return _materialize_results(records, output_csv)

    has_source_doc_ids = "source_doc_ids" in eval_df.columns
    try:
        for position, (row, method) in enumerate(work, 1):
            print(
                f"[{position}/{len(work)}] {row.ticker} Q{row.question_id} "
                f"[{method}]: {str(row.question)[:80]}..."
            )
            record = _result_for_question(
                row,
                method,
                k,
                llm_model,
                manifest["run_fingerprint"],
                project_root,
                source_chunk_column,
                has_source_doc_ids,
            )
            records[(record["qa_id"], method)] = record
            _append_checkpoint(checkpoint_path, record)
            if record["status"] != "ok":
                print(
                    "  ERROR: "
                    + (record["retrieval_error"] or record["generation_error"])
                )
    finally:
        results = _materialize_results(records, output_csv)

    successful = int((results["status"] == "ok").sum()) if not results.empty else 0
    print(
        f"Eval complete: {successful}/{len(results)} successful - results saved to {output_csv}"
    )
    return results
