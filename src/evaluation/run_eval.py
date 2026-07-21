"""Run the SEC filing RAG evaluation end to end."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

from src.config import (
    CHUNK_SCHEMA_VERSION,
    DEFAULT_LLM_MODEL,
    DEFAULT_NUM_8K,
    DEFAULT_RETRIEVAL_K,
    DEFAULT_RETRIEVER,
    DEFAULT_TICKERS,
    MAX_FILING_DATE,
)
from src.data.build_dataset import (
    build_dataset,
    prune_stale_dataset_artifacts,
    refresh_dataset_summary,
)
from src.data.utils import (
    PROJECT_ROOT,
    load_project_env,
    normalize_tickers,
    read_json,
    sha256_file,
    ticker_slug,
)
from src.ingest_data.bm25 import build_bm25_index, load_bm25_index
from src.ingest_data.embeddings import build_embeddings_index, load_embeddings_index
from src.ingest_data.index_common import IndexValidationError
from src.LLM_response.batch_eval import run_batch_eval
from src.LLM_response.ground_truth import (
    filter_completed_questions,
    load_eval_set,
    validate_source_doc_ids,
)
from src.evaluation.run_artifacts import new_run_name, output_for_run
from src.visualizations import plot_evaluation_coverage, plot_filing_timeline


def _chunks_csv(ticker: str, project_root: Path) -> Path:
    slug = ticker_slug(ticker)
    return project_root / "data" / "processed" / slug / f"{slug}_filing_chunks.csv"


def _dataset_matches_filing_policy(chunks_csv: Path, num_8k: int) -> bool:
    """Return whether cached chunks obey the exact form counts and fixed cutoff."""

    try:
        all_chunks = pd.read_csv(
            chunks_csv,
            usecols=[
                "chunk_schema_version",
                "filing_type",
                "filing_date",
                "accession_number",
            ],
        )
        documents = all_chunks.drop_duplicates("accession_number")
        dates = pd.to_datetime(documents["filing_date"], errors="raise").dt.date
    except (FileNotFoundError, KeyError, TypeError, ValueError):
        return False
    counts = documents["filing_type"].value_counts().to_dict()
    return (
        counts.get("10-K", 0) == 1
        and counts.get("10-Q", 0) == 1
        and counts.get("8-K", 0) == num_8k
        and bool((dates <= MAX_FILING_DATE).all())
        and bool(
            (all_chunks["chunk_schema_version"] == CHUNK_SCHEMA_VERSION).all()
        )
    )


def _refresh_summary_if_stale(
    ticker: str, chunks_csv: Path, project_root: Path
) -> None:
    slug = ticker_slug(ticker)
    summary_path = project_root / "outputs" / "data_summary" / f"{slug}_dataset_summary.json"
    inventory_path = project_root / "outputs" / "data_summary" / f"{slug}_filing_inventory.csv"
    cleaning_path = project_root / "outputs" / "data_summary" / f"{slug}_cleaning_summary.csv"
    current_hash = sha256_file(chunks_csv)
    try:
        chunk_frame = pd.read_csv(chunks_csv)
        filing_count = len(chunk_frame.drop_duplicates("accession_number"))
        summary = read_json(summary_path)
        current = (
            isinstance(summary, dict)
            and summary.get("chunks_sha256") == current_hash
            and int(summary.get("total_chunks", -1)) == len(chunk_frame)
            and len(pd.read_csv(inventory_path)) == filing_count
            and len(pd.read_csv(cleaning_path)) == filing_count
        )
    except (FileNotFoundError, OSError, TypeError, ValueError):
        current = False
    if not current:
        refresh_dataset_summary(ticker, project_root)
        print(f"[{ticker}] Refreshed stale dataset summary from current artifacts.")


def ensure_datasets(
    tickers: list[str] | tuple[str, ...],
    project_root: Path,
    num_8k: int,
    build_if_needed: bool = True,
) -> None:
    """Validate cached ticker datasets and transactionally rebuild incompatible ones."""

    for ticker in tickers:
        chunks_csv = _chunks_csv(ticker, project_root)
        if chunks_csv.exists() and _dataset_matches_filing_policy(chunks_csv, num_8k):
            print(f"[{ticker}] Compatible chunks CSV found - skipping download.")
            _refresh_summary_if_stale(ticker, chunks_csv, project_root)
            removed = prune_stale_dataset_artifacts(ticker, project_root)
            if removed:
                print(f"[{ticker}] Pruned {removed} stale generated artifact(s).")
            continue
        if not build_if_needed:
            raise RuntimeError(
                f"[{ticker}] --skip-build was requested, but its cached dataset is missing "
                f"or violates the filing policy (one 10-K, one 10-Q, {num_8k} 8-K, "
                f"cutoff {MAX_FILING_DATE.isoformat()})."
            )
        if chunks_csv.exists():
            print(f"[{ticker}] Cached dataset violates the filing policy - rebuilding.")
        print(f"[{ticker}] Building dataset (download -> clean -> chunk)...")
        build_dataset(ticker=ticker, num_8k=num_8k, project_root=project_root)


def ensure_indexes(
    tickers: list[str] | tuple[str, ...], retriever: str, project_root: Path
) -> None:
    """Load valid requested indexes and rebuild missing, stale, or corrupt ones."""

    retrievers = ("faiss", "bm25") if retriever == "both" else (retriever,)
    for method in retrievers:
        for ticker in tickers:
            try:
                if method == "faiss":
                    load_embeddings_index(ticker, project_root)
                else:
                    load_bm25_index(ticker, project_root)
                print(f"[{ticker}] Valid {method.upper()} index found - skipping build.")
            except (FileNotFoundError, IndexValidationError, RuntimeError, ValueError, EOFError) as exc:
                print(f"[{ticker}] Rebuilding {method.upper()} index: {exc}")
                if method == "faiss":
                    build_embeddings_index(ticker, project_root)
                else:
                    build_bm25_index(ticker, project_root)


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("must be an integer of at least 1") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def _retrieval_k_from_env() -> int:
    """Return the environment-configured retrieval depth or the built-in default."""

    raw_value = os.getenv("RAG_TOP_K")
    if raw_value is None:
        return DEFAULT_RETRIEVAL_K
    try:
        return _positive_int(raw_value)
    except argparse.ArgumentTypeError as exc:
        raise ValueError(
            "RAG_TOP_K must be an integer of at least 1; "
            f"received {raw_value!r}."
        ) from exc


def build_parser(default_retrieval_k: int | None = None) -> argparse.ArgumentParser:
    if default_retrieval_k is None:
        default_retrieval_k = _retrieval_k_from_env()
    parser = argparse.ArgumentParser(
        description="Run the FAANG SEC filing RAG evaluation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=list(DEFAULT_TICKERS),
        metavar="TICKER",
        help="Tickers to evaluate.",
    )
    parser.add_argument(
        "--retriever",
        choices=["faiss", "bm25", "both"],
        default=DEFAULT_RETRIEVER,
        help="Retrieval method; 'both' runs both methods in one checkpointed run.",
    )
    parser.add_argument(
        "--k",
        type=_positive_int,
        default=default_retrieval_k,
        help=(
            "Number of top-ranked chunks to retrieve; overrides RAG_TOP_K from "
            "the shell or .env."
        ),
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_LLM_MODEL,
        help="Anthropic model used to generate answers.",
    )
    parser.add_argument(
        "--num-8k",
        type=_positive_int,
        default=DEFAULT_NUM_8K,
        help="Exact number of latest pre-cutoff 8-K filings per ticker.",
    )
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Fail instead of rebuilding an incompatible filing dataset.",
    )
    output_group = parser.add_mutually_exclusive_group()
    output_group.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Custom final results CSV; checkpoint, manifest, and figures are kept "
            "beside it."
        ),
    )
    output_group.add_argument(
        "--run-name",
        default=None,
        help=(
            "Stable run directory name under outputs/eval_results/runs; required "
            "to resume a default-layout run."
        ),
    )
    parser.add_argument(
        "--eval-csv",
        type=Path,
        default=None,
        help="Evaluation CSV; defaults to the complete 60-question dataset.",
    )
    parser.add_argument(
        "--include-incomplete",
        action="store_true",
        help="Run rows with blank reference answers; answer metrics remain unavailable.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue an exactly matching checkpoint and retry missing/failed rows.",
    )
    return parser


def main() -> None:
    load_project_env(PROJECT_ROOT)
    try:
        parser = build_parser()
    except ValueError as exc:
        raise SystemExit(f"Error: {exc}") from None
    args = parser.parse_args()
    try:
        if args.resume and args.output is None and args.run_name is None:
            raise ValueError("--resume requires --run-name or --output to identify the run.")
        output_csv = args.output or output_for_run(
            PROJECT_ROOT, args.run_name or new_run_name()
        )
        figures_dir = output_csv.parent / "figures"
        if args.output is None and not args.resume and output_csv.parent.exists():
            raise FileExistsError(
                f"Run directory already exists: {output_csv.parent}. Choose a new "
                "--run-name or use --resume for the exact existing run."
            )
        if not args.resume:
            checkpoint_path = output_csv.with_name(
                output_csv.name + ".checkpoint.jsonl"
            )
            manifest_path = output_csv.with_name(output_csv.name + ".manifest.json")
            conflicts = [
                path
                for path in (output_csv, checkpoint_path, manifest_path)
                if path.exists()
            ]
            if figures_dir.exists():
                conflicts.extend(path for path in figures_dir.iterdir())
            if conflicts:
                raise FileExistsError(
                    "Refusing to overwrite existing run artifacts: "
                    + ", ".join(str(path) for path in conflicts)
                    + ". Use --resume for this exact run or choose a new run name/output."
                )
        tickers = normalize_tickers(args.tickers)
        ensure_datasets(
            tickers,
            PROJECT_ROOT,
            num_8k=args.num_8k,
            build_if_needed=not args.skip_build,
        )
        preflight_eval = load_eval_set(args.eval_csv) if args.eval_csv else load_eval_set()
        preflight_eval = preflight_eval[preflight_eval["ticker"].isin(tickers)].copy()
        selected_eval = (
            preflight_eval
            if args.include_incomplete
            else filter_completed_questions(preflight_eval)
        )
        if selected_eval.empty:
            raise ValueError("No evaluation questions were selected.")
        validate_source_doc_ids(selected_eval, PROJECT_ROOT)
        ensure_indexes(tickers, args.retriever, PROJECT_ROOT)

        input_figures = (
            (
                "01_evaluation_coverage.png",
                lambda: plot_evaluation_coverage(
                    preflight_eval,
                    figures_dir / "01_evaluation_coverage.png",
                ),
            ),
            (
                "02_filing_timeline.png",
                lambda: plot_filing_timeline(
                    tickers,
                    figures_dir / "02_filing_timeline.png",
                    PROJECT_ROOT,
                ),
            ),
        )
        for filename, build_figure in input_figures:
            try:
                build_figure()
            except Exception as exc:
                print(f"Warning: {filename} was not generated: {exc}")

        run_batch_eval(
            tickers=tickers,
            retriever=args.retriever,
            k=args.k,
            llm_model=args.model,
            project_root=PROJECT_ROOT,
            eval_csv=args.eval_csv,
            include_incomplete=args.include_incomplete,
            output_csv=output_csv,
            resume=args.resume,
        )
    except (FileExistsError, FileNotFoundError, RuntimeError, ValueError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
