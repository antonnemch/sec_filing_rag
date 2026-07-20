"""Static charts for communicating SEC RAG pipeline inputs and results."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from src.config import MAX_FILING_DATE
from src.data.utils import PROJECT_ROOT, normalize_tickers, ticker_slug


def _pyplot():
    cache_dir = PROJECT_ROOT / ".cache" / "matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    _pyplot().close(fig)
    print(f"Chart saved: {path}")
    return path


def plot_evaluation_coverage(eval_df: pd.DataFrame, path: Path) -> Path:
    """Plot completion status for every ticker/question pair."""
    plt = _pyplot()
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    tickers = list(dict.fromkeys(eval_df["ticker"].astype(str).str.upper()))
    question_ids = sorted(pd.to_numeric(eval_df["question_id"]).astype(int).unique())
    matrix = np.zeros((len(tickers), len(question_ids)), dtype=int)
    ticker_pos = {ticker: i for i, ticker in enumerate(tickers)}
    question_pos = {question_id: i for i, question_id in enumerate(question_ids)}

    for row in eval_df.itertuples(index=False):
        answer = "" if pd.isna(row.answer) else str(row.answer).strip()
        source_doc_id = str(row.source_doc_id).strip()
        state = 2 if source_doc_id.endswith("_MULTI") else (1 if answer else 0)
        matrix[ticker_pos[str(row.ticker).upper()], question_pos[int(row.question_id)]] = state

    fig, ax = plt.subplots(figsize=(11, 3.7))
    colors = ["#d1d5db", "#2f6f9f", "#d58b2a"]
    ax.imshow(matrix, aspect="auto", cmap=ListedColormap(colors), vmin=0, vmax=2)
    ax.set_xticks(range(len(question_ids)), [f"Q{value}" for value in question_ids])
    ax.set_yticks(range(len(tickers)), tickers)
    ax.set_xlabel("Question")
    ax.set_title("Evaluation-set coverage")
    ax.set_xticks(np.arange(-0.5, len(question_ids), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(tickers), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.legend(
        handles=[
            Patch(facecolor=colors[1], label="Completed"),
            Patch(facecolor=colors[0], label="Incomplete"),
            Patch(facecolor=colors[2], label="Multi-document"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.2),
        ncol=3,
        frameon=False,
    )
    return _save(fig, path)


def plot_filing_timeline(
    tickers: Iterable[str],
    path: Path,
    project_root: Path = PROJECT_ROOT,
) -> Path:
    """Plot the distinct filing documents represented in ticker chunk CSVs."""
    plt = _pyplot()
    normalized = normalize_tickers(tickers)
    frames: list[pd.DataFrame] = []
    for ticker in normalized:
        slug = ticker_slug(ticker)
        chunks_path = project_root / "data" / "processed" / slug / f"{slug}_filing_chunks.csv"
        frame = pd.read_csv(
            chunks_path,
            usecols=["ticker", "filing_type", "filing_date", "accession_number"],
        ).drop_duplicates("accession_number")
        frames.append(frame)
    documents = pd.concat(frames, ignore_index=True)
    documents["filing_date"] = pd.to_datetime(documents["filing_date"])

    fig, ax = plt.subplots(figsize=(11, 4.5))
    markers = {"10-K": "s", "10-Q": "o", "8-K": "D"}
    y_positions = {ticker: i for i, ticker in enumerate(normalized)}
    for filing_type, marker in markers.items():
        subset = documents[documents["filing_type"] == filing_type]
        ax.scatter(
            subset["filing_date"],
            [y_positions[str(ticker).upper()] for ticker in subset["ticker"]],
            marker=marker,
            s=75,
            label=filing_type,
        )
    ax.axvline(pd.Timestamp(MAX_FILING_DATE), color="#a94442", linestyle="--", linewidth=1.5)
    ax.text(
        pd.Timestamp(MAX_FILING_DATE),
        1.01,
        f"Cutoff {MAX_FILING_DATE.isoformat()}",
        transform=ax.get_xaxis_transform(),
        ha="right",
        va="bottom",
        color="#7f1d1d",
    )
    ax.set_yticks(range(len(normalized)), normalized)
    ax.set_xlabel("Filing date")
    ax.set_title("Frozen SEC filing selection")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(frameon=False, ncol=3, loc="upper left")
    return _save(fig, path)


def plot_retriever_comparison(scored: pd.DataFrame, path: Path) -> Path | None:
    """Plot mean available metrics for each retriever in small multiples."""
    plt = _pyplot()
    candidates = [
        ("retrieval_success", "Retrieval success rate"),
        ("generation_success", "Generation success rate"),
        ("document_recall_at_k", "Document Recall@k"),
        ("document_reciprocal_rank", "Document reciprocal rank"),
        ("chunk_recall_at_k", "Chunk Recall@k"),
        ("chunk_reciprocal_rank", "Chunk reciprocal rank"),
        ("cosine_sim", "Cosine similarity"),
        ("word_overlap_f1", "Word-overlap F1"),
        ("llm_score", "LLM judge score"),
    ]
    metrics = [(column, label) for column, label in candidates if column in scored and scored[column].notna().any()]
    if not metrics or scored.empty:
        return None
    retrievers = sorted(scored["retriever"].dropna().unique())
    columns = 2
    rows = int(np.ceil(len(metrics) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(10, 3.2 * rows), squeeze=False)
    retriever_colors = {"faiss": "#2f6f9f", "bm25": "#d58b2a"}
    for ax, (column, label) in zip(axes.flat, metrics):
        values = scored.groupby("retriever")[column].mean().reindex(retrievers)
        bars = ax.bar(
            retrievers,
            values,
            color=[retriever_colors.get(name, "#6b7280") for name in retrievers],
        )
        upper = 5 if column == "llm_score" else 1
        ax.set_ylim(0, upper)
        ax.set_title(label)
        ax.grid(axis="y", alpha=0.2)
        for bar, value in zip(bars, values):
            if pd.isna(value):
                continue
            ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.3f}", ha="center", va="bottom")
    for ax in axes.flat[len(metrics) :]:
        ax.remove()
    fig.suptitle("Retriever performance summary")
    fig.tight_layout()
    return _save(fig, path)


def plot_per_question_delta(scored: pd.DataFrame, path: Path) -> Path | None:
    """Plot FAISS minus BM25 for the best available answer-quality metric."""
    plt = _pyplot()
    if not {"faiss", "bm25"}.issubset(set(scored.get("retriever", pd.Series(dtype=str)))):
        return None
    candidates = [
        ("cosine_sim", "Cosine similarity"),
        ("document_reciprocal_rank", "Document reciprocal rank"),
        ("word_overlap_f1", "Word-overlap F1"),
    ]
    selected = next(
        ((column, label) for column, label in candidates if column in scored and scored[column].notna().any()),
        None,
    )
    if selected is None:
        return None
    column, label = selected
    wide = scored.pivot_table(index="qa_id", columns="retriever", values=column)
    if not {"faiss", "bm25"}.issubset(wide.columns):
        return None
    wide = wide.dropna(subset=["faiss", "bm25"])
    if wide.empty:
        return None
    delta = (wide["faiss"] - wide["bm25"]).sort_values()
    fig, ax = plt.subplots(figsize=(10, max(4.5, 0.32 * len(delta))))
    colors = ["#d58b2a" if value < 0 else "#2f6f9f" for value in delta]
    ax.barh(delta.index, delta.values, color=colors)
    ax.axvline(0, color="#4b5563", linewidth=1)
    ax.set_xlabel(f"FAISS minus BM25 ({label})")
    ax.set_title("Per-question retriever difference")
    ax.grid(axis="x", alpha=0.2)
    return _save(fig, path)
