"""Static figures for SEC RAG inputs, coverage, and evaluation performance."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from src.config import CATEGORY_LABELS, MAX_FILING_DATE
from src.data.utils import PROJECT_ROOT, normalize_tickers, ticker_slug


METRIC_SPECS = {
    "retrieval_success": ("Retrieval success rate", 0.0, 1.0),
    "generation_success": ("Generation success rate", 0.0, 1.0),
    "document_recall_at_k": ("Document Recall@k", 0.0, 1.0),
    "document_reciprocal_rank": ("Document reciprocal rank", 0.0, 1.0),
    "chunk_recall_at_k": ("Chunk Recall@k", 0.0, 1.0),
    "chunk_reciprocal_rank": ("Chunk reciprocal rank", 0.0, 1.0),
    "cosine_sim": ("Cosine similarity", -1.0, 1.0),
    "word_overlap_f1": ("Word-overlap F1", 0.0, 1.0),
    "llm_score": ("LLM judge score", 1.0, 5.0),
}
_RETRIEVER_COLORS = {"faiss": "#2f6f9f", "bm25": "#d58b2a"}
_STATUS_COLORS = {
    "ok": "#3a7d44",
    "generation_error": "#d58b2a",
    "retrieval_error": "#a94442",
}


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
    print(f"Figure saved: {path}")
    return path


def _available_metrics(scored: pd.DataFrame) -> list[str]:
    return [
        metric
        for metric in METRIC_SPECS
        if metric in scored.columns and scored[metric].notna().any()
    ]


def _present_metrics(scored: pd.DataFrame) -> list[str]:
    return [metric for metric in METRIC_SPECS if metric in scored.columns]


def _subplot_grid(count: int, columns: int = 2):
    plt = _pyplot()
    rows = int(np.ceil(count / columns))
    return plt.subplots(
        rows,
        columns,
        figsize=(11.5, 3.35 * rows),
        squeeze=False,
    )


def _finish_grid(fig, axes, used: int, title: str) -> None:
    for axis in axes.flat[used:]:
        axis.remove()
    fig.suptitle(title)
    fig.tight_layout()


def _retrievers(scored: pd.DataFrame) -> list[str]:
    preferred = ["faiss", "bm25"]
    present = set(scored["retriever"].dropna().astype(str))
    return [name for name in preferred if name in present] + sorted(
        present.difference(preferred)
    )


def _metric_axis(axis, metric: str) -> None:
    label, lower, upper = METRIC_SPECS[metric]
    margin = 0.08 * (upper - lower)
    axis.set_ylim(lower - margin if lower < 0 else lower, upper + margin)
    axis.set_title(label)
    axis.grid(axis="y", alpha=0.2)


def plot_evaluation_coverage(eval_df: pd.DataFrame, path: Path) -> Path:
    """Plot completion and multi-document status for each ticker/question pair."""

    plt = _pyplot()
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    tickers = list(dict.fromkeys(eval_df["ticker"].astype(str).str.upper()))
    question_ids = sorted(pd.to_numeric(eval_df["question_id"]).astype(int).unique())
    matrix = np.zeros((len(tickers), len(question_ids)), dtype=int)
    ticker_pos = {ticker: index for index, ticker in enumerate(tickers)}
    question_pos = {question_id: index for index, question_id in enumerate(question_ids)}
    for row in eval_df.itertuples(index=False):
        answer = "" if pd.isna(row.answer) else str(row.answer).strip()
        source_doc_id = str(row.source_doc_id).strip()
        state = 2 if source_doc_id.endswith("_MULTI") else (1 if answer else 0)
        matrix[ticker_pos[str(row.ticker).upper()], question_pos[int(row.question_id)]] = state

    fig, axis = plt.subplots(figsize=(11, 3.7))
    colors = ["#d1d5db", "#2f6f9f", "#d58b2a"]
    axis.imshow(matrix, aspect="auto", cmap=ListedColormap(colors), vmin=0, vmax=2)
    axis.set_xticks(range(len(question_ids)), [f"Q{value}" for value in question_ids])
    axis.set_yticks(range(len(tickers)), tickers)
    axis.set_xlabel("Question")
    axis.set_title("Evaluation-set coverage")
    axis.set_xticks(np.arange(-0.5, len(question_ids), 1), minor=True)
    axis.set_yticks(np.arange(-0.5, len(tickers), 1), minor=True)
    axis.grid(which="minor", color="white", linewidth=1.5)
    axis.tick_params(which="minor", bottom=False, left=False)
    axis.legend(
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
    """Plot distinct filing documents represented in ticker chunk CSVs."""

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

    fig, axis = plt.subplots(figsize=(11, 4.5))
    markers = {"10-K": "s", "10-Q": "o", "8-K": "D"}
    y_positions = {ticker: index for index, ticker in enumerate(normalized)}
    for filing_type, marker in markers.items():
        subset = documents[documents["filing_type"] == filing_type]
        axis.scatter(
            subset["filing_date"],
            [y_positions[str(ticker).upper()] for ticker in subset["ticker"]],
            marker=marker,
            s=75,
            label=filing_type,
        )
    axis.axvline(
        pd.Timestamp(MAX_FILING_DATE), color="#a94442", linestyle="--", linewidth=1.5
    )
    axis.text(
        pd.Timestamp(MAX_FILING_DATE),
        1.01,
        f"Cutoff {MAX_FILING_DATE.isoformat()}",
        transform=axis.get_xaxis_transform(),
        ha="right",
        va="bottom",
        color="#7f1d1d",
    )
    axis.set_yticks(range(len(normalized)), normalized)
    axis.set_xlabel("Filing date")
    axis.set_title("Frozen SEC filing selection")
    axis.grid(axis="x", alpha=0.25)
    axis.legend(frameon=False, ncol=3, loc="upper left")
    return _save(fig, path)


def plot_retriever_comparison(scored: pd.DataFrame, path: Path) -> Path | None:
    """Plot overall mean available metrics for each retriever."""

    metrics = _available_metrics(scored)
    if not metrics or scored.empty:
        return None
    retrievers = _retrievers(scored)
    fig, axes = _subplot_grid(len(metrics))
    for axis, metric in zip(axes.flat, metrics):
        values = scored.groupby("retriever")[metric].mean().reindex(retrievers)
        bars = axis.bar(
            retrievers,
            values,
            color=[_RETRIEVER_COLORS.get(name, "#6b7280") for name in retrievers],
        )
        _metric_axis(axis, metric)
        for bar, value in zip(bars, values):
            if not pd.isna(value):
                axis.text(
                    bar.get_x() + bar.get_width() / 2,
                    value,
                    f"{value:.3f}",
                    ha="center",
                    va="bottom",
                )
    _finish_grid(fig, axes, len(metrics), "Overall retriever performance")
    return _save(fig, path)


def _plot_group_performance(
    scored: pd.DataFrame,
    group_column: str,
    group_order: list[str],
    path: Path,
    title: str,
) -> Path | None:
    metrics = _available_metrics(scored)
    if not metrics or scored.empty:
        return None
    retrievers = _retrievers(scored)
    fig, axes = _subplot_grid(len(metrics))
    x = np.arange(len(group_order))
    width = 0.8 / max(len(retrievers), 1)
    for axis, metric in zip(axes.flat, metrics):
        grouped = scored.groupby([group_column, "retriever"])[metric].mean()
        for index, retriever in enumerate(retrievers):
            values = [grouped.get((group, retriever), np.nan) for group in group_order]
            axis.bar(
                x + (index - (len(retrievers) - 1) / 2) * width,
                values,
                width,
                label=retriever.upper(),
                color=_RETRIEVER_COLORS.get(retriever, "#6b7280"),
            )
        axis.set_xticks(x, group_order, rotation=20, ha="right")
        _metric_axis(axis, metric)
    if metrics:
        axes.flat[0].legend(frameon=False, ncol=max(len(retrievers), 1))
    _finish_grid(fig, axes, len(metrics), title)
    return _save(fig, path)


def plot_category_performance(scored: pd.DataFrame, path: Path) -> Path | None:
    """Plot every available metric by authoritative question category and retriever."""

    frame = scored.copy()
    frame["category_label"] = frame["category"].map(CATEGORY_LABELS)
    order = [CATEGORY_LABELS[key] for key in sorted(CATEGORY_LABELS)]
    return _plot_group_performance(
        frame,
        "category_label",
        order,
        path,
        "Performance by question category",
    )


def plot_ticker_performance(scored: pd.DataFrame, path: Path) -> Path | None:
    """Plot every available metric by ticker and retriever."""

    order = sorted(scored["ticker"].dropna().astype(str).unique())
    return _plot_group_performance(
        scored,
        "ticker",
        order,
        path,
        "Performance by ticker",
    )


def plot_answerability_performance(scored: pd.DataFrame, path: Path) -> Path | None:
    """Plot every available metric by answerability label and retriever."""

    frame = scored.copy()
    frame["answerability_label"] = frame["answerable"].map(
        {True: "Answerable", False: "Unanswerable"}
    )
    order = [
        label
        for label in ("Answerable", "Unanswerable")
        if label in set(frame["answerability_label"].dropna())
    ]
    if not order:
        return None
    return _plot_group_performance(
        frame,
        "answerability_label",
        order,
        path,
        "Performance by answerability",
    )


def plot_metric_distributions(scored: pd.DataFrame, path: Path) -> Path | None:
    """Plot row-level score distributions for each available non-binary metric."""

    metrics = [
        metric
        for metric in _available_metrics(scored)
        if metric not in {"retrieval_success", "generation_success"}
    ]
    if not metrics:
        return None
    retrievers = _retrievers(scored)
    fig, axes = _subplot_grid(len(metrics))
    for axis, metric in zip(axes.flat, metrics):
        samples = [
            scored.loc[scored["retriever"] == retriever, metric].dropna().to_numpy()
            for retriever in retrievers
        ]
        nonempty = [(retriever, values) for retriever, values in zip(retrievers, samples) if len(values)]
        if not nonempty:
            axis.remove()
            continue
        labels, values = zip(*nonempty)
        boxes = axis.boxplot(
            values,
            tick_labels=[label.upper() for label in labels],
            patch_artist=True,
        )
        for box, label in zip(boxes["boxes"], labels):
            box.set_facecolor(_RETRIEVER_COLORS.get(label, "#6b7280"))
            box.set_alpha(0.65)
        _metric_axis(axis, metric)
    _finish_grid(fig, axes, len(metrics), "Metric distributions by retriever")
    return _save(fig, path)


def _paired_deltas(scored: pd.DataFrame, metric: str) -> pd.Series:
    wide = scored.pivot_table(index="qa_id", columns="retriever", values=metric)
    if not {"faiss", "bm25"}.issubset(wide.columns):
        return pd.Series(dtype=float)
    return (wide["faiss"] - wide["bm25"]).dropna()


def plot_per_question_delta(scored: pd.DataFrame, path: Path) -> Path | None:
    """Plot paired FAISS-minus-BM25 deltas for every available metric."""

    metrics = [metric for metric in _available_metrics(scored) if not _paired_deltas(scored, metric).empty]
    if not metrics:
        return None
    fig, axes = _subplot_grid(len(metrics))
    for axis, metric in zip(axes.flat, metrics):
        delta = _paired_deltas(scored, metric).sort_values()
        y = np.arange(len(delta))
        colors = np.where(delta.to_numpy() >= 0, _RETRIEVER_COLORS["faiss"], _RETRIEVER_COLORS["bm25"])
        axis.scatter(delta, y, c=colors, s=18, alpha=0.75)
        axis.axvline(0, color="#4b5563", linewidth=1)
        axis.axvline(float(delta.median()), color="#111827", linestyle="--", linewidth=1)
        axis.set_yticks([])
        axis.set_xlabel("FAISS minus BM25")
        axis.set_title(f"{METRIC_SPECS[metric][0]} (n={len(delta)})")
        axis.grid(axis="x", alpha=0.2)
        outliers = list(delta.head(2).items()) + list(delta.tail(2).items())
        for qa_id, value in dict(outliers).items():
            position = int(delta.index.get_loc(qa_id))
            axis.annotate(
                str(qa_id),
                (value, position),
                xytext=(4 if value >= 0 else -4, 0),
                textcoords="offset points",
                ha="left" if value >= 0 else "right",
                va="center",
                fontsize=8,
            )
    _finish_grid(fig, axes, len(metrics), "Paired per-question retriever differences")
    return _save(fig, path)


def plot_category_delta_heatmap(scored: pd.DataFrame, path: Path) -> Path | None:
    """Plot mean paired retriever deltas by category and metric."""

    metrics = _available_metrics(scored)
    categories = [CATEGORY_LABELS[key] for key in sorted(CATEGORY_LABELS)]
    category_by_qa = scored.drop_duplicates("qa_id").set_index("qa_id")["category"]
    raw = np.full((len(categories), len(metrics)), np.nan)
    normalized = np.full_like(raw, np.nan)
    for metric_index, metric in enumerate(metrics):
        delta = _paired_deltas(scored, metric)
        if delta.empty:
            continue
        labels = category_by_qa.reindex(delta.index).map(CATEGORY_LABELS)
        means = delta.groupby(labels).mean()
        span = METRIC_SPECS[metric][2] - METRIC_SPECS[metric][1]
        for category_index, category in enumerate(categories):
            value = means.get(category, np.nan)
            raw[category_index, metric_index] = value
            normalized[category_index, metric_index] = value / span if span else np.nan
    if np.isnan(raw).all():
        return None

    plt = _pyplot()
    fig, axis = plt.subplots(figsize=(max(10, 1.45 * len(metrics)), 4.8))
    image = axis.imshow(normalized, cmap="RdBu", vmin=-1, vmax=1, aspect="auto")
    axis.set_xticks(range(len(metrics)), [METRIC_SPECS[metric][0] for metric in metrics], rotation=30, ha="right")
    axis.set_yticks(range(len(categories)), categories)
    axis.set_title("Mean paired delta by category (FAISS minus BM25)")
    for row in range(raw.shape[0]):
        for column in range(raw.shape[1]):
            if not np.isnan(raw[row, column]):
                axis.text(column, row, f"{raw[row, column]:+.3f}", ha="center", va="center", fontsize=8)
    colorbar = fig.colorbar(image, ax=axis, fraction=0.035, pad=0.02)
    colorbar.set_label("Delta normalized to metric range")
    fig.tight_layout()
    return _save(fig, path)


def plot_metric_availability_and_outcomes(
    scored: pd.DataFrame, path: Path
) -> Path | None:
    """Plot metric eligibility denominators and pipeline outcome counts."""

    metrics = _present_metrics(scored)
    if scored.empty or not metrics:
        return None
    plt = _pyplot()
    retrievers = _retrievers(scored)
    fig, axes = plt.subplots(1, 2, figsize=(14, max(5, 0.45 * len(metrics))))

    availability = pd.DataFrame(
        {
            retriever: [
                scored.loc[scored["retriever"] == retriever, metric].notna().mean()
                for metric in metrics
            ]
            for retriever in retrievers
        },
        index=[METRIC_SPECS[metric][0] for metric in metrics],
    )
    y = np.arange(len(metrics))
    height = 0.8 / max(len(retrievers), 1)
    for index, retriever in enumerate(retrievers):
        axes[0].barh(
            y + (index - (len(retrievers) - 1) / 2) * height,
            availability[retriever],
            height,
            label=retriever.upper(),
            color=_RETRIEVER_COLORS.get(retriever, "#6b7280"),
        )
    axes[0].set_yticks(y, availability.index)
    axes[0].set_xlim(0, 1)
    axes[0].set_xlabel("Eligible fraction of rows")
    axes[0].set_title("Metric availability")
    axes[0].grid(axis="x", alpha=0.2)
    axes[0].legend(frameon=False)

    statuses = ["ok", "generation_error", "retrieval_error"]
    bottom = np.zeros(len(retrievers))
    for status in statuses:
        counts = np.array(
            [
                int(((scored["retriever"] == retriever) & (scored["status"] == status)).sum())
                for retriever in retrievers
            ]
        )
        axes[1].bar(
            [retriever.upper() for retriever in retrievers],
            counts,
            bottom=bottom,
            label=status.replace("_", " ").title(),
            color=_STATUS_COLORS[status],
        )
        bottom += counts
    axes[1].set_ylabel("Question count")
    axes[1].set_title("Pipeline outcomes")
    axes[1].grid(axis="y", alpha=0.2)
    axes[1].legend(frameon=False)
    fig.suptitle("Evaluation coverage and reliability")
    fig.tight_layout()
    return _save(fig, path)


def plot_metric_correlations(scored: pd.DataFrame, path: Path) -> Path | None:
    """Plot within-retriever correlations among non-binary performance metrics."""

    metrics = [
        metric
        for metric in _available_metrics(scored)
        if metric not in {"retrieval_success", "generation_success"}
    ]
    if len(metrics) < 2:
        return None
    retrievers = _retrievers(scored)
    valid = [
        retriever
        for retriever in retrievers
        if scored.loc[scored["retriever"] == retriever, metrics].corr().notna().sum().sum()
    ]
    if not valid:
        return None
    plt = _pyplot()
    fig, axes = plt.subplots(1, len(valid), figsize=(7 * len(valid), 6), squeeze=False)
    labels = [METRIC_SPECS[metric][0] for metric in metrics]
    for axis, retriever in zip(axes.flat, valid):
        correlation = scored.loc[scored["retriever"] == retriever, metrics].corr()
        image = axis.imshow(correlation, cmap="RdBu", vmin=-1, vmax=1)
        axis.set_xticks(range(len(metrics)), labels, rotation=40, ha="right")
        axis.set_yticks(range(len(metrics)), labels)
        axis.set_title(retriever.upper())
        for row in range(len(metrics)):
            for column in range(len(metrics)):
                value = correlation.iloc[row, column]
                if not pd.isna(value):
                    axis.text(column, row, f"{value:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=axes.ravel().tolist(), fraction=0.025, pad=0.03, label="Pearson correlation")
    fig.suptitle("Metric agreement within each retriever")
    fig.subplots_adjust(bottom=0.3, top=0.86, wspace=0.45)
    return _save(fig, path)


def generate_evaluation_figures(scored: pd.DataFrame, figures_dir: Path) -> list[Path]:
    """Generate the complete post-scoring figure suite and return created paths."""

    if scored.empty:
        return []

    builders = [
        (plot_retriever_comparison, "03_overall_retriever_metrics.png"),
        (plot_metric_availability_and_outcomes, "04_metric_availability_and_outcomes.png"),
        (plot_category_performance, "05_category_performance.png"),
        (plot_category_delta_heatmap, "06_category_retriever_deltas.png"),
        (plot_ticker_performance, "07_ticker_performance.png"),
        (plot_answerability_performance, "08_answerability_performance.png"),
        (plot_metric_distributions, "09_metric_distributions.png"),
        (plot_per_question_delta, "10_per_question_deltas.png"),
        (plot_metric_correlations, "11_metric_correlations.png"),
    ]
    created: list[Path] = []
    for builder, filename in builders:
        target = figures_dir / filename
        try:
            path = builder(scored, target)
            if path is not None:
                created.append(path)
            elif target.exists():
                target.unlink()
        except Exception as exc:
            if target.exists():
                target.unlink()
            print(f"Warning: {filename} was not generated: {exc}")
    return created
