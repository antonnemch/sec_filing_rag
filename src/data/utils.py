"""Shared types and filesystem helpers for the filing data pipeline."""

from __future__ import annotations

import json
import os
import re
from dataclasses import MISSING, asdict, dataclass, fields
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TICKERS = ("META", "AMZN", "AAPL", "NFLX", "GOOG")
DEFAULT_MULTI_COMPANY_SLUG = "faang"
_TICKER_PATTERN = re.compile(r"^[A-Z0-9.-]+$")
_EMAIL_PATTERN = re.compile(r"\b[^@\s]+@[^@\s]+\.[^@\s]+\b")


@dataclass(frozen=True)
class FilingRecord:
    """Metadata persisted for one downloaded SEC filing."""

    company: str
    ticker: str
    cik: str
    filing_type: str
    filing_date: str
    accession_number: str
    source_url: str
    local_raw_path: str
    raw_format: str
    local_markdown_path: str = ""
    human_readable_format: str = ""
    human_readable_warning: str = ""

    def to_dict(self) -> dict[str, str]:
        """Return a JSON-serializable representation."""

        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "FilingRecord":
        """Create a record from a raw manifest entry."""

        record_values: dict[str, str] = {}
        missing: list[str] = []
        for field in fields(cls):
            if field.name in value:
                record_values[field.name] = str(value[field.name])
            elif field.default is not MISSING:
                record_values[field.name] = str(field.default)
            elif field.default_factory is not MISSING:  # type: ignore[attr-defined]
                record_values[field.name] = str(field.default_factory())  # type: ignore[misc]
            else:
                missing.append(field.name)
        if missing:
            raise ValueError(
                "Filing metadata is missing required field(s): " + ", ".join(missing)
            )
        return cls(**record_values)


def normalize_ticker(ticker: str) -> str:
    """Normalize and validate a ticker supplied on the command line."""

    normalized = ticker.strip().upper()
    if not normalized or not _TICKER_PATTERN.fullmatch(normalized):
        raise ValueError(
            "Ticker must contain only letters, numbers, periods, or hyphens."
        )
    return normalized


def ticker_slug(ticker: str) -> str:
    """Return the normalized lowercase directory name for a ticker."""

    return normalize_ticker(ticker).lower()


def normalize_tickers(tickers: Iterable[str]) -> tuple[str, ...]:
    """Normalize and validate a non-empty ticker list."""

    normalized = tuple(normalize_ticker(ticker) for ticker in tickers)
    if not normalized:
        raise ValueError("At least one ticker must be provided.")

    seen: set[str] = set()
    duplicates: list[str] = []
    for ticker in normalized:
        if ticker in seen and ticker not in duplicates:
            duplicates.append(ticker)
        seen.add(ticker)
    if duplicates:
        raise ValueError("Duplicate ticker(s) are not allowed: " + ", ".join(duplicates))

    return normalized


def dataset_slug_for_tickers(tickers: Iterable[str]) -> str:
    """Return a stable output prefix for one ticker or a ticker collection."""

    normalized = normalize_tickers(tickers)
    if normalized == DEFAULT_TICKERS:
        return DEFAULT_MULTI_COMPANY_SLUG
    if len(normalized) == 1:
        return ticker_slug(normalized[0])
    return "_".join(ticker_slug(ticker) for ticker in normalized)


def validate_num_8k(num_8k: int) -> None:
    """Validate the requested number of recent 8-K filings."""

    if num_8k < 1:
        raise ValueError("--num-8k must be at least 1.")


def validate_chunk_settings(chunk_size: int, chunk_overlap: int) -> None:
    """Validate word chunk size and overlap arguments."""

    if chunk_size < 1:
        raise ValueError("--chunk-size must be at least 1.")
    if chunk_overlap < 0:
        raise ValueError("--chunk-overlap cannot be negative.")
    if chunk_overlap >= chunk_size:
        raise ValueError("--chunk-overlap must be smaller than --chunk-size.")


def configure_sec_identity(project_root: Path = PROJECT_ROOT) -> None:
    """Load and register the SEC identity without logging its value."""

    try:
        from dotenv import load_dotenv
    except ImportError as exc:  # pragma: no cover - dependency setup failure
        raise RuntimeError(
            "python-dotenv is not installed. Run: pip install -r requirements.txt"
        ) from exc

    load_dotenv(project_root / ".env", override=False)
    identity = os.getenv("SEC_IDENTITY") or os.getenv("EDGAR_IDENTITY")
    if not identity or not identity.strip():
        raise RuntimeError(
            "SEC identity is not configured. Copy .env.example to .env and set "
            'SEC_IDENTITY="Your Name your.email@example.com" before downloading.'
        )
    if (
        not _EMAIL_PATTERN.search(identity)
        or "your.email@example.com" in identity.lower()
    ):
        raise RuntimeError(
            "SEC identity must replace the .env.example placeholder and include "
            "a real contact email address."
        )

    default_cache = project_root / "data" / "raw" / ".edgar_cache"
    os.environ.setdefault("EDGAR_LOCAL_DATA_DIR", str(default_cache.resolve()))

    try:
        from edgar import set_identity
    except ImportError as exc:  # pragma: no cover - dependency setup failure
        raise RuntimeError(
            "edgartools is not installed. Run: pip install -r requirements.txt"
        ) from exc

    set_identity(identity.strip())


def project_relative(path: Path, project_root: Path = PROJECT_ROOT) -> str:
    """Return a portable project-relative path for a generated file."""

    return path.resolve().relative_to(project_root.resolve()).as_posix()


def resolve_project_path(
    relative_path: str, project_root: Path = PROJECT_ROOT
) -> Path:
    """Resolve a manifest path and reject paths outside the project root."""

    root = project_root.resolve()
    resolved = (root / relative_path).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            f"Manifest path points outside the project: {relative_path}"
        ) from exc
    return resolved


def write_json(path: Path, value: Any) -> None:
    """Write indented UTF-8 JSON, creating the parent directory."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def read_json(path: Path) -> Any:
    """Read UTF-8 JSON with a clear missing-file error."""

    if not path.exists():
        raise FileNotFoundError(
            f"Required manifest does not exist: {path}. Run the preceding "
            "pipeline stage first."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    """Write dictionaries as UTF-8 JSON Lines."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read dictionaries from UTF-8 JSON Lines."""

    if not path.exists():
        raise FileNotFoundError(
            f"Required JSONL file does not exist: {path}. Run the preceding "
            "pipeline stage first."
        )
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def format_cik(value: Any) -> str:
    """Format a SEC CIK as a zero-padded string when possible."""

    if value is None or str(value).strip() == "":
        return ""
    text = str(value).strip()
    try:
        return f"{int(text):010d}"
    except ValueError:
        return text


def safe_filename_component(value: str) -> str:
    """Convert metadata into a conservative filename component."""

    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip())
    return cleaned.strip("-") or "unknown"
