"""Download recent company filings from SEC EDGAR with EdgarTools."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from .utils import (
    PROJECT_ROOT,
    FilingRecord,
    configure_sec_identity,
    format_cik,
    normalize_ticker,
    project_relative,
    safe_filename_component,
    ticker_slug,
    validate_num_8k,
    write_json,
)


def _normalize_latest(value: Any) -> list[Any]:
    """Normalize EdgarTools' single- and multi-filing return shapes."""

    if value is None:
        return []
    if hasattr(value, "form") and hasattr(value, "accession_no"):
        return [value]
    try:
        return list(value)
    except TypeError:
        return [value]


def _select_recent_filings(company: Any, form: str, count: int) -> list[Any]:
    """Select recent unamended filings of one exact form."""

    filings = company.get_filings(form=form, amendments=False)
    if filings is None or getattr(filings, "empty", False) or len(filings) == 0:
        return []
    return _normalize_latest(filings.latest(count))


def _extract_filing_content(filing: Any) -> tuple[str, str]:
    """Return filing content and format, preferring source HTML."""

    html = filing.html()
    if html and html.strip():
        return html, "html"

    text = filing.text()
    if text and text.strip():
        return text, "text"

    accession = getattr(filing, "accession_no", "unknown")
    raise RuntimeError(f"No usable filing content was returned for {accession}.")


def _extract_human_readable_content(
    filing: Any,
    raw_content: str,
    raw_format: str,
) -> tuple[str, str, str, str]:
    """Return human-readable content, preferring EdgarTools Markdown."""

    warnings: list[str] = []
    try:
        markdown = filing.markdown(include_page_breaks=True, start_page_number=1)
        if markdown and markdown.strip():
            return markdown.strip() + "\n", "markdown", ".md", ""
        warnings.append("markdown_empty")
    except Exception as exc:  # pragma: no cover - exercised via synthetic test
        warnings.append(f"markdown_failed:{type(exc).__name__}")

    try:
        text = filing.text()
        if text and text.strip():
            return text.strip() + "\n", "text", ".txt", "; ".join(warnings)
        warnings.append("text_empty")
    except Exception as exc:  # pragma: no cover - defensive fallback
        warnings.append(f"text_failed:{type(exc).__name__}")

    fallback = (
        raw_content.strip()
        if raw_format == "text" and raw_content.strip()
        else "Human-readable conversion failed. See the raw filing file."
    )
    return fallback + "\n", "text", ".txt", "; ".join(warnings)


def _download_one(
    filing: Any,
    ticker: str,
    raw_directory: Path,
    human_readable_directory: Path,
    project_root: Path,
) -> FilingRecord:
    """Persist one filing and return its normalized metadata."""

    content, raw_format = _extract_filing_content(filing)
    filing_date = str(filing.filing_date)
    filing_type = str(filing.form)
    accession_number = str(
        getattr(filing, "accession_number", None)
        or getattr(filing, "accession_no", "")
    )
    if not accession_number:
        raise RuntimeError(f"Filing {filing_type} {filing_date} has no accession number.")

    filename = "_".join(
        [
            safe_filename_component(filing_date),
            safe_filename_component(filing_type),
            safe_filename_component(accession_number.replace("-", "")),
        ]
    )
    suffix = ".html" if raw_format == "html" else ".txt"
    raw_path = raw_directory / f"{filename}{suffix}"
    raw_path.write_text(content, encoding="utf-8")

    human_content, human_format, human_suffix, human_warning = (
        _extract_human_readable_content(filing, content, raw_format)
    )
    human_path = human_readable_directory / f"{filename}{human_suffix}"
    human_path.write_text(human_content, encoding="utf-8")

    source_url = str(
        getattr(filing, "filing_url", None)
        or getattr(filing, "homepage_url", None)
        or ""
    )
    return FilingRecord(
        company=str(filing.company),
        ticker=ticker,
        cik=format_cik(getattr(filing, "cik", "")),
        filing_type=filing_type,
        filing_date=filing_date,
        accession_number=accession_number,
        source_url=source_url,
        local_raw_path=project_relative(raw_path, project_root),
        raw_format=raw_format,
        local_markdown_path=project_relative(human_path, project_root),
        human_readable_format=human_format,
        human_readable_warning=human_warning,
    )


def _write_inventory(
    records: Iterable[FilingRecord], ticker: str, project_root: Path
) -> Path:
    """Write the lightweight filing inventory allowed in source control."""

    summary_directory = project_root / "outputs" / "data_summary"
    summary_directory.mkdir(parents=True, exist_ok=True)
    inventory_path = summary_directory / f"{ticker_slug(ticker)}_filing_inventory.csv"
    columns = [
        "company",
        "ticker",
        "cik",
        "filing_type",
        "filing_date",
        "accession_number",
        "source_url",
        "local_raw_path",
        "raw_format",
        "local_markdown_path",
        "human_readable_format",
        "human_readable_warning",
    ]
    pd.DataFrame([record.to_dict() for record in records], columns=columns).to_csv(
        inventory_path, index=False
    )
    return inventory_path


def download_filings(
    ticker: str = "AMZN",
    num_8k: int = 5,
    project_root: Path = PROJECT_ROOT,
) -> list[FilingRecord]:
    """Download the latest 10-K, 10-Q, and requested recent 8-K filings."""

    normalized_ticker = normalize_ticker(ticker)
    validate_num_8k(num_8k)
    configure_sec_identity(project_root)

    try:
        from edgar import Company
    except ImportError as exc:  # pragma: no cover - dependency setup failure
        raise RuntimeError(
            "edgartools is not installed. Run: pip install -r requirements.txt"
        ) from exc

    try:
        company = Company(normalized_ticker)
    except Exception as exc:
        raise RuntimeError(
            f"Unable to load SEC company metadata for {normalized_ticker}: {exc}"
        ) from exc

    selected: list[Any] = []
    for form, count in (("10-K", 1), ("10-Q", 1), ("8-K", num_8k)):
        try:
            recent = _select_recent_filings(company, form, count)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to retrieve recent {form} filings for "
                f"{normalized_ticker}: {exc}"
            ) from exc
        if not recent:
            raise RuntimeError(
                f"No unamended {form} filings were found for {normalized_ticker}."
            )
        if form == "8-K" and len(recent) < num_8k:
            print(
                f"Warning: requested {num_8k} recent 8-K filings, but only "
                f"{len(recent)} were available."
            )
        selected.extend(recent)

    raw_directory = (
        project_root / "data" / "raw" / ticker_slug(normalized_ticker)
    )
    raw_directory.mkdir(parents=True, exist_ok=True)
    human_readable_directory = (
        project_root / "data" / "human_readable" / ticker_slug(normalized_ticker)
    )
    human_readable_directory.mkdir(parents=True, exist_ok=True)

    records: list[FilingRecord] = []
    for filing in selected:
        try:
            records.append(
                _download_one(
                    filing,
                    normalized_ticker,
                    raw_directory,
                    human_readable_directory,
                    project_root,
                )
            )
        except Exception as exc:
            accession = getattr(filing, "accession_no", "unknown")
            raise RuntimeError(
                f"Failed to save filing {accession} for {normalized_ticker}: {exc}"
            ) from exc
    manifest_path = raw_directory / "filing_metadata.json"
    write_json(manifest_path, [record.to_dict() for record in records])
    inventory_path = _write_inventory(records, normalized_ticker, project_root)

    print(
        f"Downloaded {len(records)} filing(s) to "
        f"{project_relative(raw_directory, project_root)}"
    )
    print(f"Metadata: {project_relative(manifest_path, project_root)}")
    print(f"Inventory: {project_relative(inventory_path, project_root)}")
    return records


def build_parser() -> argparse.ArgumentParser:
    """Create the downloader command-line parser."""

    parser = argparse.ArgumentParser(
        description="Download recent SEC filings for a company."
    )
    parser.add_argument("--ticker", default="AMZN", help="SEC company ticker.")
    parser.add_argument(
        "--num-8k",
        type=int,
        default=5,
        help="Number of recent unamended 8-K filings to download.",
    )
    return parser


def main() -> None:
    """Run the downloader CLI."""

    parser = build_parser()
    args = parser.parse_args()
    try:
        download_filings(ticker=args.ticker, num_8k=args.num_8k)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
