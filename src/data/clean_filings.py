"""Convert downloaded SEC filing HTML or text into conservative plain text."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import pandas as pd
from bs4 import BeautifulSoup

from .utils import (
    PROJECT_ROOT,
    FilingRecord,
    normalize_ticker,
    project_relative,
    read_json,
    resolve_project_path,
    ticker_slug,
    write_json,
)


HEADING_PREFIX = "## "
_HEADING_SENTINEL = "[[SEC_SECTION_HEADING]]"
_PAGE_ARTIFACT = re.compile(
    r"^(?:page\s+\d+(?:\s+of\s+\d+)?|\d+\s+of\s+\d+)$", re.IGNORECASE
)
_ITEM_HEADING = re.compile(
    r"^item\s+(?:\d+[a-z]?|[ivx]+)(?:\.\d+)?[.:]?\s+\S.*$",
    re.IGNORECASE,
)
_BLOCK_TAGS = {
    "address",
    "article",
    "aside",
    "blockquote",
    "div",
    "dl",
    "dt",
    "dd",
    "figcaption",
    "figure",
    "footer",
    "header",
    "li",
    "main",
    "nav",
    "ol",
    "p",
    "pre",
    "section",
    "table",
    "tbody",
    "tfoot",
    "thead",
    "tr",
    "ul",
}


def _normalize_inline(value: str) -> str:
    """Collapse inline whitespace while preserving visible characters."""

    return re.sub(r"\s+", " ", value.replace("\xa0", " ")).strip()


def _html_to_intermediate_text(raw_html: str) -> str:
    """Flatten HTML while inserting explicit section and block boundaries."""

    soup = BeautifulSoup(raw_html, "lxml")
    for tag in soup.find_all(["script", "style", "noscript", "template"]):
        tag.decompose()

    # Inline XBRL headers and CSS-hidden facts are machine-readable metadata,
    # not visible filing prose. Removing only explicitly hidden nodes avoids
    # leaking taxonomy identifiers into retrieval chunks.
    for tag in soup.find_all(["ix:header", "ix:hidden"]):
        tag.decompose()
    for tag in soup.find_all(True):
        if tag.parent is None:
            continue
        style = re.sub(r"\s+", "", str(tag.get("style", ""))).lower()
        explicitly_hidden = (
            tag.has_attr("hidden")
            or str(tag.get("aria-hidden", "")).lower() == "true"
            or "display:none" in style
            or "visibility:hidden" in style
        )
        if explicitly_hidden:
            tag.decompose()

    for heading in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"]):
        heading_text = _normalize_inline(heading.get_text(" ", strip=True))
        replacement = (
            f"\n{_HEADING_SENTINEL} {heading_text}\n" if heading_text else "\n"
        )
        heading.replace_with(replacement)

    for line_break in soup.find_all("br"):
        line_break.replace_with("\n")

    for cell in soup.find_all(["td", "th"]):
        cell.insert_after(" \t ")

    for block in soup.find_all(_BLOCK_TAGS):
        block.insert_before("\n")
        block.insert_after("\n")

    return soup.get_text(" ", strip=False)


def _looks_like_heading(line: str) -> bool:
    """Detect common SEC item headings without classifying long prose."""

    return len(line) <= 200 and bool(_ITEM_HEADING.fullmatch(line))


def _normalize_lines(value: str) -> str:
    """Normalize text lines and preserve section headings with markers."""

    value = (
        value.replace("\r\n", "\n")
        .replace("\r", "\n")
        .replace("\f", "\n")
        .replace("\xa0", " ")
        .replace("\u200b", "")
    )
    lines: list[str] = []
    previous_blank = True

    for raw_line in value.splitlines():
        line = _normalize_inline(raw_line)
        if not line:
            if not previous_blank:
                lines.append("")
            previous_blank = True
            continue
        if _PAGE_ARTIFACT.fullmatch(line):
            continue

        if line.startswith(_HEADING_SENTINEL):
            heading = _normalize_inline(line[len(_HEADING_SENTINEL) :])
            line = f"{HEADING_PREFIX}{heading}" if heading else ""
        elif _looks_like_heading(line):
            line = f"{HEADING_PREFIX}{line}"

        if line:
            lines.append(line)
            previous_blank = False

    while lines and not lines[-1]:
        lines.pop()
    return "\n".join(lines).strip()


def clean_raw_content(raw_content: str, raw_format: str) -> tuple[str, list[str]]:
    """Clean filing content and return conservative quality warnings."""

    normalized_format = raw_format.strip().lower()
    if normalized_format not in {"html", "text"}:
        raise ValueError(f"Unsupported raw filing format: {raw_format}")

    intermediate = (
        _html_to_intermediate_text(raw_content)
        if normalized_format == "html"
        else raw_content
    )
    cleaned = _normalize_lines(intermediate)
    warnings: list[str] = []

    if not cleaned:
        warnings.append("cleaned_text_is_empty")
    if raw_content and len(cleaned) / len(raw_content) < 0.15:
        warnings.append("large_length_reduction")
    if normalized_format == "html" and "<table" in raw_content.lower():
        warnings.append("html_tables_flattened_to_text")
    if cleaned and HEADING_PREFIX not in cleaned:
        warnings.append("no_section_headings_detected")
    if cleaned and len(cleaned.split()) < 100:
        warnings.append("very_short_cleaned_text")

    return cleaned, warnings


def clean_filings(
    ticker: str = "AMZN",
    project_root: Path = PROJECT_ROOT,
) -> list[dict[str, Any]]:
    """Clean only the raw files referenced by the current filing manifest."""

    normalized_ticker = normalize_ticker(ticker)
    slug = ticker_slug(normalized_ticker)
    raw_directory = project_root / "data" / "raw" / slug
    raw_manifest_path = raw_directory / "filing_metadata.json"
    raw_manifest = read_json(raw_manifest_path)
    if not isinstance(raw_manifest, list) or not raw_manifest:
        raise ValueError(f"Raw filing manifest is empty or invalid: {raw_manifest_path}")

    cleaned_directory = (
        project_root / "data" / "processed" / slug / "cleaned_filings"
    )
    cleaned_directory.mkdir(parents=True, exist_ok=True)

    cleaned_manifest: list[dict[str, Any]] = []
    for item in raw_manifest:
        record = FilingRecord.from_dict(item)
        if record.ticker != normalized_ticker:
            raise ValueError(
                f"Manifest ticker {record.ticker} does not match {normalized_ticker}."
            )

        raw_path = resolve_project_path(record.local_raw_path, project_root)
        if not raw_path.exists():
            raise FileNotFoundError(f"Raw filing does not exist: {raw_path}")
        raw_content = raw_path.read_text(encoding="utf-8", errors="replace")
        cleaned_text, warnings = clean_raw_content(raw_content, record.raw_format)
        if not cleaned_text:
            raise RuntimeError(
                f"Cleaning produced no text for filing {record.accession_number}."
            )

        cleaned_path = cleaned_directory / f"{raw_path.stem}.txt"
        cleaned_path.write_text(cleaned_text + "\n", encoding="utf-8")
        cleaned_manifest.append(
            {
                **record.to_dict(),
                "local_cleaned_path": project_relative(cleaned_path, project_root),
                "raw_length": len(raw_content),
                "cleaned_length": len(cleaned_text),
                "warnings": warnings,
            }
        )

    processed_directory = project_root / "data" / "processed" / slug
    cleaning_manifest_path = processed_directory / "cleaning_manifest.json"
    write_json(cleaning_manifest_path, cleaned_manifest)

    summary_directory = project_root / "outputs" / "data_summary"
    summary_directory.mkdir(parents=True, exist_ok=True)
    summary_path = summary_directory / f"{slug}_cleaning_summary.csv"
    summary_rows = [
        {
            "ticker": item["ticker"],
            "filing_type": item["filing_type"],
            "filing_date": item["filing_date"],
            "accession_number": item["accession_number"],
            "raw_length": item["raw_length"],
            "cleaned_length": item["cleaned_length"],
            "retained_ratio": round(item["cleaned_length"] / item["raw_length"], 4)
            if item["raw_length"]
            else 0.0,
            "warnings": "; ".join(item["warnings"]),
        }
        for item in cleaned_manifest
    ]
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)

    print(
        f"Cleaned {len(cleaned_manifest)} filing(s) to "
        f"{project_relative(cleaned_directory, project_root)}"
    )
    print(f"Cleaning manifest: {project_relative(cleaning_manifest_path, project_root)}")
    print(f"Cleaning summary: {project_relative(summary_path, project_root)}")
    return cleaned_manifest


def build_parser() -> argparse.ArgumentParser:
    """Create the cleaning command-line parser."""

    parser = argparse.ArgumentParser(
        description="Clean raw SEC filing HTML or text from the current manifest."
    )
    parser.add_argument("--ticker", default="AMZN", help="SEC company ticker.")
    return parser


def main() -> None:
    """Run the cleaning CLI."""

    parser = build_parser()
    args = parser.parse_args()
    try:
        clean_filings(ticker=args.ticker)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
