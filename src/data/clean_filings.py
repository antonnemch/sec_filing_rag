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
    r"^item\s+"
    r"(?P<number>(?:\d+[a-z]?|[ivx]+)(?:\s*\.\s*\d+[a-z]?)?)"
    r"\s*[.:]?\s*(?P<title>\S.*)?$",
    re.IGNORECASE,
)
_EIGHT_K_ITEM_TITLES = {
    "1.01": "Entry into a Material Definitive Agreement",
    "1.02": "Termination of a Material Definitive Agreement",
    "1.03": "Bankruptcy or Receivership",
    "1.04": "Material Cybersecurity Incidents",
    "2.01": "Completion of Acquisition or Disposition of Assets",
    "2.02": "Results of Operations and Financial Condition",
    "2.03": "Creation of a Direct Financial Obligation",
    "2.04": "Triggering Events That Accelerate or Increase a Direct Financial Obligation",
    "2.05": "Costs Associated with Exit or Disposal Activities",
    "2.06": "Material Impairments",
    "3.01": "Notice of Delisting or Failure to Satisfy a Continued Listing Rule",
    "3.02": "Unregistered Sales of Equity Securities",
    "3.03": "Material Modification to Rights of Security Holders",
    "4.01": "Changes in Registrant's Certifying Accountant",
    "4.02": "Non-Reliance on Previously Issued Financial Statements",
    "5.01": "Changes in Control of Registrant",
    "5.02": "Departure or Election of Directors or Certain Officers",
    "5.03": "Amendments to Articles of Incorporation or Bylaws",
    "5.04": "Temporary Suspension of Trading Under Registrant's Employee Benefit Plans",
    "5.05": "Amendments to the Registrant's Code of Ethics",
    "5.06": "Change in Shell Company Status",
    "5.07": "Submission of Matters to a Vote of Security Holders",
    "5.08": "Shareholder Director Nominations",
    "6.01": "ABS Informational and Computational Material",
    "6.02": "Change of Servicer or Trustee",
    "6.03": "Change in Credit Enhancement or Other External Support",
    "6.04": "Failure to Make a Required Distribution",
    "6.05": "Securities Act Updating Disclosure",
    "7.01": "Regulation FD Disclosure",
    "8.01": "Other Events",
    "9.01": "Financial Statements and Exhibits",
}
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


def _item_heading(line: str) -> tuple[str, str] | None:
    """Return a stable item key and display heading for an SEC item line."""

    match = _ITEM_HEADING.fullmatch(line)
    if not match:
        return None
    number = re.sub(r"\s+", "", match.group("number")).upper()
    item_label = line[: match.start("number")].strip()
    title = _normalize_inline(match.group("title") or "").strip(" .:")
    if not title:
        title = _EIGHT_K_ITEM_TITLES.get(number, "")
    display = f"{item_label} {number}"
    if title:
        display += f". {title}"
    return number, display


def _normalize_lines(value: str) -> str:
    """Normalize text lines and preserve section headings with markers."""

    value = (
        value.replace("\r\n", "\n")
        .replace("\r", "\n")
        .replace("\f", "\n")
        .replace("\xa0", " ")
        .replace("\u200b", "")
    )
    records: list[tuple[str, bool]] = []
    previous_blank = True

    for raw_line in value.splitlines():
        line = _normalize_inline(raw_line)
        if not line:
            if not previous_blank:
                records.append(("", False))
            previous_blank = True
            continue
        if _PAGE_ARTIFACT.fullmatch(line):
            continue

        explicit_heading = False
        if line.startswith(_HEADING_SENTINEL):
            heading = _normalize_inline(line[len(_HEADING_SENTINEL) :])
            line = heading
            explicit_heading = bool(heading)

        if line:
            records.append((line, explicit_heading))
            previous_blank = False

    item_candidates: dict[str, list[int]] = {}
    normalized_items: dict[int, str] = {}
    for index, (line, _) in enumerate(records):
        parsed = _item_heading(line)
        if parsed is None:
            continue
        key, display = parsed
        item_candidates.setdefault(key, []).append(index)
        normalized_items[index] = display

    # SEC tables of contents commonly repeat the real item headings. Keeping
    # only the final occurrence prevents navigation entries from becoming
    # retrieval sections while preserving the substantive item boundary.
    authoritative_items = {
        positions[-1] for positions in item_candidates.values() if positions
    }
    suppressed_titles: set[int] = set()
    for index in authoritative_items:
        key, _ = _item_heading(records[index][0]) or ("", "")
        expected_title = _EIGHT_K_ITEM_TITLES.get(key)
        if not expected_title:
            continue
        original_match = _ITEM_HEADING.fullmatch(records[index][0])
        if original_match and original_match.group("title"):
            continue
        for next_index in range(index + 1, len(records)):
            candidate = records[next_index][0]
            if not candidate:
                continue
            if candidate.strip(" .:").casefold() == expected_title.casefold():
                suppressed_titles.add(next_index)
            break

    lines: list[str] = []
    previous_blank = True
    for index, (line, explicit_heading) in enumerate(records):
        if index in suppressed_titles:
            continue
        if index in authoritative_items:
            line = f"{HEADING_PREFIX}{normalized_items[index]}"
        elif index in normalized_items:
            # Earlier duplicate item headings are table-of-contents navigation.
            continue
        elif explicit_heading:
            line = f"{HEADING_PREFIX}{line}"

        if not line:
            if not previous_blank:
                lines.append("")
            previous_blank = True
            continue
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
