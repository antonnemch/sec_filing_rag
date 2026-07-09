"""Offline tests for filing cleaning, chunking, and metadata handling."""

from __future__ import annotations

import csv
import json
import os
import shutil
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

from src.data.build_dataset import combine_chunk_outputs, resolve_requested_tickers
from src.data.chunk_filings import chunk_cleaned_text, chunk_filings
from src.data.clean_filings import clean_filings, clean_raw_content
from src.data.describe_dataset import describe_dataset
from src.data.download_filings import _extract_filing_content
from src.data.download_filings import _extract_human_readable_content
from src.data.utils import (
    DEFAULT_TICKERS,
    FilingRecord,
    configure_sec_identity,
    write_json,
    write_jsonl,
)


class FakeFiling:
    """Minimal filing object for testing HTML-to-text fallback behavior."""

    accession_no = "0000000000-26-000001"

    def __init__(
        self,
        html: str | None,
        text: str,
        markdown: str | None = None,
        markdown_error: Exception | None = None,
    ) -> None:
        self._html = html
        self._text = text
        self._markdown = markdown
        self._markdown_error = markdown_error

    def html(self) -> str | None:
        return self._html

    def text(self) -> str:
        return self._text

    def markdown(
        self,
        include_page_breaks: bool = False,
        start_page_number: int = 0,
    ) -> str | None:
        if self._markdown_error:
            raise self._markdown_error
        if include_page_breaks and start_page_number == 1:
            return self._markdown
        return self._markdown


class CleaningTests(unittest.TestCase):
    def test_html_cleaning_preserves_headings_and_removes_artifacts(self) -> None:
        raw = """
        <html>
          <head><style>.hidden { display: none; }</style></head>
          <body>
            <div style="display: none">
              <ix:header><ix:hidden>taxonomy_noise</ix:hidden></ix:header>
            </div>
            <h1>Item 1. Business</h1>
            <p>Amazon operates several customer-focused businesses.</p>
            <div>Page 1 of 12</div>
            <script>do_not_keep_this()</script>
            <span style="visibility: hidden">hidden_fact</span>
            <p>123</p>
            <table><tr><th>Year</th><th>Revenue</th></tr>
              <tr><td>2025</td><td>100</td></tr></table>
          </body>
        </html>
        """

        cleaned, warnings = clean_raw_content(raw, "html")

        self.assertIn("## Item 1. Business", cleaned)
        self.assertIn("Amazon operates several customer-focused businesses.", cleaned)
        self.assertIn("123", cleaned)
        self.assertIn("Revenue", cleaned)
        self.assertNotIn("Page 1 of 12", cleaned)
        self.assertNotIn("do_not_keep_this", cleaned)
        self.assertNotIn("taxonomy_noise", cleaned)
        self.assertNotIn("hidden_fact", cleaned)
        self.assertNotIn("<table", cleaned)
        self.assertIn("html_tables_flattened_to_text", warnings)

    def test_item_heading_is_detected_in_plain_text(self) -> None:
        raw = "ITEM 1A. RISK FACTORS\nMaterial risks are described here."

        cleaned, _ = clean_raw_content(raw, "text")

        self.assertEqual(
            cleaned,
            "## ITEM 1A. RISK FACTORS\nMaterial risks are described here.",
        )

    def test_unsupported_raw_format_fails(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            clean_raw_content("content", "pdf")

    def test_filing_content_falls_back_to_text(self) -> None:
        content, raw_format = _extract_filing_content(
            FakeFiling(None, "Fallback filing text")
        )

        self.assertEqual(content, "Fallback filing text")
        self.assertEqual(raw_format, "text")

    def test_human_readable_prefers_markdown(self) -> None:
        content, fmt, suffix, warning = _extract_human_readable_content(
            FakeFiling("<p>Raw</p>", "Plain text", "# Markdown filing"),
            "<p>Raw</p>",
            "html",
        )

        self.assertEqual(content, "# Markdown filing\n")
        self.assertEqual(fmt, "markdown")
        self.assertEqual(suffix, ".md")
        self.assertEqual(warning, "")

    def test_human_readable_falls_back_to_text(self) -> None:
        content, fmt, suffix, warning = _extract_human_readable_content(
            FakeFiling(
                "<p>Raw</p>",
                "Plain text filing",
                None,
                markdown_error=RuntimeError("boom"),
            ),
            "<p>Raw</p>",
            "html",
        )

        self.assertEqual(content, "Plain text filing\n")
        self.assertEqual(fmt, "text")
        self.assertEqual(suffix, ".txt")
        self.assertIn("markdown_failed", warning)

    def test_missing_sec_identity_fails_before_download(self) -> None:
        with patch.dict(
            os.environ,
            {"SEC_IDENTITY": "", "EDGAR_IDENTITY": ""},
            clear=False,
        ):
            with self.assertRaisesRegex(RuntimeError, "not configured"):
                configure_sec_identity(Path.cwd())


class ChunkingTests(unittest.TestCase):
    def test_chunks_stay_in_section_and_overlap_by_words(self) -> None:
        cleaned = (
            "## Item 1. Business\n"
            "zero one two three four five six seven eight nine"
        )

        chunks = chunk_cleaned_text(cleaned, chunk_size=4, chunk_overlap=1)

        self.assertEqual(
            chunks,
            [
                ("Item 1. Business", "zero one two three", 4),
                ("Item 1. Business", "three four five six", 4),
                ("Item 1. Business", "six seven eight nine", 4),
            ],
        )

    def test_unknown_heading_is_used_for_unlabelled_text(self) -> None:
        chunks = chunk_cleaned_text(
            "alpha beta gamma", chunk_size=2, chunk_overlap=0
        )

        self.assertEqual(chunks[0], ("unknown", "alpha beta", 2))

    def test_invalid_chunk_settings_fail(self) -> None:
        for chunk_size, overlap in ((0, 0), (4, -1), (4, 4), (4, 5)):
            with self.subTest(chunk_size=chunk_size, overlap=overlap):
                with self.assertRaises(ValueError):
                    chunk_cleaned_text(
                        "alpha beta",
                        chunk_size=chunk_size,
                        chunk_overlap=overlap,
                    )


class TickerSelectionTests(unittest.TestCase):
    def test_default_tickers_are_faang(self) -> None:
        self.assertEqual(resolve_requested_tickers(), DEFAULT_TICKERS)

    def test_single_ticker_overrides_default(self) -> None:
        self.assertEqual(resolve_requested_tickers(ticker="amzn"), ("AMZN",))

    def test_explicit_tickers_are_normalized(self) -> None:
        self.assertEqual(
            resolve_requested_tickers(tickers=["meta", "goog"]),
            ("META", "GOOG"),
        )

    def test_ticker_and_tickers_are_mutually_exclusive(self) -> None:
        with self.assertRaisesRegex(ValueError, "either"):
            resolve_requested_tickers(ticker="AMZN", tickers=["META"])

    def test_duplicate_explicit_tickers_fail(self) -> None:
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            resolve_requested_tickers(tickers=["AMZN", "amzn"])


class ManifestPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.project_root = Path.cwd() / f".test-data-{uuid.uuid4().hex}"
        self.project_root.mkdir()
        self.raw_directory = self.project_root / "data" / "raw" / "amzn"
        self.raw_directory.mkdir(parents=True)
        self.raw_path = self.raw_directory / "sample.html"
        self.raw_path.write_text(
            "<h1>Item 1. Business</h1><p>"
            + " ".join(f"word{i}" for i in range(11))
            + "</p>",
            encoding="utf-8",
        )
        record = FilingRecord(
            company="Amazon.com, Inc.",
            ticker="AMZN",
            cik="0001018724",
            filing_type="10-K",
            filing_date="2026-02-06",
            accession_number="0001018724-26-000001",
            source_url="https://www.sec.gov/example",
            local_raw_path="data/raw/amzn/sample.html",
            raw_format="html",
            local_markdown_path="data/human_readable/amzn/sample.md",
            human_readable_format="markdown",
            human_readable_warning="",
        )
        human_dir = self.project_root / "data" / "human_readable" / "amzn"
        human_dir.mkdir(parents=True)
        (human_dir / "sample.md").write_text("# Sample filing", encoding="utf-8")
        write_json(
            self.raw_directory / "filing_metadata.json",
            [record.to_dict()],
        )
        (self.raw_directory / "stale.html").write_text(
            "<p>This stale file is not in the manifest.</p>", encoding="utf-8"
        )

    def tearDown(self) -> None:
        shutil.rmtree(self.project_root, ignore_errors=True)

    def test_manifest_driven_clean_and_chunk_outputs(self) -> None:
        cleaned_manifest = clean_filings("AMZN", project_root=self.project_root)
        rows = chunk_filings(
            "AMZN",
            chunk_size=5,
            chunk_overlap=2,
            project_root=self.project_root,
        )

        self.assertEqual(len(cleaned_manifest), 1)
        self.assertNotIn("stale", cleaned_manifest[0]["local_cleaned_path"])
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0]["company"], "Amazon.com, Inc.")
        self.assertEqual(rows[0]["cik"], "0001018724")
        self.assertEqual(rows[0]["section_heading"], "Item 1. Business")
        self.assertEqual(rows[0]["chunk_index"], 0)
        self.assertEqual(
            rows[0]["chunk_id"], "amzn_000101872426000001_0000"
        )
        self.assertTrue(all(row["word_count"] <= 5 for row in rows))

        processed = self.project_root / "data" / "processed" / "amzn"
        csv_path = processed / "amzn_filing_chunks.csv"
        jsonl_path = processed / "amzn_filing_chunks.jsonl"
        with csv_path.open(encoding="utf-8", newline="") as handle:
            csv_rows = list(csv.DictReader(handle))
        jsonl_rows = [
            json.loads(line)
            for line in jsonl_path.read_text(encoding="utf-8").splitlines()
        ]
        self.assertEqual(len(csv_rows), len(jsonl_rows))
        self.assertEqual(len(jsonl_rows), len(rows))

    def test_missing_raw_manifest_fails_clearly(self) -> None:
        empty_directory = self.project_root / "empty-raw"
        empty_directory.mkdir()
        with self.assertRaisesRegex(FileNotFoundError, "preceding pipeline"):
            clean_filings("AMZN", project_root=empty_directory)

    def test_missing_cleaning_manifest_fails_clearly(self) -> None:
        empty_directory = self.project_root / "empty-clean"
        empty_directory.mkdir()
        with self.assertRaisesRegex(FileNotFoundError, "preceding pipeline"):
            chunk_filings("AMZN", project_root=empty_directory)


class AggregatePipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.project_root = Path.cwd() / f".test-data-{uuid.uuid4().hex}"
        self.project_root.mkdir()

    def tearDown(self) -> None:
        shutil.rmtree(self.project_root, ignore_errors=True)

    def _write_ticker_artifacts(
        self,
        ticker: str,
        words: list[int],
        filing_type: str = "10-K",
    ) -> None:
        slug = ticker.lower()
        raw_dir = self.project_root / "data" / "raw" / slug
        human_dir = self.project_root / "data" / "human_readable" / slug
        clean_dir = self.project_root / "data" / "processed" / slug / "cleaned_filings"
        processed_dir = self.project_root / "data" / "processed" / slug
        raw_dir.mkdir(parents=True)
        human_dir.mkdir(parents=True)
        clean_dir.mkdir(parents=True)
        accession = f"0000000000-26-{len(ticker):06d}"
        raw_path = raw_dir / "sample.html"
        human_path = human_dir / "sample.md"
        cleaned_path = clean_dir / "sample.txt"
        raw_path.write_text("<h1>Item 1. Business</h1>", encoding="utf-8")
        human_path.write_text("# Item 1. Business", encoding="utf-8")
        cleaned_path.write_text("## Item 1. Business\nalpha beta", encoding="utf-8")
        record = FilingRecord(
            company=f"{ticker} Company",
            ticker=ticker,
            cik=f"{len(ticker):010d}",
            filing_type=filing_type,
            filing_date="2026-01-01",
            accession_number=accession,
            source_url="https://www.sec.gov/example",
            local_raw_path=f"data/raw/{slug}/sample.html",
            raw_format="html",
            local_markdown_path=f"data/human_readable/{slug}/sample.md",
            human_readable_format="markdown",
            human_readable_warning="",
        )
        write_json(raw_dir / "filing_metadata.json", [record.to_dict()])
        write_json(
            processed_dir / "cleaning_manifest.json",
            [
                {
                    **record.to_dict(),
                    "local_cleaned_path": f"data/processed/{slug}/cleaned_filings/sample.txt",
                    "raw_length": 1000,
                    "cleaned_length": 120,
                    "warnings": ["large_length_reduction"],
                }
            ],
        )
        rows = [
            {
                "company": f"{ticker} Company",
                "ticker": ticker,
                "cik": f"{len(ticker):010d}",
                "filing_type": filing_type,
                "filing_date": "2026-01-01",
                "accession_number": accession,
                "source_url": "https://www.sec.gov/example",
                "source_file": f"data/processed/{slug}/cleaned_filings/sample.txt",
                "chunk_id": f"{slug}_{index:04d}",
                "chunk_index": index,
                "section_heading": "Item 1. Business" if index else "unknown",
                "text": " ".join(f"w{i}" for i in range(word_count)),
                "word_count": word_count,
            }
            for index, word_count in enumerate(words)
        ]
        write_jsonl(processed_dir / f"{slug}_filing_chunks.jsonl", rows)
        with (processed_dir / f"{slug}_filing_chunks.csv").open(
            "w",
            encoding="utf-8",
            newline="",
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def test_combined_chunks_and_description_are_generated(self) -> None:
        self._write_ticker_artifacts("META", [5, 400])
        self._write_ticker_artifacts("AMZN", [50, 450], filing_type="8-K")

        combined = combine_chunk_outputs(
            ["META", "AMZN"],
            dataset_slug="faang_test",
            project_root=self.project_root,
        )
        summary = describe_dataset(
            ["META", "AMZN"],
            dataset_slug="faang_test",
            chunk_size=400,
            project_root=self.project_root,
        )

        self.assertEqual(len(combined), 4)
        self.assertEqual(len({row["chunk_id"] for row in combined}), 4)
        combined_csv = (
            self.project_root
            / "data"
            / "processed"
            / "faang_test"
            / "faang_test_filing_chunks.csv"
        )
        combined_jsonl = (
            self.project_root
            / "data"
            / "processed"
            / "faang_test"
            / "faang_test_filing_chunks.jsonl"
        )
        self.assertTrue(combined_csv.exists())
        self.assertEqual(
            len(combined_jsonl.read_text(encoding="utf-8").splitlines()),
            4,
        )
        report_path = self.project_root / summary["output_paths"]["data_description"]
        report = report_path.read_text(encoding="utf-8")
        self.assertIn("Companies included", report)
        self.assertIn("Raw and cleaned character statistics", report)
        self.assertIn("Chunk counts by ticker and form", report)
        self.assertIn("Missing data policy", report)
        self.assertIn("Outlier policy", report)

        outliers = (
            self.project_root
            / "outputs"
            / "data_summary"
            / "faang_test_outlier_summary.csv"
        ).read_text(encoding="utf-8")
        self.assertIn("oversized_chunks,1", outliers)
        self.assertIn("short_chunks_under_100_words,2", outliers)


if __name__ == "__main__":
    unittest.main()
