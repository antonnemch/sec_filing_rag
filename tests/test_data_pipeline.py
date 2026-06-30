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

from src.data.chunk_filings import chunk_cleaned_text, chunk_filings
from src.data.clean_filings import clean_filings, clean_raw_content
from src.data.download_filings import _extract_filing_content
from src.data.utils import FilingRecord, configure_sec_identity, write_json


class FakeFiling:
    """Minimal filing object for testing HTML-to-text fallback behavior."""

    accession_no = "0000000000-26-000001"

    def __init__(self, html: str | None, text: str) -> None:
        self._html = html
        self._text = text

    def html(self) -> str | None:
        return self._html

    def text(self) -> str:
        return self._text


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
        )
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


if __name__ == "__main__":
    unittest.main()
