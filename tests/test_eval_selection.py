"""Tests for evaluation-set validation and completed-question selection."""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src.LLM_response.ground_truth import (
    filter_completed_questions,
    load_eval_set,
    validate_source_doc_ids,
)


class EvalSelectionTests(unittest.TestCase):
    @staticmethod
    def _row(qa_id: str, answer: object = "reference") -> dict:
        return {
            "qa_id": qa_id,
            "question_id": 1,
            "question": "What happened?",
            "answer": answer,
            "ticker": "TEST",
            "category": 1,
            "answerable": True,
            "source_doc_id": "TEST_10-K_2026-01-01",
        }

    def test_completed_filter_excludes_blank_and_missing_answers(self) -> None:
        df = pd.DataFrame(
            [self._row("Q1"), self._row("Q2", "  "), self._row("Q3", None)]
        )

        selected = filter_completed_questions(df)

        self.assertEqual(selected["qa_id"].tolist(), ["Q1"])

    def test_default_eval_set_is_complete_dataset(self) -> None:
        explicit = load_eval_set(Path("eval_sets/faang_eval_set_complete.csv"))

        self.assertEqual(load_eval_set()["qa_id"].tolist(), explicit["qa_id"].tolist())

    def test_missing_required_column_fails(self) -> None:
        row = self._row("Q1")
        del row["answer"]

        with patch("src.LLM_response.ground_truth.pd.read_csv", return_value=pd.DataFrame([row])):
            with self.assertRaisesRegex(ValueError, "missing required column.*answer"):
                load_eval_set(Path("README.md"))

    def test_duplicate_qa_id_fails(self) -> None:
        df = pd.DataFrame([self._row("Q1"), self._row("Q1")])

        with patch("src.LLM_response.ground_truth.pd.read_csv", return_value=df):
            with self.assertRaisesRegex(ValueError, "duplicate qa_id.*Q1"):
                load_eval_set(Path("README.md"))

    def test_blank_question_fails(self) -> None:
        row = self._row("Q1")
        row["question"] = "  "

        with patch("src.LLM_response.ground_truth.pd.read_csv", return_value=pd.DataFrame([row])):
            with self.assertRaisesRegex(ValueError, "blank question"):
                load_eval_set(Path("README.md"))

    def test_invalid_category_fails(self) -> None:
        row = self._row("Q1")
        row["category"] = 5

        with patch("src.LLM_response.ground_truth.pd.read_csv", return_value=pd.DataFrame([row])):
            with self.assertRaisesRegex(ValueError, "invalid category"):
                load_eval_set(Path("README.md"))

    def test_invalid_answerable_fails(self) -> None:
        row = self._row("Q1")
        row["answerable"] = "maybe"

        with patch(
            "src.LLM_response.ground_truth.pd.read_csv",
            return_value=pd.DataFrame([row]),
        ):
            with self.assertRaisesRegex(ValueError, "invalid answerable"):
                load_eval_set(Path("README.md"))

    def test_source_doc_id_matches_chunk_database(self) -> None:
        eval_df = pd.DataFrame([self._row("Q1")])
        chunks = pd.DataFrame(
            [{"ticker": "TEST", "filing_type": "10-K", "filing_date": "2026-01-01"}]
        )

        with (
            patch("pathlib.Path.exists", return_value=True),
            patch("src.LLM_response.ground_truth.pd.read_csv", return_value=chunks),
        ):
            validate_source_doc_ids(eval_df, Path("project"))

    def test_stale_source_doc_id_lists_available_documents(self) -> None:
        eval_df = pd.DataFrame([self._row("Q1")])
        chunks = pd.DataFrame(
            [{"ticker": "TEST", "filing_type": "10-K", "filing_date": "2026-02-01"}]
        )

        with (
            patch("pathlib.Path.exists", return_value=True),
            patch("src.LLM_response.ground_truth.pd.read_csv", return_value=chunks),
            self.assertRaisesRegex(
                ValueError,
                "Q1: expected TEST_10-K_2026-01-01; available: TEST_10-K_2026-02-01",
            ),
        ):
            validate_source_doc_ids(eval_df, Path("project"))

    def test_multi_marker_without_exact_ids_is_allowed(self) -> None:
        row = self._row("Q1")
        row["source_doc_id"] = "TEST_MULTI"
        eval_df = pd.DataFrame([row])
        chunks = pd.DataFrame(
            [{"ticker": "TEST", "filing_type": "10-K", "filing_date": "2026-01-01"}]
        )

        with (
            patch("pathlib.Path.exists", return_value=True),
            patch("src.LLM_response.ground_truth.pd.read_csv", return_value=chunks),
        ):
            validate_source_doc_ids(eval_df, Path("project"))

    def test_multi_exact_document_ids_are_all_validated(self) -> None:
        row = self._row("Q1")
        row["source_doc_id"] = "TEST_MULTI"
        row["source_doc_ids"] = "TEST_10-K_2026-01-01|TEST_10-Q_2026-02-01"
        eval_df = pd.DataFrame([row])
        chunks = pd.DataFrame(
            [{"ticker": "TEST", "filing_type": "10-K", "filing_date": "2026-01-01"}]
        )

        with (
            patch("pathlib.Path.exists", return_value=True),
            patch("src.LLM_response.ground_truth.pd.read_csv", return_value=chunks),
            self.assertRaisesRegex(ValueError, "expected TEST_10-Q_2026-02-01"),
        ):
            validate_source_doc_ids(eval_df, Path("project"))


if __name__ == "__main__":
    unittest.main()
