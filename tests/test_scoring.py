"""Tests for retrieval scoring."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import pandas as pd

from src.config import EVALUATION_SCHEMA_VERSION
from src.run_tests.score_eval import (
    _parse_judge_score,
    _retrieval_scores,
    _word_f1,
    score,
)


class RetrievalScoringTests(unittest.TestCase):
    @staticmethod
    def _result_row(**overrides: object) -> dict:
        row = {
            "schema_version": EVALUATION_SCHEMA_VERSION,
            "run_fingerprint": "run-1",
            "qa_id": "Q1",
            "ticker": "TEST",
            "question": "What happened?",
            "ground_truth": "Revenue increased.",
            "source_doc_id": "DOC-A",
            "retrieved_doc_ids": "DOC-A",
            "retrieved_chunk_ids": "CHUNK-1",
            "retrieval_k": 5,
            "retriever": "bm25",
            "category": 1,
            "answerable": True,
            "llm_answer": "Revenue increased.",
            "retrieval_status": "ok",
            "generation_status": "ok",
            "status": "ok",
        }
        row.update(overrides)
        return row

    def test_document_and_chunk_metrics_use_first_relevant_rank(self) -> None:
        df = pd.DataFrame(
            [
                {
                    "source_doc_id": "DOC-B",
                    "retrieved_doc_ids": "DOC-A|DOC-B|DOC-B",
                    "source_chunk_ids": "CHUNK-2|CHUNK-3",
                    "retrieved_chunk_ids": "CHUNK-1|CHUNK-2|CHUNK-4",
                }
            ]
        )

        scored = _retrieval_scores(df).iloc[0]

        self.assertEqual(scored.document_recall_at_k, 1.0)
        self.assertEqual(scored.document_reciprocal_rank, 0.5)
        self.assertEqual(scored.chunk_recall_at_k, 0.5)
        self.assertEqual(scored.chunk_reciprocal_rank, 0.5)

    def test_missing_chunk_labels_produce_unavailable_metrics(self) -> None:
        df = pd.DataFrame(
            [
                {
                    "source_doc_id": "DOC-A",
                    "retrieved_doc_ids": "DOC-A",
                    "source_chunk_ids": "",
                    "retrieved_chunk_ids": "CHUNK-1",
                }
            ]
        )

        scored = _retrieval_scores(df).iloc[0]

        self.assertTrue(pd.isna(scored.chunk_recall_at_k))
        self.assertTrue(pd.isna(scored.chunk_reciprocal_rank))

    def test_multi_marker_without_exact_docs_has_unavailable_document_metrics(self) -> None:
        df = pd.DataFrame(
            [
                {
                    "source_doc_id": "AMZN_MULTI",
                    "retrieved_doc_ids": "AMZN_10-K_2026-02-06|AMZN_10-Q_2026-04-30",
                    "source_chunk_ids": "",
                    "retrieved_chunk_ids": "CHUNK-1|CHUNK-2",
                }
            ]
        )

        scored = _retrieval_scores(df).iloc[0]

        self.assertTrue(pd.isna(scored.document_recall_at_k))
        self.assertTrue(pd.isna(scored.document_reciprocal_rank))

    def test_frequency_aware_word_f1_is_not_set_overlap(self) -> None:
        self.assertEqual(_word_f1("a a b", "a b b"), 0.6667)

    def test_generation_failure_still_receives_retrieval_metrics(self) -> None:
        result = self._result_row(
            llm_answer="",
            generation_status="error",
            status="generation_error",
        )

        scored = score(pd.DataFrame([result]), use_embeddings=False).iloc[0]

        self.assertEqual(scored.document_recall_at_k, 1.0)
        self.assertTrue(pd.isna(scored.word_overlap_f1))

    def test_all_error_rows_are_retained_without_embedding_call(self) -> None:
        result = self._result_row(
            llm_answer="",
            retrieved_doc_ids="",
            retrieval_status="error",
            generation_status="not_run",
            status="retrieval_error",
        )
        with patch("src.run_tests.score_eval._embed") as embed:
            scored = score(pd.DataFrame([result]), use_embeddings=True)

        embed.assert_not_called()
        self.assertEqual(len(scored), 1)
        self.assertTrue(pd.isna(scored.iloc[0].cosine_sim))

    def test_judge_parser_accepts_only_in_range_integer_response(self) -> None:
        self.assertEqual(_parse_judge_score("5."), 5.0)
        self.assertEqual(_parse_judge_score(" 1 "), 1.0)
        self.assertTrue(pd.isna(_parse_judge_score("Score: 5")))
        self.assertTrue(pd.isna(_parse_judge_score("6")))

    def test_inconsistent_status_contract_is_rejected(self) -> None:
        invalid = self._result_row(
            retrieval_status="error",
            generation_status="ok",
            status="ok",
        )

        with self.assertRaisesRegex(ValueError, "inconsistent"):
            score(pd.DataFrame([invalid]), use_embeddings=False)


if __name__ == "__main__":
    unittest.main()
