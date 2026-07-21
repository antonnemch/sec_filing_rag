"""Smoke tests for the complete evaluation figure suite."""

from __future__ import annotations

import shutil
import unittest

import pandas as pd

from src.data.utils import PROJECT_ROOT
from src.visualizations import generate_evaluation_figures


class EvaluationVisualizationTests(unittest.TestCase):
    output = PROJECT_ROOT / "tests" / ".generated-eval-figures"

    def setUp(self) -> None:
        shutil.rmtree(self.output, ignore_errors=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.output, ignore_errors=True)

    @staticmethod
    def _scored_results() -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        for question in range(1, 9):
            for retriever in ("faiss", "bm25"):
                adjustment = 0.08 if retriever == "faiss" else 0.0
                rows.append(
                    {
                        "qa_id": f"TEST_Q{question}",
                        "ticker": "META" if question <= 4 else "AMZN",
                        "category": ((question - 1) % 4) + 1,
                        "answerable": question % 3 != 0,
                        "retriever": retriever,
                        "status": "generation_error" if question == 8 else "ok",
                        "retrieval_success": 1.0,
                        "generation_success": 0.0 if question == 8 else 1.0,
                        "document_recall_at_k": min(1.0, 0.55 + question * 0.03 + adjustment),
                        "document_reciprocal_rank": min(1.0, 0.45 + question * 0.04 + adjustment),
                        "chunk_recall_at_k": min(1.0, 0.35 + question * 0.04 + adjustment),
                        "chunk_reciprocal_rank": min(1.0, 0.30 + question * 0.04 + adjustment),
                        "cosine_sim": 0.60 + question * 0.02 + adjustment,
                        "word_overlap_f1": 0.40 + question * 0.025 + adjustment,
                        "llm_score": min(5.0, 2.5 + question * 0.2 + adjustment),
                    }
                )
        return pd.DataFrame(rows)

    def test_complete_suite_is_written_to_one_directory(self) -> None:
        created = generate_evaluation_figures(self._scored_results(), self.output)

        self.assertEqual(len(created), 9)
        self.assertTrue(all(path.parent == self.output for path in created))
        self.assertTrue(all(path.exists() and path.stat().st_size > 0 for path in created))

    def test_empty_results_skip_all_performance_figures(self) -> None:
        created = generate_evaluation_figures(pd.DataFrame(), self.output)

        self.assertEqual(created, [])


if __name__ == "__main__":
    unittest.main()
