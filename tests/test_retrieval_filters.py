"""Tests for filing-aware retrieval and dense metadata documents."""

from __future__ import annotations

import shutil
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

from src.config import CHUNK_SCHEMA_VERSION
from src.ingest_data.bm25 import search_bm25
from src.ingest_data.embeddings import search_embeddings
from src.ingest_data.index_common import select_ranked_indices
from src.LLM_response.retrieve_context import (
    infer_filing_constraints,
    retrieve_chunks,
)
from src.evaluation.run_eval import _dataset_matches_filing_policy


class FilingConstraintTests(unittest.TestCase):
    def test_explicit_form_and_comparison_constraints_are_inferred(self) -> None:
        self.assertEqual(
            infer_filing_constraints("What event is in the recent 8-K?"),
            (("8-K",), False),
        )
        self.assertEqual(
            infer_filing_constraints("Compared with the 10-K, what changed in the 10-Q?"),
            (("10-K", "10-Q"), True),
        )
        self.assertEqual(
            infer_filing_constraints("What drove recent profitability?"),
            ((), False),
        )

    def test_comparison_selection_reserves_one_chunk_per_named_form(self) -> None:
        chunks = [
            {"filing_type": "10-K"},
            {"filing_type": "10-K"},
            {"filing_type": "10-Q"},
        ]

        selected = select_ranked_indices(
            [0, 1, 2],
            chunks,
            k=2,
            filing_types=("10-K", "10-Q"),
            require_each_filing_type=True,
        )

        self.assertEqual(selected, [0, 2])

    def test_bm25_filters_to_the_explicit_form_without_index_text_boosting(self) -> None:
        chunks = [
            {"filing_type": "10-K", "text": "recent material event acquisition"},
            {"filing_type": "8-K", "text": "shareholders elected directors"},
        ]
        index = BM25Okapi([chunk["text"].lower().split() for chunk in chunks])

        result = search_bm25(
            "What recent material event is disclosed in the 8-K?",
            index,
            chunks,
            k=1,
            filing_types=("8-K",),
        )

        self.assertEqual(result[0]["filing_type"], "8-K")

    def test_faiss_searches_past_a_higher_scoring_wrong_form(self) -> None:
        chunks = [
            {"filing_type": "10-K", "text": "cover"},
            {"filing_type": "8-K", "text": "event"},
        ]
        index = faiss.IndexFlatIP(2)
        vectors = np.asarray([[1.0, 0.0], [0.8, 0.6]], dtype="float32")
        faiss.normalize_L2(vectors)
        index.add(vectors)
        client = MagicMock()
        client.embeddings.create.return_value = SimpleNamespace(
            data=[SimpleNamespace(embedding=[1.0, 0.0])]
        )

        with patch("src.ingest_data.embeddings._client", return_value=client):
            result = search_embeddings(
                "8-K event",
                index,
                chunks,
                model="test-model",
                k=1,
                filing_types=("8-K",),
            )

        self.assertEqual(result[0]["filing_type"], "8-K")

    def test_retrieve_chunks_passes_inferred_constraints_to_bm25(self) -> None:
        index = MagicMock()
        chunks = [{"filing_type": "8-K", "text": "event"}]
        with (
            patch(
                "src.LLM_response.retrieve_context.load_bm25_index",
                return_value=(index, chunks),
            ),
            patch(
                "src.LLM_response.retrieve_context.search_bm25",
                return_value=chunks,
            ) as search,
        ):
            retrieve_chunks(
                "What event is in the 8-K?",
                "TEST",
                retriever="bm25",
                build_if_missing=False,
            )

        search.assert_called_once_with(
            "What event is in the 8-K?",
            index,
            chunks,
            k=5,
            filing_types=("8-K",),
            require_each_filing_type=False,
        )


class DatasetSchemaTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = Path.cwd() / f".test-policy-{uuid.uuid4().hex}"
        self.directory.mkdir()
        self.chunks = self.directory / "chunks.csv"

    def tearDown(self) -> None:
        shutil.rmtree(self.directory, ignore_errors=True)

    def _write_chunks(self, schema_version: int) -> None:
        import pandas as pd

        pd.DataFrame(
            [
                {
                    "chunk_schema_version": schema_version,
                    "filing_type": filing_type,
                    "filing_date": filing_date,
                    "accession_number": accession,
                }
                for filing_type, filing_date, accession in (
                    ("10-K", "2026-02-01", "a"),
                    ("10-Q", "2026-04-30", "b"),
                    ("8-K", "2026-06-01", "c"),
                )
            ]
        ).to_csv(self.chunks, index=False)

    def test_old_chunk_schema_requires_transactional_rebuild(self) -> None:
        self._write_chunks(CHUNK_SCHEMA_VERSION - 1)
        self.assertFalse(_dataset_matches_filing_policy(self.chunks, num_8k=1))

        self._write_chunks(CHUNK_SCHEMA_VERSION)
        self.assertTrue(_dataset_matches_filing_policy(self.chunks, num_8k=1))


if __name__ == "__main__":
    unittest.main()
