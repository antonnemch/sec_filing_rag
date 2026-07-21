"""Offline tests for retrieval index manifests and shared metadata."""

from __future__ import annotations

import shutil
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src.config import DENSE_DOCUMENT_FORMAT_VERSION
from src.ingest_data.bm25 import build_bm25_index, load_bm25_index
from src.ingest_data.embeddings import build_embeddings_index, load_embeddings_index
from src.ingest_data.index_common import (
    CHUNKS_METADATA_FILE,
    IndexValidationError,
    load_validated_index_metadata,
)


class IndexIntegrityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.project_root = Path.cwd() / f".test-index-{uuid.uuid4().hex}"
        self.processed = self.project_root / "data" / "processed" / "test"
        self.processed.mkdir(parents=True)
        self.chunks_path = self.processed / "test_filing_chunks.csv"
        pd.DataFrame(
            [
                {
                    "chunk_id": "test_1",
                    "text": "revenue increased",
                    "ticker": "TEST",
                    "company": "Test Company",
                    "filing_type": "10-Q",
                    "filing_date": "2026-04-30",
                    "section_heading": "Item 2. Management Discussion",
                },
                {
                    "chunk_id": "test_2",
                    "text": "risk factors",
                    "ticker": "TEST",
                    "company": "Test Company",
                    "filing_type": "10-K",
                    "filing_date": "2026-01-31",
                    "section_heading": "Item 1A. Risk Factors",
                },
            ]
        ).to_csv(self.chunks_path, index=False)

    def tearDown(self) -> None:
        shutil.rmtree(self.project_root, ignore_errors=True)

    def test_bm25_uses_one_shared_snapshot_and_detects_stale_chunks(self) -> None:
        (self.processed / "bm25_chunks.pkl").write_bytes(b"legacy")
        build_bm25_index("TEST", self.project_root)

        index, chunks = load_bm25_index("TEST", self.project_root)
        self.assertEqual(len(index.doc_freqs), 2)
        self.assertEqual([row["chunk_id"] for row in chunks], ["test_1", "test_2"])
        self.assertTrue((self.processed / CHUNKS_METADATA_FILE).exists())
        self.assertFalse((self.processed / "bm25_chunks.pkl").exists())

        changed = pd.read_csv(self.chunks_path)
        changed.loc[0, "text"] = "materially changed source text"
        changed.to_csv(self.chunks_path, index=False)
        with self.assertRaisesRegex(IndexValidationError, "Stale bm25 index"):
            load_bm25_index("TEST", self.project_root)

    def test_manifest_settings_are_part_of_validation(self) -> None:
        build_bm25_index("TEST", self.project_root)

        with self.assertRaisesRegex(IndexValidationError, "tokenizer_version"):
            load_validated_index_metadata(
                "TEST",
                "bm25",
                self.project_root,
                tokenizer_version="different-tokenizer",
            )

    def test_faiss_manifest_includes_embedding_model_and_dimension(self) -> None:
        (self.processed / "embeddings_chunks.pkl").write_bytes(b"legacy")
        with patch(
            "src.ingest_data.embeddings._embed_texts",
            return_value=[[1.0, 0.0], [0.0, 1.0]],
        ) as embed:
            build_embeddings_index("TEST", self.project_root, model="test-model")

        embedded_texts = embed.call_args.args[0]
        self.assertIn("Filing type: 10-Q", embedded_texts[0])
        self.assertIn("Filing date: 2026-04-30", embedded_texts[0])
        self.assertIn("Section: Item 2. Management Discussion", embedded_texts[0])
        self.assertTrue(embedded_texts[0].endswith("Content:\nrevenue increased"))

        index, chunks = load_embeddings_index(
            "TEST", self.project_root, model="test-model"
        )
        self.assertEqual(index.d, 2)
        self.assertEqual(len(chunks), 2)
        self.assertFalse((self.processed / "embeddings_chunks.pkl").exists())
        _, manifest = load_validated_index_metadata(
            "TEST",
            "faiss",
            self.project_root,
            embedding_model="test-model",
            vector_dimension=2,
            document_format_version=DENSE_DOCUMENT_FORMAT_VERSION,
        )
        self.assertEqual(
            manifest["document_format_version"], DENSE_DOCUMENT_FORMAT_VERSION
        )
        with self.assertRaisesRegex(IndexValidationError, "embedding_model"):
            load_embeddings_index("TEST", self.project_root, model="changed-model")


if __name__ == "__main__":
    unittest.main()
