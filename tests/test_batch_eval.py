"""Tests for durable evaluation status and checkpoint behavior."""

from __future__ import annotations

import json
import shutil
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from src.config import EVALUATION_SCHEMA_VERSION
from src.LLM_response.batch_eval import (
    RESULT_COLUMNS,
    _read_checkpoint,
    _result_for_question,
    run_batch_eval,
)


def _eval_frame(rows: int = 1) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "qa_id": f"TEST_{index}",
                "question_id": index,
                "question": "What happened?",
                "answer": "Revenue increased.",
                "ticker": "TEST",
                "category": 1,
                "answerable": True,
                "source_doc_id": "TEST_10-K_2026-01-01",
            }
            for index in range(1, rows + 1)
        ]
    )


def _checkpoint_record(qa_id: str, status: str) -> dict:
    record = {column: "" for column in RESULT_COLUMNS}
    record.update(
        {
            "schema_version": EVALUATION_SCHEMA_VERSION,
            "run_fingerprint": "run-1",
            "qa_id": qa_id,
            "retriever": "bm25",
            "status": status,
            "retrieval_status": "ok",
            "generation_status": "ok" if status == "ok" else "error",
        }
    )
    return record


class BatchEvaluationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.project_root = Path.cwd() / f".test-eval-{uuid.uuid4().hex}"
        self.project_root.mkdir()
        self.output = self.project_root / "results.csv"

    def tearDown(self) -> None:
        shutil.rmtree(self.project_root, ignore_errors=True)

    def test_generation_failure_preserves_retrieval_evidence(self) -> None:
        row = SimpleNamespace(**_eval_frame().iloc[0].to_dict())
        retrieved = [
            {
                "ticker": "TEST",
                "filing_type": "10-K",
                "filing_date": "2026-01-01",
                "section_heading": "Business",
                "chunk_id": "chunk-1",
                "retrieval_score": 0.75,
            }
        ]
        with (
            patch("src.LLM_response.batch_eval.retrieve_chunks", return_value=retrieved),
            patch("src.LLM_response.batch_eval.answer_question", side_effect=RuntimeError("LLM down")),
        ):
            result = _result_for_question(
                row, "bm25", 5, "model", "fingerprint", self.project_root, None, False
            )

        self.assertEqual(result["status"], "generation_error")
        self.assertEqual(result["retrieval_status"], "ok")
        self.assertEqual(result["retrieved_chunk_ids"], "chunk-1")
        self.assertEqual(result["retrieved_doc_ids"], "TEST_10-K_2026-01-01")

    def test_nonpositive_k_fails_before_loading_evaluation_data(self) -> None:
        with patch("src.LLM_response.batch_eval.load_eval_set") as load:
            with self.assertRaisesRegex(ValueError, "at least 1"):
                run_batch_eval(tickers=["TEST"], k=0, project_root=self.project_root)
        load.assert_not_called()

    def test_truncated_final_checkpoint_line_is_removed(self) -> None:
        checkpoint = self.project_root / "checkpoint.jsonl"
        valid = _checkpoint_record("TEST_1", "ok")
        checkpoint.write_text(json.dumps(valid) + "\n{\"qa_id\":", encoding="utf-8")

        records = _read_checkpoint(checkpoint)

        self.assertEqual(list(records), [("TEST_1", "bm25")])
        self.assertEqual(checkpoint.read_text(encoding="utf-8"), json.dumps(valid) + "\n")

    def test_complete_final_checkpoint_line_gets_missing_newline_repaired(self) -> None:
        checkpoint = self.project_root / "checkpoint.jsonl"
        valid = _checkpoint_record("TEST_1", "ok")
        checkpoint.write_text(json.dumps(valid), encoding="utf-8")

        _read_checkpoint(checkpoint)

        self.assertEqual(checkpoint.read_text(encoding="utf-8"), json.dumps(valid) + "\n")

    def test_resume_retries_failed_rows_and_rejects_changed_fingerprint(self) -> None:
        evaluation = _eval_frame()
        first = _checkpoint_record("TEST_1", "generation_error")
        successful = _checkpoint_record("TEST_1", "ok")
        common_patches = (
            patch("src.LLM_response.batch_eval.load_eval_set", return_value=evaluation),
            patch("src.LLM_response.batch_eval.validate_source_doc_ids"),
        )
        with common_patches[0], common_patches[1], patch(
            "src.LLM_response.batch_eval._build_run_manifest",
            return_value={"run_fingerprint": "run-1"},
        ), patch(
            "src.LLM_response.batch_eval._result_for_question", return_value=first
        ):
            run_batch_eval(
                tickers=["TEST"],
                retriever="bm25",
                project_root=self.project_root,
                eval_csv=self.project_root / "eval.csv",
                output_csv=self.output,
            )

        with patch(
            "src.LLM_response.batch_eval.load_eval_set", return_value=evaluation
        ), patch("src.LLM_response.batch_eval.validate_source_doc_ids"), patch(
            "src.LLM_response.batch_eval._build_run_manifest",
            return_value={"run_fingerprint": "run-1"},
        ), patch(
            "src.LLM_response.batch_eval._result_for_question", return_value=successful
        ) as rerun:
            resumed = run_batch_eval(
                tickers=["TEST"],
                retriever="bm25",
                project_root=self.project_root,
                eval_csv=self.project_root / "eval.csv",
                output_csv=self.output,
                resume=True,
            )
        self.assertEqual(rerun.call_count, 1)
        self.assertEqual(resumed.iloc[0]["status"], "ok")

        with patch(
            "src.LLM_response.batch_eval.load_eval_set", return_value=evaluation
        ), patch("src.LLM_response.batch_eval.validate_source_doc_ids"), patch(
            "src.LLM_response.batch_eval._build_run_manifest",
            return_value={"run_fingerprint": "changed"},
        ), self.assertRaisesRegex(ValueError, "Cannot resume"):
            run_batch_eval(
                tickers=["TEST"],
                retriever="bm25",
                project_root=self.project_root,
                eval_csv=self.project_root / "eval.csv",
                output_csv=self.output,
                resume=True,
            )

    def test_interruption_materializes_completed_checkpoint_rows(self) -> None:
        evaluation = _eval_frame(2)
        first = _checkpoint_record("TEST_1", "ok")
        with patch(
            "src.LLM_response.batch_eval.load_eval_set", return_value=evaluation
        ), patch("src.LLM_response.batch_eval.validate_source_doc_ids"), patch(
            "src.LLM_response.batch_eval._build_run_manifest",
            return_value={"run_fingerprint": "run-1"},
        ), patch(
            "src.LLM_response.batch_eval._result_for_question",
            side_effect=[first, KeyboardInterrupt()],
        ), self.assertRaises(KeyboardInterrupt):
            run_batch_eval(
                tickers=["TEST"],
                retriever="bm25",
                project_root=self.project_root,
                eval_csv=self.project_root / "eval.csv",
                output_csv=self.output,
            )

        saved = pd.read_csv(self.output)
        self.assertEqual(saved["qa_id"].tolist(), ["TEST_1"])


if __name__ == "__main__":
    unittest.main()
