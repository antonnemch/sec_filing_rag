"""Tests for evaluation-runner environment and CLI configuration."""

from __future__ import annotations

import os
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from src.config import DEFAULT_RETRIEVAL_K
from src.evaluation.run_eval import build_parser
from src.evaluation.run_artifacts import new_run_name, output_for_run, validate_run_name


class RunEvalConfigTests(unittest.TestCase):
    def test_builtin_k_is_used_without_environment_or_cli_value(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            args = build_parser().parse_args([])

        self.assertEqual(args.k, DEFAULT_RETRIEVAL_K)

    def test_environment_sets_default_k(self) -> None:
        with patch.dict(os.environ, {"RAG_TOP_K": "9"}, clear=True):
            args = build_parser().parse_args([])

        self.assertEqual(args.k, 9)

    def test_cli_k_overrides_environment(self) -> None:
        with patch.dict(os.environ, {"RAG_TOP_K": "9"}, clear=True):
            args = build_parser().parse_args(["--k", "3"])

        self.assertEqual(args.k, 3)

    def test_invalid_environment_k_fails_clearly(self) -> None:
        for value in ("0", "abc", ""):
            with self.subTest(value=value):
                with patch.dict(os.environ, {"RAG_TOP_K": value}, clear=True):
                    with self.assertRaisesRegex(
                        ValueError,
                        r"RAG_TOP_K must be an integer of at least 1",
                    ):
                        build_parser()

    def test_run_name_maps_to_isolated_results_directory(self) -> None:
        output = output_for_run(Path("project"), "k10-comparison")

        self.assertEqual(
            output,
            Path("project/outputs/eval_results/runs/k10-comparison/eval_results.csv"),
        )

    def test_generated_run_name_is_stable_utc_timestamp(self) -> None:
        instant = datetime(2026, 7, 20, 19, 5, 4, 123456, tzinfo=timezone.utc)

        self.assertEqual(new_run_name(instant), "20260720T190504_123456Z")

    def test_path_traversal_run_name_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Run names must"):
            validate_run_name("../old-run")


if __name__ == "__main__":
    unittest.main()
