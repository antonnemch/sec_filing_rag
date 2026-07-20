"""Tests for Anthropic response parsing."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from src.config import API_MAX_RETRIES, API_TIMEOUT_SECONDS
from src.LLM_response.LLM import _extract_text_content, answer_question


class LLMResponseTests(unittest.TestCase):
    def test_all_text_blocks_are_joined(self) -> None:
        message = SimpleNamespace(
            content=[
                SimpleNamespace(type="text", text="First"),
                SimpleNamespace(type="tool_use", text=None),
                SimpleNamespace(type="text", text="Second"),
            ]
        )

        self.assertEqual(_extract_text_content(message), "First\nSecond")

    def test_missing_text_is_rejected(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "text content"):
            _extract_text_content(SimpleNamespace(content=[]))

    def test_answer_client_has_explicit_timeout_and_retries(self) -> None:
        client = MagicMock()
        client.messages.create.return_value = SimpleNamespace(
            content=[SimpleNamespace(type="text", text="Answer [Chunk 1]")]
        )
        with (
            patch("src.LLM_response.LLM.load_project_env"),
            patch("src.LLM_response.LLM.anthropic.Anthropic", return_value=client) as constructor,
        ):
            result = answer_question(
                "Question?",
                [{"text": "Evidence", "filing_type": "10-K", "filing_date": "2026-01-01"}],
            )

        self.assertEqual(result, "Answer [Chunk 1]")
        constructor.assert_called_once_with(
            timeout=API_TIMEOUT_SECONDS,
            max_retries=API_MAX_RETRIES,
        )


if __name__ == "__main__":
    unittest.main()
