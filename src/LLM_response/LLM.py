"""Call an LLM to answer questions grounded in retrieved filing context."""

from __future__ import annotations

import anthropic

DEFAULT_MODEL = "claude-sonnet-4-6"
MAX_TOKENS = 1024

_SYSTEM = (
    "You are a financial analyst assistant that answers questions about SEC filings. "
    "Answer using only the provided filing context. "
    "If the context does not contain enough information, say so clearly. "
    "Be concise and accurate."
)


def _format_context(chunks: list[dict]) -> str:
    parts = []
    for i, chunk in enumerate(chunks, 1):
        header = (
            f"[Chunk {i}] {chunk.get('filing_type', '')} "
            f"({chunk.get('filing_date', '')}) — {chunk.get('section_heading', '')}"
        )
        parts.append(f"{header}:\n{chunk.get('text', '')}")
    return "\n\n".join(parts)


def answer_question(
    question: str,
    context_chunks: list[dict],
    model: str = DEFAULT_MODEL,
) -> str:
    """Call the LLM with retrieved context and return the answer text."""
    client = anthropic.Anthropic()
    context = _format_context(context_chunks)
    user_message = f"Context from SEC filings:\n\n{context}\n\nQuestion: {question}"

    message = client.messages.create(
        model=model,
        max_tokens=MAX_TOKENS,
        system=_SYSTEM,
        messages=[{"role": "user", "content": user_message}],
    )
    return message.content[0].text
