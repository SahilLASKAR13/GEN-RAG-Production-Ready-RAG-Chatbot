# generation/prompt_builder.py
# Assembles the final prompt from system instructions, retrieved context,
# and the user's question. Handles token budget enforcement.

from __future__ import annotations
from typing import Any


SearchResult = dict[str, Any]

SYSTEM_TEMPLATE = """\
You are a helpful, accurate assistant. Answer the user's question using ONLY
the information provided in the context below. If the context does not contain
enough information to answer confidently, say so.

For each claim you make, cite the source using [Source N] notation based on
the numbered context items below.\
"""

CONTEXT_TEMPLATE = """\
=== Context ===
{context}
===============\
"""

CONTEXT_ITEM_TEMPLATE = "[Source {n}] (from {source})\n{text}"


class PromptBuilder:
    """
    Builds a (system_prompt, user_message) tuple ready for the LLM.
    Trims context to stay within max_context_chars.
    """

    def __init__(self, max_context_chars: int = 6000):
        self.max_context_chars = max_context_chars

    def build(
        self,
        query: str,
        retrieved: list[SearchResult],
        chat_history: list[dict] | None = None,
    ) -> tuple[str, list[dict]]:
        """
        Returns:
            system_prompt  — str to pass as the system message
            messages       — list of {"role": ..., "content": ...} dicts
        """
        context_str = self._format_context(retrieved)
        system_prompt = SYSTEM_TEMPLATE + "\n\n" + CONTEXT_TEMPLATE.format(context=context_str)

        messages: list[dict] = []
        for turn in (chat_history or []):
            messages.append(turn)
        messages.append({"role": "user", "content": query})

        return system_prompt, messages

    def _format_context(self, retrieved: list[SearchResult]) -> str:
        items = []
        total_chars = 0
        for n, chunk in enumerate(retrieved, start=1):
            source = chunk["metadata"].get("source", "unknown")
            text = chunk["text"].strip()
            item = CONTEXT_ITEM_TEMPLATE.format(n=n, source=source, text=text)
            if total_chars + len(item) > self.max_context_chars:
                break
            items.append(item)
            total_chars += len(item)
        return "\n\n".join(items)
