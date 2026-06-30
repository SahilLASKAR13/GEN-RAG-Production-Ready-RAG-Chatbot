# generation/prompt_builder.py
# Assembles the final prompt from system instructions, retrieved context,
# and the user's question. Handles token budget enforcement.

from __future__ import annotations

from typing import Any

from config.settings import settings


SearchResult = dict[str, Any]


SYSTEM_PROMPT = """
You are a helpful AI assistant.

Answer the user's question ONLY using the retrieved context below.

Instructions:

1. Use only information present in the retrieved context.
2. Do not invent facts or use outside knowledge.
3. If the answer cannot be found in the context, reply:
   "I couldn't find enough information in the provided documents."
4. Write a complete, well-structured explanation instead of a one-line answer.
5. Combine information from multiple retrieved sources whenever possible.
6. Use bullet points whenever they improve readability.
7. Explain technical concepts in simple language.
8. Cite every paragraph or factual statement using [Source X].
9. If multiple sources support a statement, cite all of them.
10. Do not simply copy the context. Rewrite it naturally.
"""


class PromptBuilder:

    def __init__(
        self,
        max_context_chars: int = settings.max_context_chars,
    ):
        self.max_context_chars = max_context_chars

    def build(
        self,
        query: str,
        retrieved: list[SearchResult],
        chat_history: list[dict] | None = None,
    ) -> tuple[str, list[dict]]:

        context = self._build_context(retrieved)

        system_prompt = (
            SYSTEM_PROMPT
            + "\n\nRetrieved Context:\n\n"
            + context
        )

        messages = []

        if chat_history:

            messages.extend(chat_history)

        messages.append(
            {
                "role": "user",
                "content": query,
            }
        )

        return system_prompt, messages

    def _build_context(
        self,
        retrieved: list[SearchResult],
    ) -> str:

        context = []
        total_chars = 0

        for i, result in enumerate(retrieved, start=1):

            metadata = result["metadata"]

            source = metadata.get("source", "Unknown")

            page = metadata.get("page")

            if page is not None:
                header = f"[Source {i}] ({source}, page {page})"
            else:
                header = f"[Source {i}] ({source})"

            chunk = f"{header}\n{result['text'].strip()}"

            if total_chars + len(chunk) > self.max_context_chars:
                break

            context.append(chunk)

            total_chars += len(chunk)

        return "\n\n".join(context)
