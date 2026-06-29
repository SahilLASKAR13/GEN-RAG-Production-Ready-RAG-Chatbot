# generation/prompt_builder.py
# Assembles the final prompt from system instructions, retrieved context,
# and the user's question. Handles token budget enforcement.

from __future__ import annotations

from typing import Any

from config.settings import settings


SearchResult = dict[str, Any]


SYSTEM_PROMPT = """
You are a helpful AI assistant.

Use ONLY the information provided in the retrieved context.

Rules:
1. Do not use outside knowledge.
2. If the answer cannot be found in the context, say:
   "I couldn't find enough information in the provided documents."
3. Cite every factual statement using [Source X].
4. If multiple sources support the answer, cite all relevant sources.
5. Keep answers clear, concise and accurate.
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
