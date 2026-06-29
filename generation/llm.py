# generation/llm.py
# Thin provider-agnostic wrapper for chat completions.
# Supports OpenAI, Anthropic Claude, and local Ollama.

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_groq import ChatGroq

from config.settings import settings


@dataclass
class LLMResponse:
    text: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0


class GroqLLM:
    """
    Groq LLM wrapper using LangChain.
    """

    def __init__(self):

        self.llm = ChatGroq(
            api_key=settings.groq_api_key,
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
        )

    def _build_messages(
        self,
        system: str,
        messages: list[dict],
    ):

        chat = [
            SystemMessage(content=system)
        ]

        for message in messages:

            role = message["role"]
            content = message["content"]

            if role == "user":
                chat.append(HumanMessage(content=content))

            elif role == "assistant":
                chat.append(AIMessage(content=content))

        return chat

    def complete(
        self,
        system: str,
        messages: list[dict],
    ) -> LLMResponse:

        response = self.llm.invoke(
            self._build_messages(system, messages)
        )

        return LLMResponse(
            text=response.content,
            model=settings.llm_model,
        )

    def stream(
        self,
        system: str,
        messages: list[dict],
    ) -> Iterator[str]:

        for chunk in self.llm.stream(
            self._build_messages(system, messages)
        ):

            if chunk.content:
                yield chunk.content


def get_llm() -> GroqLLM:
    """
    Return the configured LLM.
    """

    return GroqLLM()
