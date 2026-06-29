# generation/llm.py
# Thin provider-agnostic wrapper for chat completions.
# Supports OpenAI, Anthropic Claude, and local Ollama.

from __future__ import annotations
from dataclasses import dataclass
from typing import Iterator

from config.settings import settings


@dataclass
class LLMResponse:
    text: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0


class OpenAILLM:
    def __init__(self):
        from openai import OpenAI
        self.client = OpenAI(api_key=settings.openai_api_key)

    def complete(self, system: str, messages: list[dict]) -> LLMResponse:
        resp = self.client.chat.completions.create(
            model=settings.llm_model,
            messages=[{"role": "system", "content": system}, *messages],
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
        )
        choice = resp.choices[0].message
        return LLMResponse(
            text=choice.content,
            model=resp.model,
            input_tokens=resp.usage.prompt_tokens,
            output_tokens=resp.usage.completion_tokens,
        )

    def stream(self, system: str, messages: list[dict]) -> Iterator[str]:
        stream = self.client.chat.completions.create(
            model=settings.llm_model,
            messages=[{"role": "system", "content": system}, *messages],
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta


class AnthropicLLM:
    def __init__(self):
        import anthropic
        self.client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

    def complete(self, system: str, messages: list[dict]) -> LLMResponse:
        resp = self.client.messages.create(
            model=settings.llm_model,
            system=system,
            messages=messages,
            max_tokens=settings.llm_max_tokens,
            temperature=settings.llm_temperature,
        )
        return LLMResponse(
            text=resp.content[0].text,
            model=resp.model,
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
        )

    def stream(self, system: str, messages: list[dict]) -> Iterator[str]:
        with self.client.messages.stream(
            model=settings.llm_model,
            system=system,
            messages=messages,
            max_tokens=settings.llm_max_tokens,
        ) as stream:
            yield from stream.text_stream


class OllamaLLM:
    """Local Ollama server — no API key required."""

    def __init__(self, base_url: str = "http://localhost:11434"):
        import httpx
        self.base_url = base_url
        self.http = httpx.Client()

    def complete(self, system: str, messages: list[dict]) -> LLMResponse:
        payload = {
            "model": settings.llm_model,
            "messages": [{"role": "system", "content": system}, *messages],
            "stream": False,
        }
        resp = self.http.post(f"{self.base_url}/api/chat", json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        return LLMResponse(
            text=data["message"]["content"],
            model=data.get("model", settings.llm_model),
        )

    def stream(self, system: str, messages: list[dict]) -> Iterator[str]:
        import json
        payload = {
            "model": settings.llm_model,
            "messages": [{"role": "system", "content": system}, *messages],
            "stream": True,
        }
        with self.http.stream("POST", f"{self.base_url}/api/chat", json=payload, timeout=120) as r:
            for line in r.iter_lines():
                if line:
                    data = json.loads(line)
                    delta = data.get("message", {}).get("content", "")
                    if delta:
                        yield delta


def get_llm():
    provider = settings.llm_provider
    if provider == "openai":
        return OpenAILLM()
    if provider == "anthropic":
        return AnthropicLLM()
    if provider == "ollama":
        return OllamaLLM()
    raise ValueError(f"Unknown LLM provider: {provider}")
