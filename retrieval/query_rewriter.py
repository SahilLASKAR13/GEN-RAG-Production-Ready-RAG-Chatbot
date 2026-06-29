# retrieval/query_rewriter.py
# Transforms a raw user query into one or more better search queries.
# Strategies: plain pass-through, HyDE, step-back, multi-query.

from __future__ import annotations
import json

from config.settings import settings


def _llm_complete(prompt: str, max_tokens: int = 256) -> str:
    """Lightweight one-shot LLM call used only for rewriting."""
    from openai import OpenAI
    client = OpenAI(api_key=settings.openai_api_key)
    resp = client.chat.completions.create(
        model=settings.llm_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content.strip()


class PassthroughRewriter:
    """No rewriting — returns the query as-is."""

    def rewrite(self, query: str) -> list[str]:
        return [query]


class HyDERewriter:
    """
    Hypothetical Document Embeddings: ask the LLM to write a short
    passage that *would* answer the query, then embed that passage.
    This aligns query semantics with document semantics.
    """

    def rewrite(self, query: str) -> list[str]:
        prompt = (
            "Write a concise 3-sentence passage that directly answers the following question. "
            "Do not say 'Here is' or any preamble — just the passage.\n\n"
            f"Question: {query}"
        )
        hypothesis = _llm_complete(prompt)
        return [hypothesis]


class StepBackRewriter:
    """
    Step-back prompting: generate a broader, more general version of
    the query to retrieve supporting background context.
    Returns both the original and the step-back query.
    """

    def rewrite(self, query: str) -> list[str]:
        prompt = (
            "Given the specific question below, write ONE broader, more general question "
            "that would help answer it by providing useful background knowledge.\n\n"
            f"Specific question: {query}\n\nBroader question:"
        )
        broader = _llm_complete(prompt)
        return [query, broader]


class MultiQueryRewriter:
    """
    Generate N alternative phrasings of the query to improve recall.
    Returns the original plus the alternatives.
    """

    def __init__(self, n: int = 3):
        self.n = n

    def rewrite(self, query: str) -> list[str]:
        prompt = (
            f"Generate {self.n} alternative phrasings of the following question. "
            "Return them as a JSON array of strings, no other text.\n\n"
            f"Question: {query}"
        )
        raw = _llm_complete(prompt, max_tokens=512)
        try:
            alternatives = json.loads(raw)
            if not isinstance(alternatives, list):
                raise ValueError
        except (json.JSONDecodeError, ValueError):
            alternatives = []
        return [query, *alternatives]


def get_query_rewriter(strategy: str = "passthrough"):
    strategies = {
        "passthrough":  PassthroughRewriter,
        "hyde":         HyDERewriter,
        "step_back":    StepBackRewriter,
        "multi_query":  MultiQueryRewriter,
    }
    cls = strategies.get(strategy)
    if not cls:
        raise ValueError(f"Unknown rewriter strategy: {strategy}")
    return cls()
