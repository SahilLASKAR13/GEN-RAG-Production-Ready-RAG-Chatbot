# retrieval/query_rewriter.py
# Transforms a raw user query into one or more better search queries.
# Strategies: plain pass-through, HyDE, step-back, multi-query.

from __future__ import annotations

import json

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

from config.settings import settings


class LLMClient:
    """Shared Groq LLM client."""

    def __init__(self):
        self.llm = ChatGroq(
            model=settings.llm_model,
            api_key=settings.groq_api_key,
            temperature=0.0,
            max_tokens=settings.llm_max_tokens,
        )

    def complete(
        self,
        prompt: str,
    ) -> str:

        response = self.llm.invoke(
            [HumanMessage(content=prompt)]
        )

        return response.content.strip()


llm = LLMClient()


class PassthroughRewriter:
    """Return the original query unchanged."""

    def rewrite(
        self,
        query: str,
    ) -> list[str]:

        return [query]


class HyDERewriter:
    """
    Hypothetical Document Embeddings.
    """

    def rewrite(
        self,
        query: str,
    ) -> list[str]:

        prompt = f"""
Write a concise three-sentence passage that directly answers the question.

Question:
{query}

Only return the passage.
"""

        hypothesis = llm.complete(prompt)

        return [hypothesis]


class StepBackRewriter:
    """
    Generate a broader version of the query.
    """

    def rewrite(
        self,
        query: str,
    ) -> list[str]:

        prompt = f"""
Given the following question, generate one broader question that would
help retrieve useful background information.

Question:
{query}

Only return the broader question.
"""

        broader = llm.complete(prompt)

        return [query, broader]


class MultiQueryRewriter:
    """
    Generate multiple paraphrases of the query.
    """

    def __init__(
        self,
        n: int = 3,
    ):
        self.n = n

    def rewrite(
        self,
        query: str,
    ) -> list[str]:

        prompt = f"""
Generate {self.n} different paraphrases of the following question.

Return ONLY a JSON list.

Question:
{query}
"""

        response = llm.complete(prompt)

        try:
            queries = json.loads(response)

            if not isinstance(queries, list):
                raise ValueError

        except Exception:
            queries = []

        return [query, *queries]


def get_query_rewriter(
    strategy: str = "passthrough",
):

    strategies = {
        "passthrough": PassthroughRewriter,
        "hyde": HyDERewriter,
        "step_back": StepBackRewriter,
        "multi_query": MultiQueryRewriter,
    }

    if strategy not in strategies:
        raise ValueError(f"Unknown strategy: {strategy}")

    return strategies[strategy]()
