# generation/guardrails.py
# Post-generation quality checks: hallucination detection and safety filter.

from __future__ import annotations

import re
from dataclasses import dataclass

from config.settings import settings


@dataclass
class GuardrailResult:
    passed: bool
    reason: str = ""
    safe_response: str = ""


class SafetyFilter:
    """
    Blocks unsafe requests.
    """

    BLOCKED_PATTERNS = [
        r"\b(how to make|how to build)\s+(bomb|explosive|weapon|poison)\b",
        r"\bself[- ]?harm\b",
        r"\bsuicide\b",
    ]

    def check(
        self,
        text: str,
    ) -> GuardrailResult:

        for pattern in self.BLOCKED_PATTERNS:

            if re.search(pattern, text, re.IGNORECASE):

                return GuardrailResult(
                    passed=False,
                    reason="Unsafe response detected.",
                    safe_response="I'm not able to help with that request.",
                )

        return GuardrailResult(passed=True)


class GroundednessChecker:
    """
    Ensure answers are supported by retrieved documents.
    """

    UNCERTAINTY_PHRASES = [
        "i don't know",
        "cannot find",
        "not found",
        "not mentioned",
        "not available",
        "not enough information",
    ]

    def check(
        self,
        answer: str,
        retrieved_chunks: list[str],
    ) -> GuardrailResult:

        answer_lower = answer.lower()

        if any(
            phrase in answer_lower
            for phrase in self.UNCERTAINTY_PHRASES
        ):
            return GuardrailResult(True)

        if (
            len(answer) >= settings.min_answer_length_for_citation
            and re.search(r"\[Source \d+\]", answer) is None
        ):

            return GuardrailResult(
                passed=False,
                reason="Missing citations.",
                safe_response=(
                    "I couldn't generate a fully grounded answer from the retrieved documents."
                ),
            )

        return GuardrailResult(True)


class GuardrailPipeline:

    def __init__(self):

        self.enabled = settings.enable_guardrails

        self.checks = [
            SafetyFilter(),
            GroundednessChecker(),
        ]

    def run(
        self,
        answer: str,
        context_chunks: list[str],
    ) -> GuardrailResult:

        if not self.enabled:
            return GuardrailResult(True)

        for check in self.checks:

            if isinstance(check, GroundednessChecker):

                result = check.check(
                    answer,
                    context_chunks,
                )

            else:

                result = check.check(answer)

            if not result.passed:
                return result

        return GuardrailResult(True)
