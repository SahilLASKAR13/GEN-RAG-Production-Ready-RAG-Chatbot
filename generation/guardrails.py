# generation/guardrails.py
# Post-generation quality checks: hallucination detection and safety filter.

from __future__ import annotations
import re
from dataclasses import dataclass


@dataclass
class GuardrailResult:
    passed: bool
    reason: str = ""
    safe_response: str = ""


class GroundednessChecker:
    """
    Checks whether the answer is grounded in the retrieved context.
    Strategy: look for claims in the answer that reference [Source N] markers.
    A stricter implementation would use NLI (Natural Language Inference).
    """

    def check(self, answer: str, context_texts: list[str]) -> GuardrailResult:
        # Fast path: if answer admits uncertainty, it's acceptable
        uncertainty_phrases = [
            "i don't know", "not mentioned", "not provided",
            "cannot find", "no information", "context does not",
        ]
        lower = answer.lower()
        if any(p in lower for p in uncertainty_phrases):
            return GuardrailResult(passed=True)

        # Check that at least one source citation exists for non-trivial answers
        if len(answer) > 200 and not re.search(r"\[Source \d+\]", answer):
            return GuardrailResult(
                passed=False,
                reason="Answer contains no source citations.",
                safe_response=(
                    "I found some relevant information but cannot provide a "
                    "fully cited answer. Please review the source documents directly."
                ),
            )
        return GuardrailResult(passed=True)


class SafetyFilter:
    """
    Basic keyword-level safety filter.
    In production, replace with a dedicated moderation API call.
    """

    BLOCKED_PATTERNS = [
        r"\b(how to make|synthesise|synthesize)\s+(bomb|explosiv|weapon|poison)\b",
        r"\bself[- ]?harm\b",
    ]

    def check(self, text: str) -> GuardrailResult:
        for pattern in self.BLOCKED_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                return GuardrailResult(
                    passed=False,
                    reason="Response flagged by safety filter.",
                    safe_response="I'm not able to help with that request.",
                )
        return GuardrailResult(passed=True)


class GuardrailPipeline:
    """Runs all checks in sequence; short-circuits on first failure."""

    def __init__(self):
        self.checks = [SafetyFilter(), GroundednessChecker()]

    def run(self, answer: str, context_texts: list[str]) -> GuardrailResult:
        for check in self.checks:
            if isinstance(check, GroundednessChecker):
                result = check.check(answer, context_texts)
            else:
                result = check.check(answer)
            if not result.passed:
                return result
        return GuardrailResult(passed=True)
