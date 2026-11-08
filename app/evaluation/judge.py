"""Citation judge: does the passage cited for a claim really state it?

Stricter than the Fact Checker on purpose: same topic is not enough, figures, dates and scope
must match. With the local model, the judge is the same model as the Fact Checker, so its
verdicts are a signal, not ground truth; manual annotations (eval/annotations.json) are
reported next to them, with the agreement between the two.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.agents.base import UNTRUSTED_CONTENT_RULE, escape_untrusted
from app.evaluation.metrics import Verdict
from app.llm import LLMClient

SYSTEM = (
    "You audit citations. You are given a claim and the passages cited as evidence for it. Decide "
    "whether the passages, read literally, state the claim. SUPPORTED: they state every part of the "
    "claim, with the same figures, dates and scope. PARTIAL: they state part of the claim, or the "
    "claim changes a figure, a date or the scope. NOT_SUPPORTED: they do not state the claim, even if "
    "they are about the same topic. Do not use your own knowledge: a true claim the passages do not "
    "state is NOT_SUPPORTED. " + UNTRUSTED_CONTENT_RULE
)


class Judgment(BaseModel):
    verdict: Verdict
    reason: str = Field(description="One sentence: what the passages state or do not state")


def build_prompt(claim: str, passages: list[str]) -> str:
    blocks = "\n\n".join(f'<document id="P{i}">\n{escape_untrusted(p)}\n</document>'
                         for i, p in enumerate(passages, start=1))
    return f"Claim: {claim}\n\nCited passages:\n{blocks}\n\nGive your verdict."


def judge(llm: LLMClient, claim: str, passages: list[str]) -> Judgment:
    if not passages:
        return Judgment(verdict="NOT_SUPPORTED", reason="no passage is cited")
    return llm.structured(SYSTEM, build_prompt(claim, passages), Judgment)
