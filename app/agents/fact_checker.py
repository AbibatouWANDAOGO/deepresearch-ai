"""Fact Checker: checks each claim against passages retrieved for that claim.

The model proposes a verdict; code then enforces the rules a verdict must satisfy, so a
status is never stronger than the cited evidence:

- VERIFIED or PARTIALLY_VERIFIED needs at least one supporting passage;
- CONFLICTING needs at least one contradicting passage;
- VERIFIED with contradicting passages becomes CONFLICTING.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field

from app.agents.analyst import Claim
from app.agents.base import (
    UNTRUSTED_CONTENT_RULE,
    LabeledPassage,
    format_passages,
    label_passages,
    language_name,
    valid_labels,
)
from app.llm import LLMClient
from app.rag.retriever import Passage


class Status(StrEnum):
    VERIFIED = "VERIFIED"
    PARTIALLY_VERIFIED = "PARTIALLY_VERIFIED"
    CONFLICTING = "CONFLICTING"
    UNVERIFIED = "UNVERIFIED"


class VerdictDraft(BaseModel):
    status: Status
    supporting_ids: list[str] = Field(description="Ids of passages that state or directly imply the claim")
    contradicting_ids: list[str] = Field(description="Ids of passages that contradict the claim")
    justification: str = Field(description="One or two sentences explaining the verdict")


class CheckedClaim(BaseModel):
    claim: Claim
    status: Status
    supporting_chunk_ids: list[str]
    contradicting_chunk_ids: list[str]
    supporting_source_ids: list[str]
    justification: str
    adjusted: bool = False  # the model's status was downgraded by the rules above
    skipped: bool = False  # not checked (claim or time limit): unverified, but not evidence of a gap


SYSTEM = (
    "You are the fact-checking agent of a research assistant. For one claim, you decide whether the "
    "passages support it. VERIFIED: a passage states it clearly. PARTIALLY_VERIFIED: passages support "
    "part of it, or support it with different figures, dates or scope. CONFLICTING: some passages "
    "support it and others contradict it. UNVERIFIED: no passage supports it. Judge only from the "
    "passages, not from your own knowledge. " + UNTRUSTED_CONTENT_RULE
)


def build_prompt(claim: Claim, labeled: list[LabeledPassage], language: str = "en") -> str:
    return (f"Claim: {claim.text}\n\nPassages:\n{format_passages(labeled)}\n\n"
            f"Give the verdict for the claim. Write the justification in {language_name(language)}.")


def enforce_rules(draft: VerdictDraft, labeled: list[LabeledPassage]) -> tuple[Status, list[str], list[str], bool]:
    supporting = valid_labels(draft.supporting_ids, labeled)
    contradicting = [x for x in valid_labels(draft.contradicting_ids, labeled) if x not in supporting]
    status = draft.status
    if status == Status.VERIFIED and contradicting:
        status = Status.CONFLICTING
    if status in (Status.VERIFIED, Status.PARTIALLY_VERIFIED) and not supporting:
        status = Status.UNVERIFIED
    if status == Status.CONFLICTING and not contradicting:
        status = Status.PARTIALLY_VERIFIED if supporting else Status.UNVERIFIED
    return status, supporting, contradicting, status != draft.status


class FactChecker:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def run(self, claim: Claim, passages: list[Passage], language: str = "en") -> CheckedClaim:
        if not passages:
            return CheckedClaim(claim=claim, status=Status.UNVERIFIED, supporting_chunk_ids=[],
                                contradicting_chunk_ids=[], supporting_source_ids=[],
                                justification="No passage was found to check this claim.")
        labeled = label_passages(passages)
        chunk_of = {item.label: item.passage.chunk for item in labeled}
        draft = self.llm.structured(SYSTEM, build_prompt(claim, labeled, language), VerdictDraft)
        status, supporting, contradicting, adjusted = enforce_rules(draft, labeled)
        return CheckedClaim(
            claim=claim,
            status=status,
            supporting_chunk_ids=[chunk_of[x].chunk_id for x in supporting],
            contradicting_chunk_ids=[chunk_of[x].chunk_id for x in contradicting],
            supporting_source_ids=list(dict.fromkeys(chunk_of[x].source_id for x in supporting)),
            justification=draft.justification.strip(),
            adjusted=adjusted,
        )
