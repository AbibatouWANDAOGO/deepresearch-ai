"""Analyst: answers one sub-question from retrieved passages, with a citation for every finding."""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.agents.base import (
    UNTRUSTED_CONTENT_RULE,
    LabeledPassage,
    format_passages,
    label_passages,
    language_name,
    strip_passage_refs,
    valid_labels,
)
from app.agents.planner import SubQuestion
from app.llm import LLMClient
from app.rag.retriever import Passage


class Finding(BaseModel):
    statement: str = Field(description="One factual statement supported by the cited passages")
    passage_ids: list[str] = Field(description="Ids of the passages that support it, such as P1")


class AnalysisDraft(BaseModel):
    summary: str = Field(description="Short answer to the sub-question based only on the passages")
    findings: list[Finding]
    disagreements: list[str] = Field(description="Points where passages disagree, citing their ids")
    gaps: list[str] = Field(description="What the passages do not establish")


class Claim(BaseModel):
    """A statement to fact-check, linked to the chunks it was drawn from."""

    claim_id: str
    sub_question_id: str
    text: str
    chunk_ids: list[str]
    source_ids: list[str]

    @property
    def cited(self) -> bool:
        return bool(self.chunk_ids)


class SubQuestionAnalysis(BaseModel):
    sub_question_id: str
    question: str
    summary: str
    claims: list[Claim]
    disagreements: list[str]
    gaps: list[str]
    passages_used: int
    invalid_citations: int = 0


SYSTEM = (
    "You are the analysis agent of a research assistant. You answer a sub-question using only the "
    "passages provided. Every finding must cite the ids of the passages that support it. If the "
    "passages disagree, say so; if they do not answer part of the question, list it as a gap. Never "
    "add facts that are not in the passages. Write each statement as a standalone fact: put passage "
    "ids only in passage_ids, never in the statement text. " + UNTRUSTED_CONTENT_RULE
)


def build_prompt(sub_question: SubQuestion, labeled: list[LabeledPassage], language: str) -> str:
    return (f"Sub-question: {sub_question.question}\n\nPassages:\n{format_passages(labeled)}\n\n"
            f"Write in {language_name(language)}. Give 2 to 6 findings, each citing passage ids.")


def _clean_list(items: list[str]) -> list[str]:
    return [text for text in (strip_passage_refs(x)[0] for x in items) if text]


class Analyst:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def run(self, sub_question: SubQuestion, passages: list[Passage], language: str = "en",
            claim_offset: int = 0) -> SubQuestionAnalysis:
        if not passages:
            return SubQuestionAnalysis(sub_question_id=sub_question.id, question=sub_question.question,
                                       summary="", claims=[], disagreements=[],
                                       gaps=["No relevant passage was found for this sub-question."],
                                       passages_used=0)
        labeled = label_passages(passages)
        by_label = {item.label: item.passage for item in labeled}
        draft = self.llm.structured(SYSTEM, build_prompt(sub_question, labeled, language), AnalysisDraft)

        claims, invalid = [], 0
        for finding in draft.findings:
            text, mentioned = strip_passage_refs(finding.statement)
            if not text:
                continue
            # ids written in the statement count as citations when the list itself is wrong
            labels = valid_labels(finding.passage_ids, labeled) or valid_labels(mentioned, labeled)
            invalid += len({x.strip().strip("[]") for x in finding.passage_ids}) - len(labels)
            chunks = [by_label[label].chunk for label in labels]
            claims.append(Claim(
                claim_id=f"C{claim_offset + len(claims) + 1}",
                sub_question_id=sub_question.id,
                text=text,
                chunk_ids=[c.chunk_id for c in chunks],
                source_ids=list(dict.fromkeys(c.source_id for c in chunks)),
            ))
        return SubQuestionAnalysis(sub_question_id=sub_question.id, question=sub_question.question,
                                   summary=strip_passage_refs(draft.summary)[0], claims=claims,
                                   disagreements=_clean_list(draft.disagreements),
                                   gaps=_clean_list(draft.gaps),
                                   passages_used=len(passages), invalid_citations=invalid)
