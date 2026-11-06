"""Writer: drafts the report from checked claims only, citing sources as [S1], [S2]...

The model only sees claims with their verification status and source labels, never raw
web text, so it cannot pick up instructions from pages. After drafting, code removes
citations to unknown sources, builds the bibliography from the sources actually cited, and
lists every claim that is not VERIFIED as an uncertainty.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, Field

from app.agents.analyst import SubQuestionAnalysis
from app.agents.base import language_name
from app.agents.fact_checker import CheckedClaim, Status
from app.database.models import Source
from app.llm import LLMClient

CITATION = re.compile(r"\[(S\d+)\]")


class ReportSection(BaseModel):
    heading: str
    body: str = Field(description="Markdown paragraphs citing sources as [S1], [S2]")


class ReportDraft(BaseModel):
    title: str
    executive_summary: str = Field(description="3 to 5 sentences answering the question, with citations")
    sections: list[ReportSection]
    limitations: list[str] = Field(description="Limits of the evidence and of this research")


class BibliographyEntry(BaseModel):
    label: str
    title: str
    url: str
    published_date: str | None = None
    collected_at: str
    source_type: str | None = None


class Report(BaseModel):
    question: str
    title: str
    executive_summary: str
    sections: list[ReportSection]
    limitations: list[str]
    uncertainties: list[str]
    bibliography: list[BibliographyEntry]
    status_counts: dict[str, int]
    removed_citations: int = 0


SYSTEM = (
    "You are the writing agent of a research assistant. You write a clear, structured report that "
    "answers the research question using only the checked claims provided. Cite sources with their "
    "labels in square brackets, e.g. [S2]. State VERIFIED claims as facts. Present "
    "PARTIALLY_VERIFIED claims with their nuance, CONFLICTING claims as disputed with both sides, "
    "and never present UNVERIFIED claims as facts. Do not invent sources, numbers, dates or authors."
)


def build_prompt(question: str, analyses: list[SubQuestionAnalysis], checked: list[CheckedClaim],
                 labels: dict[str, str], language: str) -> str:
    lines = []
    for analysis in analyses:
        lines.append(f"\n## {analysis.sub_question_id}: {analysis.question}")
        for item in (c for c in checked if c.claim.sub_question_id == analysis.sub_question_id):
            cites = " ".join(f"[{labels[s]}]" for s in item.supporting_source_ids if s in labels)
            lines.append(f"- [{item.status.value}] {item.claim.text} {cites}".rstrip())
        for gap in analysis.gaps:
            lines.append(f"- [GAP] {gap}")
        for disagreement in analysis.disagreements:
            lines.append(f"- [DISAGREEMENT] {disagreement}")
    return (f"Research question: {question}\n\nChecked claims by sub-question:{''.join(chr(10) + x for x in lines)}\n\n"
            f"Write the report in {language_name(language)}: a title, an executive summary, one section "
            "per sub-question, and the limitations.")


def assign_labels(checked: list[CheckedClaim]) -> dict[str, str]:
    """S1, S2... in order of first citation."""
    labels: dict[str, str] = {}
    for item in checked:
        for source_id in item.supporting_source_ids:
            labels.setdefault(source_id, f"S{len(labels) + 1}")
    return labels


class Writer:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def run(self, question: str, analyses: list[SubQuestionAnalysis], checked: list[CheckedClaim],
            sources: dict[str, Source], language: str = "en") -> Report:
        labels = assign_labels(checked)
        draft = self.llm.structured(SYSTEM, build_prompt(question, analyses, checked, labels, language), ReportDraft)

        known = set(labels.values())
        removed = 0

        def clean(text: str) -> str:
            nonlocal removed

            def keep(match: re.Match) -> str:
                nonlocal removed
                if match[1] in known:
                    return match[0]
                removed += 1
                return ""
            text = CITATION.sub(keep, text)
            text = re.sub(r"(?<=\S) {2,}(?=\S)", " ", text)  # gap left by a removed citation
            return re.sub(r" +([.,;:])", r"\1", text).strip()

        summary = clean(draft.executive_summary)
        sections = [ReportSection(heading=s.heading.strip(), body=clean(s.body)) for s in draft.sections]
        used = {m for text in [summary, *(s.body for s in sections)] for m in CITATION.findall(text)}
        by_label = {label: source_id for source_id, label in labels.items()}
        bibliography = [
            BibliographyEntry(label=label, title=sources[sid].title or sources[sid].url, url=sources[sid].url,
                              published_date=sources[sid].published_date,
                              collected_at=sources[sid].collected_at.date().isoformat(),
                              source_type=sources[sid].source_type)
            for label, sid in sorted(by_label.items(), key=lambda kv: int(kv[0][1:]))
            if label in used and sid in sources
        ]
        uncertainties = [f"[{c.status.value}] {c.claim.text}" for c in checked if c.status != Status.VERIFIED]
        counts = {s.value: sum(c.status == s for c in checked) for s in Status}
        return Report(question=question, title=draft.title.strip(), executive_summary=summary,
                      sections=sections, limitations=[x for x in draft.limitations if x.strip()],
                      uncertainties=uncertainties, bibliography=bibliography, status_counts=counts,
                      removed_citations=removed)
