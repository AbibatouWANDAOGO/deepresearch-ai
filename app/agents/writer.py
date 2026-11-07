"""Writer: turns checked claims into a cited report.

Traceability does not depend on the model. Code builds every sourced statement from the
checked claims: each one carries its citations ([S1], [S2]...), the wording of its status
(a partially verified claim is presented as such, a conflicting one as disputed), and links
to the exact passages that support or contradict it (chunk, page or character offsets, and
a short quote). UNVERIFIED claims never appear in the body: they are listed apart, with the
reason.

The model only writes the title, the executive summary and the limitations. Each summary
sentence must name the claims it states (C1, C2...): code drops sentences that name none
of the reportable claims and adds the citations itself. If the model fails or every sentence
is dropped, the summary is built from the verified claims. The model never sees web text.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field

from app.agents.analyst import SubQuestionAnalysis
from app.agents.base import language_name, strip_passage_refs
from app.agents.fact_checker import CheckedClaim, Status
from app.agents.planner import SubQuestion
from app.database.models import Source
from app.llm import LLMClient, LLMError, recoverable
from app.rag.chunking import Chunk
from app.rag.retriever import bm25_tokens
from app.reports.i18n import COMPLETE, localize

CITATION = re.compile(r"\s*\[(S\d+)\]")
REPORTABLE = (Status.VERIFIED, Status.PARTIALLY_VERIFIED, Status.CONFLICTING)
MAX_SUMMARY_SENTENCES = 6
QUOTE_CHARS = 320

WORDING = {
    "fr": {
        Status.PARTIALLY_VERIFIED: "Partiellement confirmé : ",
        Status.CONFLICTING: "Point contesté : ",
        "contradicted": "contredit par",
        "no_claim": "Aucune affirmation de cette partie n'a pu être vérifiée dans les sources collectées.",
        "no_answer": "Les sources collectées ne permettent pas de répondre à la question avec des "
                     "affirmations vérifiées.",
        "chars": "car.",
        "lim_unverified": "{n} affirmation(s) n'ont pas pu être vérifiées : elles sont listées à part et ne "
                          "sont pas présentées comme des faits.",
        "lim_skipped": "{n} d'entre elles n'ont pas été examinées (limite de durée ou d'affirmations par "
                       "sous-question).",
        "lim_all_skipped": "Aucune d'elles n'a été examinée (limite de durée ou d'affirmations par sous-question).",
        "lim_stopped": "La recherche s'est arrêtée avant que chaque sous-question soit étayée ({reason}).",
        "lim_empty": "Aucune affirmation vérifiée pour : {heading}",
        "lim_single": "{n} affirmation(s) du rapport ne reposent que sur une seule source.",
    },
    "en": {
        Status.PARTIALLY_VERIFIED: "Partially supported: ",
        Status.CONFLICTING: "Disputed: ",
        "contradicted": "contradicted by",
        "no_claim": "No claim of this part could be verified in the collected sources.",
        "no_answer": "The collected sources do not answer the question with verified claims.",
        "chars": "chars",
        "lim_unverified": "{n} claim(s) could not be verified: they are listed apart and not presented as facts.",
        "lim_skipped": "{n} of them were not examined (time limit or claim limit per sub-question).",
        "lim_all_skipped": "None of them was examined (time limit or claim limit per sub-question).",
        "lim_stopped": "The research stopped before every sub-question was supported ({reason}).",
        "lim_empty": "No verified claim for: {heading}",
        "lim_single": "{n} claim(s) of the report rest on a single source.",
    },
}


def wording(language: str) -> dict:
    return WORDING.get(language, WORDING["en"])


# --- Report model ------------------------------------------------------------------------


class Citation(BaseModel):
    """One passage cited for a statement, precise enough to find it in the source."""

    label: str
    source_id: str
    chunk_id: str
    role: Literal["supports", "contradicts"]
    location: str = ""  # "p. 3" for PDFs, character range otherwise
    quote: str = ""  # the sentence(s) of the passage closest to the claim
    start_char: int | None = None  # offsets of the quote in the extracted source text
    end_char: int | None = None
    page: int | None = None


class CitedStatement(BaseModel):
    claim_id: str
    sub_question_id: str
    text: str
    status: Status
    labels: list[str]
    contradicting_labels: list[str] = []
    justification: str = ""
    citations: list[Citation] = []
    skipped: bool = False


class ReportSection(BaseModel):
    sub_question_id: str
    heading: str
    statements: list[CitedStatement]
    body: str = Field(description="The statements rendered as Markdown")


class SummarySentence(BaseModel):
    text: str = Field(description="One sentence of the summary, without citation marks")
    claim_ids: list[str] = Field(description="Ids of the claims this sentence states, such as C2")


class ReportDraft(BaseModel):
    title: str
    summary: list[SummarySentence] = Field(description="3 to 5 sentences answering the question")


class BibliographyEntry(BaseModel):
    label: str
    source_id: str
    title: str
    url: str
    site_name: str | None = None
    author: str | None = None
    published_date: str | None = None
    collected_at: str
    source_type: str | None = None


class Report(BaseModel):
    question: str
    language: str = "en"
    title: str
    executive_summary: str
    summary_origin: Literal["model", "fallback"] = "model"
    sections: list[ReportSection]
    uncertainties: list[CitedStatement]  # UNVERIFIED claims, kept out of the body
    limitations: list[str]
    bibliography: list[BibliographyEntry]
    status_counts: dict[str, int]
    removed_sentences: int = 0  # summary sentences dropped because they cited no reportable claim
    research_id: str = ""
    stop_reason: str | None = None
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


# --- Evidence ----------------------------------------------------------------------------

_SENTENCE = re.compile(r"[^.!?\n\f]+(?:[.!?]+|$)")


def best_excerpt(chunk: Chunk, claim_text: str, max_chars: int = QUOTE_CHARS) -> tuple[str, int, int, int | None]:
    """The sentence of the chunk that shares most words with the claim, extended to a
    readable length. Returns (quote, start, end, page) with offsets in the source text."""
    text = chunk.text
    spans = [(m.start(), m.end()) for m in _SENTENCE.finditer(text) if m.group().strip()]
    if not spans:
        return "", chunk.start_char, chunk.start_char, chunk.page_start
    wanted = set(bm25_tokens(claim_text))
    scores = [len(wanted & set(bm25_tokens(text[s:e]))) for s, e in spans]
    best = max(range(len(spans)), key=lambda i: (scores[i], -i))
    start, end = spans[best]
    nxt = best + 1
    while (end - start < 120 and nxt < len(spans) and spans[nxt][1] - start <= max_chars
           and "\f" not in text[end:spans[nxt][1]]):  # stay on one page, so the page number holds
        end = spans[nxt][1]
        nxt += 1
    while text[start].isspace():
        start += 1
    quote = " ".join(text[start:end].split())
    if len(quote) > max_chars:
        quote = quote[:max_chars].rsplit(" ", 1)[0] + "…"
    page = None
    if chunk.page_start is not None:
        page = chunk.page_start + text[:start].count("\f")
    return quote, chunk.start_char + start, chunk.start_char + end, page


def source_of(chunk_id: str, chunks: dict[str, Chunk]) -> str:
    chunk = chunks.get(chunk_id)
    return chunk.source_id if chunk else chunk_id.rsplit("-", 1)[0]


def assign_labels(statements: list[CheckedClaim], chunks: dict[str, Chunk] | None = None,
                  known: set[str] | None = None) -> dict[str, str]:
    """S1, S2... in order of first citation (supporting sources, then contradicting ones),
    numbering only the sources in ``known`` when given."""
    chunks = chunks or {}
    labels: dict[str, str] = {}
    for item in statements:
        contradicting = [source_of(c, chunks) for c in item.contradicting_chunk_ids]
        for source_id in [*item.supporting_source_ids, *contradicting]:
            if known is None or source_id in known:
                labels.setdefault(source_id, f"S{len(labels) + 1}")
    return labels


def build_citations(item: CheckedClaim, labels: dict[str, str], chunks: dict[str, Chunk],
                    language: str) -> list[Citation]:
    citations = []
    for role, chunk_ids in (("supports", item.supporting_chunk_ids), ("contradicts", item.contradicting_chunk_ids)):
        for chunk_id in chunk_ids:
            source_id = source_of(chunk_id, chunks)
            if source_id not in labels:
                continue
            chunk = chunks.get(chunk_id)
            if chunk is None:
                citations.append(Citation(label=labels[source_id], source_id=source_id, chunk_id=chunk_id, role=role))
                continue
            quote, start, end, page = best_excerpt(chunk, item.claim.text)
            location = f"p. {page}" if page is not None else f"{wording(language)['chars']} {start}-{end}"
            citations.append(Citation(label=labels[source_id], source_id=source_id, chunk_id=chunk_id, role=role,
                                      location=location, quote=quote, start_char=start, end_char=end, page=page))
    return citations


_CLAIM_LIST = r"(?:(?:claims?|affirmations?)\s+)?C\d+(?:\s*(?:,|et|and|&)\s*C\d+)*"
_CLAIM_REFS = [
    re.compile(rf"\s*[\[(]\s*{_CLAIM_LIST}\s*[\])]", re.IGNORECASE),  # "(C1, C2)"
    re.compile(rf"\b{_CLAIM_LIST}\b\s*[:\u2013\u2014-]?\s*", re.IGNORECASE),  # "C1, C2 : ..." or a bare "C3"
]
_PASSAGE_LIST = re.compile(r"\bP\d+(?:\s*(?:,|et|and|&)\s*P\d+)*\b")


def clean_model_text(text: str) -> str:
    """Model prose without citation marks or claim ids of its own (code adds the citations)."""
    text = CITATION.sub("", text)
    for pattern in _CLAIM_REFS:
        text = pattern.sub(" ", text)
    return strip_passage_refs(text)[0]


def relabel_passages(text: str, passage_map: dict[str, str], labels: dict[str, str],
                     chunks: dict[str, Chunk]) -> str:
    """'Passage P2 states...' -> 'Passage [S3] states...': prompt ids become report labels.
    Ids that cannot be resolved are removed."""
    def replace(match: re.Match) -> str:
        found = []
        for passage in re.findall(r"P\d+", match[0]):
            chunk_id = passage_map.get(passage)
            label = labels.get(source_of(chunk_id, chunks)) if chunk_id else None
            if label:
                found.append(label)
        return ", ".join(f"[{label}]" for label in by_number(found))
    text = _PASSAGE_LIST.sub(replace, text)
    text = re.sub(r"\s+([.,;:])", r"\1", text)
    return re.sub(r"\s{2,}", " ", text).strip()


_NUMBER = re.compile(r"\d+(?:[.,]\d+)?")
MIN_SUMMARY_OVERLAP = 0.6


def _stems(text: str) -> set[str]:
    """Word stems (first 6 letters) so 'limitée' matches 'limitées' and 'inconvénient' matches
    'inconvénients'. Numbers are checked separately."""
    return {t[:6] for t in bm25_tokens(text) if not t.isdigit()}


def _numbers(text: str) -> set[str]:
    return {n.replace(",", ".") for n in _NUMBER.findall(text)}


def grounded(sentence: str, claims: list[str], framing: str) -> bool:
    """A summary sentence may only restate the claims it names: every number must come from
    them, and most of its words from them or from the question (``framing``)."""
    source = " ".join(claims)
    if not _numbers(sentence) <= _numbers(source):
        return False
    words = _stems(sentence)
    if not words:
        return False
    known = _stems(source) | _stems(framing)
    return len(words & known) / len(words) >= MIN_SUMMARY_OVERLAP


def by_number(labels: list[str]) -> list[str]:
    return sorted(dict.fromkeys(labels), key=lambda label: int(label[1:]))


def attach(text: str, labels: list[str], suffix: str = "") -> str:
    """'Prices fell.' + [S2, S1] -> 'Prices fell [S1] [S2].'"""
    text = text.strip()
    end = ""
    if text and text[-1] in ".!?":
        text, end = text[:-1].rstrip(), text[-1]
    cites = "".join(f" [{label}]" for label in by_number(labels))
    return f"{text}{cites}{suffix}{end or '.'}"


def render_statement(statement: CitedStatement, language: str) -> str:
    words = wording(language)
    prefix = words.get(statement.status, "")
    suffix = ""
    if statement.contradicting_labels:
        suffix = f" ; {words['contradicted']}" if language == "fr" else f"; {words['contradicted']}"
        suffix += "".join(f" [{label}]" for label in statement.contradicting_labels)
    return prefix + attach(statement.text, statement.labels, suffix)


# --- Writer ------------------------------------------------------------------------------

SYSTEM = (
    "You are the writing agent of a research assistant. You write the title and a short executive "
    "summary of a report, using only the checked claims provided. Each summary "
    "sentence must list the ids of the claims it states (such as C2) and must not go beyond them. "
    "Present PARTIALLY_VERIFIED claims with their nuance and CONFLICTING claims as disputed. Never "
    "state the claims listed as unverified. Do not write citation marks, sources, numbers, dates or "
    "authors that are not in the claims."
)


def build_prompt(question: str, sub_questions: list[SubQuestion], reportable: list[CheckedClaim],
                 unverified: list[CheckedClaim], gaps: list[str], language: str) -> str:
    lines = []
    for sq in sub_questions:
        items = [c for c in reportable if c.claim.sub_question_id == sq.id]
        if items:
            lines.append(f"\n{sq.id}: {sq.question}")
            lines += [f"- {c.claim.claim_id} [{c.status.value}] {c.claim.text}" for c in items]
    if unverified:
        lines.append("\nUnverified claims (do not state them):")
        lines += [f"- {c.claim.text}" for c in unverified]
    if gaps:
        lines.append("\nKnown gaps:")
        lines += [f"- {g}" for g in gaps]
    return (f"Research question: {question}\n\nChecked claims:{''.join(chr(10) + x for x in lines)}\n\n"
            f"Write in {language_name(language)}: a title and 3 to 5 summary sentences that answer the "
            "question. Give the ids of the claims each sentence states in claim_ids, not in the text.")


def _dedupe(items: list[str]) -> list[str]:
    seen, kept = set(), []
    for item in items:
        key = item.casefold().rstrip(". ")
        if item and key not in seen:
            seen.add(key)
            kept.append(item)
    return kept


class Writer:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def run(self, question: str, sub_questions: list[SubQuestion], analyses: list[SubQuestionAnalysis],
            checked: list[CheckedClaim], sources: dict[str, Source], chunks: dict[str, Chunk] | None = None,
            language: str = "en", stop_reason: str | None = None) -> Report:
        chunks = chunks or {}
        rank = {Status.VERIFIED: 0, Status.PARTIALLY_VERIFIED: 1, Status.CONFLICTING: 2}
        order = [sq.id for sq in sub_questions]
        reportable = sorted((c for c in checked if c.status in REPORTABLE),
                            key=lambda c: (order.index(c.claim.sub_question_id) if c.claim.sub_question_id in order
                                           else len(order), rank[c.status]))
        unverified = [c for c in checked if c.status not in REPORTABLE]
        labels = assign_labels(reportable, chunks, known=set(sources))

        def statement(item: CheckedClaim) -> CitedStatement:
            contradicting = list(dict.fromkeys(source_of(c, chunks) for c in item.contradicting_chunk_ids))
            return CitedStatement(
                claim_id=item.claim.claim_id, sub_question_id=item.claim.sub_question_id,
                text=strip_passage_refs(item.claim.text)[0] or item.claim.text, status=item.status,
                justification=relabel_passages(item.justification, item.passage_map, labels, chunks),
                labels=by_number([labels[s] for s in item.supporting_source_ids if s in labels]),
                contradicting_labels=by_number([labels[s] for s in contradicting if s in labels]),
                citations=build_citations(item, labels, chunks, language) if item.status in REPORTABLE else [],
                skipped=item.skipped,
            )

        statements = {c.claim.claim_id: statement(c) for c in reportable}
        sections = []
        for sq in sub_questions:
            items = [statements[c.claim.claim_id] for c in reportable if c.claim.sub_question_id == sq.id]
            body = "\n".join(f"- {render_statement(s, language)}" for s in items) or wording(language)["no_claim"]
            sections.append(ReportSection(sub_question_id=sq.id, heading=sq.question, statements=items, body=body))

        gaps = _dedupe([g for a in analyses for g in a.gaps])
        title, summary, origin, removed = question, "", "fallback", 0
        if reportable:
            try:
                draft = self.llm.structured(SYSTEM, build_prompt(question, sub_questions, reportable, unverified,
                                                                 gaps, language), ReportDraft)
            except LLMError as exc:
                if not recoverable(exc):
                    raise
                draft = None
            if draft is not None:
                title = clean_model_text(draft.title) or question
                sentences = []
                for sentence in draft.summary[:MAX_SUMMARY_SENTENCES]:
                    text = clean_model_text(sentence.text)
                    ids = [i.strip() for i in sentence.claim_ids if i.strip() in statements]
                    framing = " ".join([question, *(sq.question for sq in sub_questions)])
                    if not text or not ids or not grounded(text, [statements[i].text for i in ids], framing):
                        removed += 1
                        continue
                    cited = list(dict.fromkeys(label for i in ids for label in statements[i].labels))
                    sentences.append(attach(text, cited))
                if sentences:
                    summary, origin = " ".join(sentences), "model"
        if not summary:
            summary = self._fallback_summary(sections, language)

        used = {label for s in statements.values() for label in [*s.labels, *s.contradicting_labels]}
        bibliography = [
            BibliographyEntry(label=label, source_id=sid, title=sources[sid].title or sources[sid].url,
                              url=sources[sid].url, site_name=sources[sid].site_name, author=sources[sid].author,
                              published_date=sources[sid].published_date,
                              collected_at=sources[sid].collected_at.date().isoformat(),
                              source_type=sources[sid].source_type)
            for sid, label in sorted(labels.items(), key=lambda kv: int(kv[1][1:])) if label in used
        ]
        return Report(
            question=question, language=language, title=title, executive_summary=summary, summary_origin=origin,
            sections=sections, uncertainties=[statement(c) for c in unverified],
            limitations=_dedupe(self._limitations(sections, unverified, list(statements.values()), stop_reason,
                                                  language) + gaps),
            bibliography=bibliography, status_counts={s.value: sum(c.status == s for c in checked) for s in Status},
            removed_sentences=removed, stop_reason=stop_reason,
        )

    @staticmethod
    def _limitations(sections: list[ReportSection], unverified: list[CheckedClaim],
                     statements: list[CitedStatement], stop_reason: str | None, language: str) -> list[str]:
        """Limits stated from what the research knows about itself, never from the model."""
        words, out = wording(language), []
        if stop_reason and stop_reason != COMPLETE:
            out.append(words["lim_stopped"].format(reason=localize(stop_reason, language)))
        if unverified:
            out.append(words["lim_unverified"].format(n=len(unverified)))
            skipped = sum(c.skipped for c in unverified)
            if skipped:
                out.append(words["lim_skipped"].format(n=skipped) if skipped < len(unverified)
                           else words["lim_all_skipped"])
        out += [words["lim_empty"].format(heading=s.heading) for s in sections if not s.statements]
        single = sum(len(s.labels) == 1 for s in statements)
        if single:
            out.append(words["lim_single"].format(n=single))
        return out

    @staticmethod
    def _fallback_summary(sections: list[ReportSection], language: str) -> str:
        """One statement per section, the strongest first: verified before partial."""
        picks = [section.statements[0] for section in sections if section.statements
                 and section.statements[0].status != Status.CONFLICTING]
        return " ".join(render_statement(s, language) for s in picks) or wording(language)["no_answer"]
