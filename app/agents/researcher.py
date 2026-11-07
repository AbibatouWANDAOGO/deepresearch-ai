"""Researcher: runs the searches of a sub-question and picks the sources worth collecting."""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.agents.base import UNTRUSTED_CONTENT_RULE, escape_untrusted
from app.agents.planner import SubQuestion
from app.llm import LLMClient, LLMError, recoverable
from app.tools.document_parser import CollectTarget
from app.tools.web_search import SearchError, SearchResult, WebSearch, normalize_url


class SelectedSource(BaseModel):
    label: str = Field(description="Label of the candidate, such as R2")
    primary: bool = Field(description="True for a primary source (official, scientific, original data)")
    reason: str = Field(description="Why this source helps answer the sub-question, in a few words")


class SourceSelection(BaseModel):
    selected: list[SelectedSource]
    follow_up_queries: list[str] = Field(description="0 to 2 queries for aspects the results do not cover")


class ResearcherOutput(BaseModel):
    sub_question_id: str
    queries: list[str]
    targets: list[CollectTarget]
    primary_labels: list[str] = []
    follow_up_queries: list[str] = []
    errors: list[str] = []


SYSTEM = (
    "You are the research agent of a research assistant. From web search results you select the "
    "sources most likely to answer a sub-question reliably. Prefer primary sources (official "
    "documentation, scientific publications, public institutions, company reports, original data) "
    "over blogs, forums and marketing pages. " + UNTRUSTED_CONTENT_RULE
)


# Keep the selection prompt short: on a CPU, prompt length dominates the response time.
MAX_CANDIDATES = 10
SNIPPET_CHARS = 250


def build_prompt(sub_question: SubQuestion, candidates: list[tuple[str, SearchResult]], max_selected: int) -> str:
    lines = [
        f'<document id="{label}" title="{escape_untrusted(r.title).replace(chr(34), chr(39))}" '
        f'url="{r.url}">{escape_untrusted(r.snippet[:SNIPPET_CHARS])}</document>'
        for label, r in candidates
    ]
    return (f"Sub-question: {sub_question.question}\n\nSearch results:\n" + "\n".join(lines)
            + f"\n\nSelect at most {max_selected} results by label. If the results miss an important "
            "aspect, propose up to 2 complementary search queries.")


class Researcher:
    def __init__(self, llm: LLMClient, search: WebSearch, max_results: int = 5, max_selected: int = 3):
        self.llm, self.search = llm, search
        self.max_results, self.max_selected = max_results, max_selected

    def run(self, sub_question: SubQuestion, research_id: str, known_urls: set[str] | None = None) -> ResearcherOutput:
        known = {normalize_url(u) for u in (known_urls or set())}
        candidates: list[SearchResult] = []
        seen: set[str] = set()
        errors: list[str] = []
        for query in sub_question.search_queries:
            try:
                response = self.search.search(query, self.max_results, research_id)
            except (SearchError, ValueError) as exc:
                errors.append(f"{query!r}: {exc}")
                continue
            errors.extend(response.errors)
            for result in response.results:
                key = normalize_url(result.url)
                if key and key not in seen and key not in known:
                    seen.add(key)
                    candidates.append(result)

        output = ResearcherOutput(sub_question_id=sub_question.id, queries=sub_question.search_queries,
                                  targets=[], errors=errors)
        if not candidates:
            return output

        labeled = [(f"R{i}", r) for i, r in enumerate(candidates[:MAX_CANDIDATES], start=1)]
        by_label = dict(labeled)
        try:
            selection = self.llm.structured(SYSTEM, build_prompt(sub_question, labeled, self.max_selected),
                                            SourceSelection)
        except LLMError as exc:
            if not recoverable(exc):
                raise
            # The search results are still good: keep the engine's ranking.
            output.errors.append(f"source selection failed, using search ranking: {exc}")
            selection = SourceSelection(selected=[], follow_up_queries=[])

        chosen: list[str] = []
        for item in selection.selected:
            label = item.label.strip().strip("[]")
            if label in by_label and label not in chosen:
                chosen.append(label)
        if not chosen:  # the model picked nothing usable: fall back to the search engine's ranking
            chosen = [label for label, _ in labeled]
        chosen = chosen[: self.max_selected]

        output.targets = [CollectTarget.from_search_result(by_label[label]) for label in chosen]
        output.primary_labels = [s.label for s in selection.selected if s.primary and s.label in chosen]
        output.follow_up_queries = [" ".join(q.split()) for q in selection.follow_up_queries if q.strip()][:2]
        return output
