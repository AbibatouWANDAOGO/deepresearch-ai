"""Graph nodes. Each reads the state, calls one agent or tool, and returns a state update.

Nodes check the time limit before each expensive call: past the deadline they stop
working and the run goes straight to writing the report with what it has.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

from app.agents.analyst import Analyst, SubQuestionAnalysis
from app.agents.fact_checker import CheckedClaim, FactChecker, Status
from app.agents.planner import Planner, PlannerInput
from app.agents.researcher import Researcher
from app.agents.writer import Writer
from app.config import Settings
from app.graph.routing import decide, time_left, unanswered
from app.graph.state import (
    ResearchState,
    event,
    get_analyses,
    get_checked,
    get_plan,
    get_sub_question,
)
from app.llm import LLMError, recoverable
from app.rag.retriever import HybridRetriever
from app.reports.export import save_report
from app.tools.document_parser import CollectTarget, DocumentCollector


@dataclass
class Dependencies:
    planner: Planner
    researcher: Researcher
    collector: DocumentCollector
    retriever: HybridRetriever
    analyst: Analyst
    fact_checker: FactChecker
    writer: Writer
    settings: Settings
    clock: Callable[[], float] = time.time


class Nodes:
    def __init__(self, deps: Dependencies):
        self.d = deps

    def _late(self, state: ResearchState) -> bool:
        return time_left(state, self.d.settings, self.d.clock()) <= 0

    # --- Planning ------------------------------------------------------------------------

    def plan(self, state: ResearchState) -> dict:
        plan = self.d.planner.run(PlannerInput(question=state["question"], language=state["language"],
                                               max_sub_questions=self.d.settings.max_sub_questions))
        return {
            "plan": plan.model_dump(mode="json"),
            "cycle": 0,
            "queries": {sq.id: sq.search_queries for sq in plan.sub_questions},
            "focus": [sq.id for sq in plan.sub_questions],
            "follow_ups": {},
            "claim_counter": 0,
            "analyses": {},
            "checked": {},
            "events": [event("plan", f"{len(plan.sub_questions)} sub-questions: "
                                     + " | ".join(sq.question for sq in plan.sub_questions))],
        }

    # --- Search and collection -----------------------------------------------------------

    def research(self, state: ResearchState) -> dict:
        known = {s.url for s in self.d.collector.repository.list(state["research_id"])}
        targets, follow_ups, events = [], dict(state.get("follow_ups", {})), []
        for sq_id in state["focus"]:
            if self._late(state):
                events.append(event("research", "time limit reached: remaining searches skipped"))
                break
            sq = get_sub_question(state, sq_id).model_copy(update={"search_queries": state["queries"][sq_id]})
            out = self.d.researcher.run(sq, state["research_id"], known_urls=known)
            known |= {t.url for t in out.targets}
            targets += [t.model_dump(mode="json") for t in out.targets]
            follow_ups[sq_id] = out.follow_up_queries
            events.append(event("research", f"{sq_id}: {len(out.targets)} sources selected from "
                                            f"{len(out.queries)} queries" + (f", {len(out.errors)} errors" if out.errors else "")))
        return {"targets": targets, "follow_ups": follow_ups, "events": events}

    def collect(self, state: ResearchState) -> dict:
        targets = [CollectTarget.model_validate(t) for t in state.get("targets", [])]
        report = self.d.collector.collect(targets, state["research_id"])
        counts = ", ".join(f"{n} {status}" for status, n in sorted(report.counts().items())) or "nothing new"
        return {"events": [event("collect", f"{len(targets)} URLs: {counts}")]}

    def index(self, state: ResearchState) -> dict:
        created = self.d.retriever.index(state["research_id"])
        return {"events": [event("index", f"{created} new chunks indexed")]}

    # --- Analysis and verification -------------------------------------------------------

    def analyze(self, state: ResearchState) -> dict:
        analyses = dict(state.get("analyses", {}))
        counter, events = state.get("claim_counter", 0), []
        for sq_id in state["focus"]:
            if self._late(state):
                events.append(event("analyze", "time limit reached: remaining analyses skipped"))
                break
            sq = get_sub_question(state, sq_id)
            passages = self.d.retriever.retrieve(sq.question, state["research_id"])[: self.d.settings.passages_per_analysis]
            try:
                analysis = self.d.analyst.run(sq, passages, state["language"], claim_offset=counter)
            except LLMError as exc:
                if not recoverable(exc):
                    raise
                analysis = SubQuestionAnalysis(sub_question_id=sq.id, question=sq.question, summary="", claims=[],
                                               disagreements=[], gaps=[f"Analysis failed: {exc}"],
                                               passages_used=len(passages))
                events.append(event("analyze", f"{sq_id}: analysis failed ({exc}), recorded as a gap"))
            counter += len(analysis.claims)
            analyses[sq_id] = analysis.model_dump(mode="json")
            events.append(event("analyze", f"{sq_id}: {len(analysis.claims)} claims from {len(passages)} passages"))
        return {"analyses": analyses, "claim_counter": counter, "events": events}

    def fact_check(self, state: ResearchState) -> dict:
        checked = {k: [c.model_dump(mode="json") for c in v] for k, v in get_checked(state).items()}
        analyses = get_analyses(state)
        limit, events = self.d.settings.max_claims_per_sub_question, []
        for sq_id in state["focus"]:
            if sq_id not in analyses:
                continue
            results = []
            for i, claim in enumerate(analyses[sq_id].claims):
                if i >= limit or self._late(state):
                    reason = "not checked: claim limit per sub-question" if i >= limit else "not checked: time limit"
                    results.append(CheckedClaim(claim=claim, status=Status.UNVERIFIED, supporting_chunk_ids=[],
                                                contradicting_chunk_ids=[], supporting_source_ids=[],
                                                justification=reason, skipped=True))
                    continue
                passages = self.d.retriever.retrieve(claim.text, state["research_id"])[: self.d.settings.passages_per_analysis]
                try:
                    results.append(self.d.fact_checker.run(claim, passages, state["language"]))
                except LLMError as exc:
                    if not recoverable(exc):
                        raise
                    results.append(CheckedClaim(claim=claim, status=Status.UNVERIFIED, supporting_chunk_ids=[],
                                                contradicting_chunk_ids=[], supporting_source_ids=[],
                                                justification=f"not checked: verification failed ({exc})",
                                                skipped=True))
            checked[sq_id] = [r.model_dump(mode="json") for r in results]
            summary = ", ".join(f"{sum(r.status == s for r in results)} {s.value}" for s in Status
                                if any(r.status == s for r in results)) or "no claims"
            events.append(event("fact_check", f"{sq_id}: {summary}"))
        return {"checked": checked, "events": events}

    # --- Routing -------------------------------------------------------------------------

    def route(self, state: ResearchState) -> str:
        return decide(state, self.d.settings, self.d.clock())[0]

    def next_cycle(self, state: ResearchState) -> dict:
        missing = unanswered(state, self.d.settings)
        queries = {}
        for sq_id in missing:
            sq = get_sub_question(state, sq_id)
            queries[sq_id] = state.get("follow_ups", {}).get(sq_id) or [sq.question]
        cycle = state.get("cycle", 0) + 1
        reason = decide(state, self.d.settings, self.d.clock())[1]
        return {"cycle": cycle, "focus": missing, "queries": queries,
                "events": [event("route", f"extra research cycle {cycle}: {reason}")]}

    # --- Report --------------------------------------------------------------------------

    def write(self, state: ResearchState) -> dict:
        _, reason = decide(state, self.d.settings, self.d.clock())
        rid = state["research_id"]
        plan = get_plan(state)
        analyses = get_analyses(state)
        checked = get_checked(state)
        ordered: list[SubQuestionAnalysis] = [analyses[sq.id] for sq in plan.sub_questions if sq.id in analyses]
        all_checked = [c for sq in plan.sub_questions for c in checked.get(sq.id, [])]
        sources = {s.source_id: s for s in self.d.collector.repository.list(rid)}
        report = self.d.writer.run(state["question"], plan.sub_questions, ordered, all_checked, sources,
                                   self.d.retriever.chunk_map(rid), state["language"], stop_reason=reason)
        report = report.model_copy(update={"research_id": rid})
        files, error = save_report(report, self.d.settings.reports_dir / rid)
        message = (f"report written ({reason}); {len(report.bibliography)} sources cited; "
                   f"saved as {', '.join(files)}" + (f"; {error}" if error else ""))
        return {"report": report.model_dump(mode="json"), "stop_reason": reason,
                "events": [event("write", message)]}
