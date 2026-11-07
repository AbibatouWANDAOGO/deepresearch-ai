"""Shared, typed state of a research run.

Everything stored here is plain JSON (dicts, lists, strings, numbers): checkpoints stay
readable and resumable whatever the library versions. Nodes rebuild Pydantic objects
from these dicts with the helpers below.
"""

from __future__ import annotations

import operator
from datetime import UTC, datetime
from typing import Annotated, TypedDict

from app.agents.analyst import SubQuestionAnalysis
from app.agents.fact_checker import CheckedClaim
from app.agents.planner import ResearchPlan, SubQuestion


class Event(TypedDict):
    at: str
    node: str
    message: str


class ResearchState(TypedDict, total=False):
    # Request
    research_id: str
    question: str
    language: str
    started_at: float  # epoch seconds, for the duration limit

    # Progress
    cycle: int  # 0 = first pass, then extra research cycles
    plan: dict  # ResearchPlan
    queries: dict[str, list[str]]  # sub-question id -> queries to search in this cycle
    follow_ups: dict[str, list[str]]  # sub-question id -> complementary queries proposed
    targets: list[dict]  # CollectTarget, this cycle
    focus: list[str]  # sub-question ids handled in this cycle
    claim_counter: int  # next free claim number, so claim ids stay unique across cycles
    analyses: dict[str, dict]  # sub-question id -> SubQuestionAnalysis
    checked: dict[str, list[dict]]  # sub-question id -> CheckedClaim list

    # Outcome
    report: dict  # Report
    stop_reason: str

    # Append-only log shown in the interface
    events: Annotated[list[Event], operator.add]


def event(node: str, message: str) -> Event:
    return {"at": datetime.now(UTC).isoformat(timespec="seconds"), "node": node, "message": message}


def get_plan(state: ResearchState) -> ResearchPlan:
    return ResearchPlan.model_validate(state["plan"])


def get_sub_question(state: ResearchState, sub_question_id: str) -> SubQuestion:
    return next(sq for sq in get_plan(state).sub_questions if sq.id == sub_question_id)


def get_analyses(state: ResearchState) -> dict[str, SubQuestionAnalysis]:
    return {k: SubQuestionAnalysis.model_validate(v) for k, v in state.get("analyses", {}).items()}


def get_checked(state: ResearchState) -> dict[str, list[CheckedClaim]]:
    return {k: [CheckedClaim.model_validate(c) for c in v] for k, v in state.get("checked", {}).items()}
