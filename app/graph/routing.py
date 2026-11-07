"""Decisions between nodes: is the evidence sufficient, or should another research cycle run?"""

from __future__ import annotations

import time
from typing import Literal

from app.agents.fact_checker import CheckedClaim, Status
from app.config import Settings
from app.graph.state import ResearchState, get_checked, get_plan

SUPPORTED = {Status.VERIFIED, Status.PARTIALLY_VERIFIED}


def time_left(state: ResearchState, settings: Settings, now: float | None = None) -> float:
    elapsed = (now if now is not None else time.time()) - state["started_at"]
    return settings.max_research_minutes * 60 - elapsed


def is_answered(checked: list[CheckedClaim], min_ratio: float) -> bool:
    """At least one supported claim, and a large enough share of supported claims.

    Claims skipped because of a limit are left out: not checking a claim is no evidence
    that the sub-question lacks support.
    """
    evaluated = [c for c in checked if not c.skipped]
    supported = sum(c.status in SUPPORTED for c in evaluated)
    return supported > 0 and supported / len(evaluated) >= min_ratio


def unanswered(state: ResearchState, settings: Settings) -> list[str]:
    checked = get_checked(state)
    return [sq.id for sq in get_plan(state).sub_questions
            if not is_answered(checked.get(sq.id, []), settings.min_supported_ratio)]


def decide(state: ResearchState, settings: Settings, now: float | None = None) -> tuple[Literal["next_cycle", "write"], str]:
    """Route after fact checking, with the reason recorded in the state."""
    if time_left(state, settings, now) <= 0:
        return "write", "time limit reached"
    missing = unanswered(state, settings)
    if not missing:
        return "write", "every sub-question is supported by evidence"
    if state.get("cycle", 0) >= settings.max_extra_research_cycles:
        return "write", f"research cycle limit reached ({settings.max_extra_research_cycles} extra cycles)"
    return "next_cycle", f"insufficient evidence for {', '.join(missing)}"
