"""Planner: turns a research question into sub-questions, search queries and quality criteria."""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.agents.base import language_name
from app.llm import LLMClient
from app.tools.web_search import MAX_QUERY_LENGTH


class SubQuestion(BaseModel):
    id: str = Field(description="Identifier such as Q1")
    question: str = Field(description="A precise sub-question that can be answered from sources")
    search_queries: list[str] = Field(description="1 to 3 short web search queries for this sub-question")


class ResearchPlan(BaseModel):
    objective: str = Field(description="What the final report must establish, in one sentence")
    sub_questions: list[SubQuestion]
    quality_criteria: list[str] = Field(description="What good sources and a good answer look like")


class PlannerInput(BaseModel):
    question: str
    language: str = "en"
    max_sub_questions: int = Field(default=4, ge=1, le=8)


SYSTEM = (
    "You are the planning agent of a research assistant. You break a research question into a few "
    "complementary sub-questions that together answer it, and you write short web search queries "
    "that favor primary sources (official documentation, scientific papers, public institutions, "
    "company reports, original datasets). You never answer the question yourself."
)


def build_prompt(task: PlannerInput) -> str:
    return (
        f"Research question: {task.question}\n\n"
        f"Write a plan with at most {task.max_sub_questions} sub-questions, each with 1 to 3 search "
        f"queries of a few words. Write the objective, sub-questions and criteria in "
        f"{language_name(task.language)}; search queries may be in English when that finds better "
        f"sources. Give 2 to 4 quality criteria."
    )


def normalize_plan(plan: ResearchPlan, task: PlannerInput) -> ResearchPlan:
    """Enforce limits the model may ignore: count, ids, non-empty and unique queries."""
    sub_questions = []
    for sq in plan.sub_questions:
        question = " ".join(sq.question.split())
        if not question:
            continue
        queries: list[str] = []
        for q in sq.search_queries:
            q = " ".join(q.split())[:MAX_QUERY_LENGTH]
            if q and q.lower() not in {x.lower() for x in queries}:
                queries.append(q)
        sub_questions.append(SubQuestion(id="", question=question, search_queries=queries[:3] or [question[:MAX_QUERY_LENGTH]]))
        if len(sub_questions) == task.max_sub_questions:
            break
    if not sub_questions:  # a plan must always contain at least the original question
        sub_questions = [SubQuestion(id="", question=task.question, search_queries=[task.question[:MAX_QUERY_LENGTH]])]
    for i, sq in enumerate(sub_questions, start=1):
        sq.id = f"Q{i}"
    return ResearchPlan(objective=plan.objective.strip() or task.question,
                        sub_questions=sub_questions,
                        quality_criteria=[c.strip() for c in plan.quality_criteria if c.strip()][:4])


class Planner:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def run(self, task: PlannerInput) -> ResearchPlan:
        plan = self.llm.structured(SYSTEM, build_prompt(task), ResearchPlan)
        return normalize_plan(plan, task)
