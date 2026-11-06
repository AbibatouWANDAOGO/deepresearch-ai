from app.agents.planner import Planner, PlannerInput, ResearchPlan, SubQuestion
from tests.fakes import FakeLLM


def plan(*sub_questions, objective="Explain battery prices", criteria=("recent data",)):
    return ResearchPlan(objective=objective, sub_questions=list(sub_questions), quality_criteria=list(criteria))


def test_plan_is_normalized():
    llm = FakeLLM(plan(
        SubQuestion(id="x", question="  How did   prices change? ", search_queries=["battery prices", "Battery Prices", " "]),
        SubQuestion(id="y", question="What drives costs?", search_queries=[]),
    ))
    result = Planner(llm).run(PlannerInput(question="Why are batteries cheaper?", language="fr"))
    assert [sq.id for sq in result.sub_questions] == ["Q1", "Q2"]
    assert result.sub_questions[0].question == "How did prices change?"
    assert result.sub_questions[0].search_queries == ["battery prices"]  # duplicate and blank removed
    assert result.sub_questions[1].search_queries == ["What drives costs?"]  # never left without a query
    assert "French" in llm.prompts[0][1]


def test_sub_question_count_is_capped():
    many = [SubQuestion(id=str(i), question=f"Question {i}?", search_queries=[f"q{i}"]) for i in range(10)]
    result = Planner(FakeLLM(plan(*many))).run(PlannerInput(question="Q", max_sub_questions=3))
    assert len(result.sub_questions) == 3


def test_empty_plan_falls_back_to_the_question():
    result = Planner(FakeLLM(plan(objective=""))).run(PlannerInput(question="Is solar power cheap?"))
    assert result.sub_questions[0].question == "Is solar power cheap?"
    assert result.objective == "Is solar power cheap?"
