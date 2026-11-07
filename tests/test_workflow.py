import json
import sqlite3

import pytest
from langgraph.checkpoint.sqlite import SqliteSaver

from app.agents.analyst import AnalysisDraft, Analyst, Finding
from app.agents.fact_checker import FactChecker, Status, VerdictDraft
from app.agents.planner import Planner, ResearchPlan, SubQuestion
from app.agents.researcher import ResearcherOutput
from app.agents.writer import ReportDraft, ReportSection, Writer
from app.config import Settings
from app.graph.nodes import Dependencies
from app.graph.workflow import ResearchRunner
from app.llm import CostTracker
from app.tools.document_parser import CollectTarget
from tests.fakes import make_passage, make_source

# --- Fakes -------------------------------------------------------------------------------


class RuleLLM:
    """Answers by schema; the fact-check verdict comes from a function of the claim text."""

    name, model = "fake", "fake-model"

    def __init__(self, verdict, findings_per_analysis=2):
        self.verdict, self.findings = verdict, findings_per_analysis
        self.calls: dict[str, int] = {}
        self.tracker = CostTracker(0.0)
        self.crash_on: str | None = None

    def structured(self, system, user, schema):
        name = schema.__name__
        self.calls[name] = self.calls.get(name, 0) + 1
        if self.crash_on == name:
            self.crash_on = None
            raise RuntimeError("simulated crash")
        if name == "ResearchPlan":
            return ResearchPlan(objective="Understand alpha and beta", quality_criteria=["primary sources"], sub_questions=[
                SubQuestion(id="", question="What is alpha?", search_queries=["alpha"]),
                SubQuestion(id="", question="What is beta?", search_queries=["beta"])])
        if name == "AnalysisDraft":
            topic = "alpha" if "alpha" in user.split("Passages:")[0] else "beta"
            return AnalysisDraft(summary=f"About {topic}.", disagreements=[], gaps=[], findings=[
                Finding(statement=f"{topic} fact {i}", passage_ids=["P1"]) for i in range(self.findings)])
        if name == "VerdictDraft":
            claim = user.split("\n", 1)[0]
            status = self.verdict(claim, self.calls[name])
            return VerdictDraft(status=status, contradicting_ids=[], justification="checked",
                                supporting_ids=["P1"] if status != Status.UNVERIFIED else [])
        if name == "ReportDraft":
            return ReportDraft(title="Report", executive_summary="Alpha and beta explained [S1].",
                               sections=[ReportSection(heading="Findings", body="See [S1].")], limitations=[])
        raise AssertionError(name)


class FakeRepository:
    def __init__(self):
        self.sources = {}

    def list(self, research_id):
        return list(self.sources.values())


class FakeCollector:
    def __init__(self):
        self.repository, self.calls = FakeRepository(), []

    def collect(self, targets, research_id):
        from app.tools.document_parser import CollectionReport

        self.calls.append([t.url for t in targets])
        for t in targets:
            sid = t.url.split("//")[1].split(".")[0]
            self.repository.sources[sid] = make_source(sid, url=t.url)
        return CollectionReport(research_id=research_id, sources=list(self.repository.sources.values()))


class FakeResearcher:
    def __init__(self):
        self.calls = []

    def run(self, sq, research_id, known_urls=None):
        self.calls.append((sq.id, list(sq.search_queries)))
        topic = "alpha" if "alpha" in sq.question else "beta"
        n = len(self.calls)
        return ResearcherOutput(sub_question_id=sq.id, queries=sq.search_queries,
                                targets=[CollectTarget(url=f"https://{topic}{n}.org")],
                                follow_up_queries=[f"{topic} more evidence {n}"])


class FakeRetriever:
    def __init__(self, collector):
        self.collector = collector

    def index(self, research_id):
        return len(self.collector.repository.sources)

    def retrieve(self, query, research_id):
        topic = "alpha" if "alpha" in query else "beta"
        return [make_passage(sid, 0, f"{topic} evidence from {sid}")
                for sid in self.collector.repository.sources if sid.startswith(topic)]


def runner_with(llm, tmp_path, clock=None, **settings_overrides):
    settings = Settings(_env_file=None, storage_dir=tmp_path, **settings_overrides)
    collector = FakeCollector()
    deps = Dependencies(planner=Planner(llm), researcher=FakeResearcher(), collector=collector,
                        retriever=FakeRetriever(collector), analyst=Analyst(llm), fact_checker=FactChecker(llm),
                        writer=Writer(llm), settings=settings)
    if clock is not None:
        deps.clock = clock
    saver = SqliteSaver(sqlite3.connect(tmp_path / "checkpoints.db", check_same_thread=False))
    return ResearchRunner(deps, saver)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in Settings.model_fields:
        monkeypatch.delenv(name.upper(), raising=False)


def always(status):
    return lambda claim, n: status


# --- Scenarios ---------------------------------------------------------------------------


def test_sufficient_evidence_goes_straight_to_the_report(tmp_path):
    llm = RuleLLM(always(Status.VERIFIED))
    runner = runner_with(llm, tmp_path)
    state = runner.start("Explain alpha and beta", "en", research_id="r1")
    assert state["cycle"] == 0 and state["report"]["title"] == "Report"
    assert state["stop_reason"] == "every sub-question is supported by evidence"
    assert [e["node"] for e in state["events"]] == ["plan", "research", "research", "collect", "index",
                                                    "analyze", "analyze", "fact_check", "fact_check", "write"]
    assert llm.calls == {"ResearchPlan": 1, "AnalysisDraft": 2, "VerdictDraft": 4, "ReportDraft": 1}
    assert runner.pending("r1") == ()


def test_only_unanswered_sub_questions_get_another_cycle(tmp_path):
    # beta claims fail in the first pass, then succeed
    llm = RuleLLM(lambda claim, n: Status.UNVERIFIED if "beta" in claim and n <= 4 else Status.VERIFIED)
    runner = runner_with(llm, tmp_path)
    state = runner.start("Explain alpha and beta", research_id="r1")
    researcher = runner.deps.researcher
    assert state["cycle"] == 1
    assert researcher.calls[-1] == ("Q2", ["beta more evidence 2"])  # the follow-up query is used
    assert llm.calls["AnalysisDraft"] == 3  # Q1 once, Q2 twice
    assert state["stop_reason"] == "every sub-question is supported by evidence"
    assert any("extra research cycle 1: insufficient evidence for Q2" in e["message"] for e in state["events"])


def test_research_stops_after_the_extra_cycle_limit(tmp_path):
    runner = runner_with(RuleLLM(always(Status.UNVERIFIED)), tmp_path, max_extra_research_cycles=2)
    state = runner.start("Explain alpha and beta", research_id="r1")
    assert state["cycle"] == 2
    assert state["stop_reason"] == "research cycle limit reached (2 extra cycles)"
    assert state["report"]["uncertainties"]  # unverified claims are reported as such


def test_time_limit_skips_work_and_writes_what_it_has(tmp_path):
    ticks = iter([0, 0] + [10_000] * 100)  # start and Q1 search on time, deadline passed before Q2
    llm = RuleLLM(always(Status.VERIFIED))
    runner = runner_with(llm, tmp_path, clock=lambda: next(ticks), max_research_minutes=1)
    state = runner.start("Explain alpha and beta", research_id="r1")
    assert state["stop_reason"] == "time limit reached"
    assert "report" in state
    assert any("time limit reached" in e["message"] for e in state["events"] if e["node"] == "research")
    assert "VerdictDraft" not in llm.calls  # no fact check was started after the deadline


def test_claims_beyond_the_limit_are_marked_not_checked(tmp_path):
    llm = RuleLLM(always(Status.VERIFIED), findings_per_analysis=5)
    state = runner_with(llm, tmp_path, max_claims_per_sub_question=2).start("Explain alpha and beta", research_id="r1")
    q1 = state["checked"]["Q1"]
    assert [c["status"] for c in q1] == ["VERIFIED", "VERIFIED", "UNVERIFIED", "UNVERIFIED", "UNVERIFIED"]
    assert q1[2]["justification"] == "not checked: claim limit per sub-question" and q1[2]["skipped"]
    assert llm.calls["VerdictDraft"] == 4
    assert state["cycle"] == 0  # skipped claims do not trigger another research cycle
    assert "[UNVERIFIED] alpha fact 2" in state["report"]["uncertainties"]


def test_claim_ids_stay_unique_across_cycles(tmp_path):
    llm = RuleLLM(lambda claim, n: Status.UNVERIFIED if "beta" in claim and n <= 4 else Status.VERIFIED)
    state = runner_with(llm, tmp_path).start("Explain alpha and beta", research_id="r1")
    ids = [c["claim"]["claim_id"] for checks in state["checked"].values() for c in checks]
    assert len(ids) == len(set(ids))


def test_interrupted_research_resumes_from_its_last_step(tmp_path):
    llm = RuleLLM(always(Status.VERIFIED))
    llm.crash_on = "VerdictDraft"
    runner = runner_with(llm, tmp_path)
    with pytest.raises(RuntimeError, match="simulated crash"):
        runner.start("Explain alpha and beta", research_id="r1")
    assert runner.pending("r1") == ("fact_check",)
    state = runner.resume("r1")
    assert state["report"]["title"] == "Report"
    assert llm.calls["ResearchPlan"] == 1 and llm.calls["AnalysisDraft"] == 2  # nothing redone before fact_check
    with pytest.raises(ValueError, match="nothing left to run"):
        runner.resume("r1")


def only_json_types(value) -> bool:
    """Strictly JSON types: an Enum or a model would be serialized as a Python object."""
    if isinstance(value, dict):
        return all(type(k) is str and only_json_types(v) for k, v in value.items())
    if isinstance(value, list):
        return all(only_json_types(v) for v in value)
    return value is None or type(value) in (str, int, float, bool)


def test_state_is_plain_json(tmp_path):
    runner = runner_with(RuleLLM(always(Status.VERIFIED)), tmp_path)
    state = runner.start("Explain alpha and beta", research_id="r1")
    json.dumps(state)
    # checkpoints never depend on Python classes (LangGraph warns and will refuse them)
    assert only_json_types(state)


def test_streaming_reports_each_step(tmp_path):
    runner = runner_with(RuleLLM(always(Status.VERIFIED)), tmp_path)
    nodes = [node for node, _ in runner.stream("Explain alpha and beta", research_id="r1")]
    assert nodes == ["plan", "research", "collect", "index", "analyze", "fact_check", "write"]


# --- Graceful degradation ----------------------------------------------------------------


class FlakyLLM(RuleLLM):
    """Times out on chosen schemas, like a slow local model."""

    def __init__(self, fail_on, error=None):
        super().__init__(always(Status.VERIFIED))
        self.fail_on, self.error = set(fail_on), error

    def structured(self, system, user, schema):
        if schema.__name__ in self.fail_on:
            from app.llm import LLMError

            raise self.error or LLMError("Ollama timed out after 120.0s")
        return super().structured(system, user, schema)


def test_failed_analysis_becomes_a_gap_and_the_report_is_still_written(tmp_path):
    state = runner_with(FlakyLLM({"AnalysisDraft"}), tmp_path, max_extra_research_cycles=0).start("Explain alpha and beta", research_id="r1")
    assert state["analyses"]["Q1"]["gaps"] == ["Analysis failed: Ollama timed out after 120.0s"]
    assert "report" in state


def test_failed_fact_check_marks_the_claim_as_skipped(tmp_path):
    state = runner_with(FlakyLLM({"VerdictDraft"}), tmp_path, max_extra_research_cycles=0).start("Explain alpha and beta", research_id="r1")
    claim = state["checked"]["Q1"][0]
    assert claim["status"] == "UNVERIFIED" and claim["skipped"]
    assert claim["justification"].startswith("not checked: verification failed")


def test_missing_model_stops_the_research(tmp_path):
    from app.llm import LLMConfigError

    runner = runner_with(FlakyLLM({"AnalysisDraft"}, LLMConfigError("model not installed")), tmp_path)
    with pytest.raises(LLMConfigError):
        runner.start("Explain alpha and beta", research_id="r1")
    assert runner.pending("r1") == ("analyze",)  # can be resumed once the model is installed


def test_resume_restarts_the_duration_limit(tmp_path):
    now = {"t": 0.0}
    llm = RuleLLM(always(Status.VERIFIED))
    llm.crash_on = "AnalysisDraft"
    runner = runner_with(llm, tmp_path, clock=lambda: now["t"], max_research_minutes=1)
    with pytest.raises(RuntimeError):
        runner.start("Explain alpha and beta", research_id="r1")
    now["t"] = 3600.0  # resumed an hour later
    state = runner.resume("r1")
    assert state["started_at"] == 3600.0
    assert state["stop_reason"] != "time limit reached"
