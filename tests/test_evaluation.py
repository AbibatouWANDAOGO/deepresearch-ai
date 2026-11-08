import json

import pytest

from app.agents.analyst import Claim
from app.agents.fact_checker import CheckedClaim, Status
from app.config import Settings
from app.evaluation.judge import Judgment, judge
from app.evaluation.metrics import (
    EvalQuestion,
    KeyPoint,
    aggregate,
    coverage,
    normalize,
    numbers,
    score_question,
    term_found,
)
from app.evaluation.run import (
    annotation_agreement,
    current_id,
    load_questions,
    replay_evidence_rules,
    research_id,
    run_all,
    score_all,
    to_markdown,
)
from app.jobs import JobManager, ResearchParams, ResearchStatus
from tests.fakes import FakeLLM, make_passage, make_source

QUESTION = EvalQuestion.model_validate({
    "id": "q1", "question": "Pourquoi les prix ont-ils baissé ?", "language": "fr",
    "key_points": [{"name": "économies d'échelle", "any_of": ["economies d'echelle", "scale"]},
                   {"name": "chimie", "any_of": ["chimi", "lfp"]},
                   {"name": "COP", "any_of": ["cop"]}],
})


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in Settings.model_fields:
        monkeypatch.delenv(name.upper(), raising=False)


def test_the_questions_file_is_valid():
    questions = load_questions()
    assert len(questions) == 10 and len({q.id for q in questions}) == 10
    assert {q.language for q in questions} == {"fr", "en"}
    assert all(len(q.key_points) >= 4 for q in questions)


def test_terms_match_at_the_start_of_a_word_ignoring_case_and_accents():
    text = normalize("Les ÉCONOMIES d’échelle et la chimie LFP. Scope 3.")
    assert term_found("économies d'échelle", text) and term_found("chimi", text)
    assert not term_found("cop", text)  # 'scope' is not 'COP'
    assert coverage(QUESTION, "La chimie LFP compte.") == (["chimie"], ["économies d'échelle", "COP"])


def test_numbers_ignore_thousands_separators():
    assert numbers("USD 1 400 per kWh in 2010") == numbers("$1,400/kWh (2010)") == {"1400", "2010"}
    assert numbers("2,5 % et 3.5") == {"2.5", "3.5"}


def checked(claim_id, text, origin, status=Status.VERIFIED, support=(), shown=(), skipped=False):
    """A checked claim drawn from chunk ``origin``; ``shown`` are the passages retrieved to check it."""
    return CheckedClaim(
        claim=Claim(claim_id=claim_id, sub_question_id="Q1", text=text, chunk_ids=[origin],
                    source_ids=[origin.split("-")[0]]),
        status=status, supporting_chunk_ids=list(support),
        contradicting_chunk_ids=[], supporting_source_ids=list(dict.fromkeys(c.split("-")[0] for c in support)),
        justification="", passage_map={f"P{i}": c for i, c in enumerate(shown, start=1)}, skipped=skipped,
    )


def state_with(claims):
    statements = [{"text": c.claim.text, "status": c.status.value} for c in claims]
    return {"research_id": "eval-t-q1", "stop_reason": "done",
            "checked": {"Q1": [c.model_dump(mode="json") for c in claims]},
            "analyses": {"Q1": {"summary": "", "claims": [{"text": "Le COP vaut 3."}]}},
            "report": {"executive_summary": "Les économies d'échelle expliquent la baisse.",
                       "sections": [{"statements": statements}], "bibliography": [{"label": "S1"}],
                       "summary_origin": "model", "removed_sentences": 1}}


def test_score_question_measures_retrieval_citations_and_coverage():
    a, b = make_passage("a", 0, "Prices fell 90% since 2010 thanks to economies of scale.").chunk, \
        make_passage("b", 0, "LFP chemistry is cheaper.").chunk
    claims = [
        # supported by its own chunk, found first: numbers ok
        checked("C1", "Prices fell 90% since 2010.", a.chunk_id, support=[a.chunk_id], shown=[a.chunk_id]),
        # supported by another source, but '2020' is in no cited passage; origin ranked second
        checked("C2", "LFP chemistry got cheaper in 2020.", a.chunk_id, support=[b.chunk_id],
                shown=[b.chunk_id, a.chunk_id]),
        checked("C3", "Not supported.", b.chunk_id, status=Status.UNVERIFIED, shown=[a.chunk_id]),
        checked("C4", "Skipped.", b.chunk_id, status=Status.UNVERIFIED, skipped=True),
    ]
    sources = [make_source("a"), make_source("b", url="https://other.org/x"),
               make_source("c").model_copy(update={"access_status": "timeout"})]
    result = score_question(QUESTION, state_with(claims), {a.chunk_id: a, b.chunk_id: b}, sources,
                            run={"seconds": 600, "steps": {"plan": 30}, "usage": {"calls": 9}},
                            verdicts={"C1": "SUPPORTED", "C2": "PARTIAL"})
    assert result["coverage"] == {"report": pytest.approx(0.667, abs=1e-3), "research": 1.0, "missing": ["COP"]}
    assert result["claims"]["checked"] == 3 and result["claims"]["skipped"] == 1
    assert result["claims"]["supported"] == 2 and result["claims"]["UNVERIFIED"] == 1
    assert result["retrieval"] == {"claims": 3, "origin_recall": pytest.approx(0.667, abs=1e-3),
                                   "origin_mrr": pytest.approx(0.5)}
    c = result["citations"]
    assert (c["numbers_ok"], c["independent"], c["precision"]) == (0.5, 0.5, 0.75)
    assert [d["numbers_ok"] for d in c["details"]] == [True, False]
    assert c["passage_precision"] == 1.0 and [d["passages_cited"] for d in c["details"]] == [1, 1]
    assert result["sources"] == {"tried": 3, "usable": 2, "fetch_rate": 0.667, "domains": 2, "chunks": 2, "cited": 1}
    assert result["answered_sub_questions"] == 1.0 and result["latency"]["total_seconds"] == 600


def test_aggregate_pools_claims_and_ignores_unfinished_researches():
    a = make_passage("a", 0, "Prices fell 90% since 2010.").chunk
    done = score_question(QUESTION, state_with([checked("C1", "Prices fell 90%.", a.chunk_id, support=[a.chunk_id],
                                                        shown=[a.chunk_id])]),
                          {a.chunk_id: a}, [make_source("a")], run={"seconds": 300, "steps": {"plan": 20},
                                                                    "usage": {"calls": 4, "input_tokens": 100}},
                          verdicts={"C1": "NOT_SUPPORTED"})
    failed = score_question(QUESTION, {"research_id": "x"}, {}, [], run={"error": "boom", "usage": {"calls": 1}})
    summary = aggregate([done, failed])
    assert (summary["questions"], summary["completed"]) == (2, 1)
    assert summary["citation_precision"] == 0.0 and summary["claims_supported"] == 1
    assert summary["llm_calls"] == 5 and summary["seconds_mean"] == 300
    markdown = to_markdown("t", {"depth": "quick", "max_sources": 5}, summary, [done, failed])
    assert "not completed: boom" in markdown and "| plan | 0.3 min |" in markdown


def test_manual_annotations_give_precision_and_judge_agreement():
    a = make_passage("a", 0, "x").chunk
    result = score_question(QUESTION, state_with([checked("C1", "x.", a.chunk_id, support=[a.chunk_id]),
                                                  checked("C2", "y.", a.chunk_id, support=[a.chunk_id])]),
                            {a.chunk_id: a}, [], verdicts={"C1": "SUPPORTED", "C2": "SUPPORTED"})
    agreement = annotation_agreement([result], {"eval-t-q1": {"C1": "SUPPORTED", "C2": "NOT_SUPPORTED"}})
    # both claims are VERIFIED: the Fact Checker agrees on C1 only
    assert agreement == {"annotated": 2, "citation_precision_annotated": 0.5, "judge_agreement": 0.5,
                         "fact_checker_agreement": 0.5}


def test_replaying_the_text_rules_needs_no_model_call():
    a = make_passage("a", 0, "Pack prices fell 89% thanks to new cathodes.").chunk
    b = make_passage("b", 0, "Lithium mining expanded in Chile.").chunk
    claims = [checked("C1", "Pack prices fell 89% from 2010 thanks to new cathodes, as described in.", a.chunk_id,
                      support=[a.chunk_id, b.chunk_id]),
              checked("C2", "Not checked.", a.chunk_id, status=Status.UNVERIFIED, skipped=True)]
    replayed = replay_evidence_rules(state_with(claims), {a.chunk_id: a, b.chunk_id: b})
    c1, c2 = replayed["checked"]["Q1"]
    assert c1["claim"]["text"] == "Pack prices fell 89% from 2010 thanks to new cathodes."
    assert c1["status"] == "PARTIALLY_VERIFIED" and c1["adjusted"]  # 2010 is not in the passage
    assert c1["supporting_chunk_ids"] == [a.chunk_id] and c1["supporting_source_ids"] == ["a"]
    assert c2["status"] == "UNVERIFIED" and c2["skipped"]


def test_judge_asks_the_model_and_needs_a_passage():
    llm = FakeLLM(Judgment(verdict="PARTIAL", reason="figure differs"))
    assert judge(llm, "Prices fell 90%.", ["Prices fell 80%."]).verdict == "PARTIAL"
    system, user, _ = llm.prompts[0]
    assert "same topic" in system and "Prices fell 80%." in user
    assert judge(FakeLLM(), "x", []).verdict == "NOT_SUPPORTED"  # no model call


# --- Running ---------------------------------------------------------------------------------


@pytest.fixture
def settings(tmp_path):
    s = Settings(_env_file=None, storage_dir=tmp_path)
    s.ensure_storage()
    return s


def test_run_all_runs_resumes_and_skips(settings, tmp_path):
    from tests.test_jobs import FakeRunner

    questions = [QUESTION, QUESTION.model_copy(update={"id": "q2"}), QUESTION.model_copy(update={"id": "q3"})]
    manager = JobManager(settings)
    manager.repo.create("done", "fr", ResearchParams(), research_id=research_id("t", questions[0]),
                        status=ResearchStatus.COMPLETED)
    manager.repo.create("cut", "fr", ResearchParams(), research_id=research_id("t", questions[1]),
                        status=ResearchStatus.INTERRUPTED)
    runners = []

    def factory(job_settings):
        runners.append(FakeRunner(fail=len(runners) == 1))  # the third question crashes
        return runners[-1]

    failed = run_all(settings, questions, "t", ResearchParams(depth="quick"), runner_factory=factory, log=lambda m: None)
    assert failed == []  # one runner is built and reused, so the "crash" runner is never made
    assert len(runners) == 1 and runners[0].calls[0] == ("resume",) and runners[0].calls[1][0] == "stream"
    run = json.loads((settings.storage_dir / "eval" / "t" / "runs" / "q2.json").read_text(encoding="utf-8"))
    assert run["resumed"] and set(run["steps"]) == {"write"} and run["error"] is None
    assert manager.repo.get(research_id("t", questions[2])).status == ResearchStatus.COMPLETED
    assert not (settings.storage_dir / "eval" / "t" / "runs" / "q1.json").exists()


def test_run_all_records_a_failed_question_and_continues(settings):
    from tests.test_jobs import FakeRunner

    questions = [QUESTION, QUESTION.model_copy(update={"id": "q2"})]
    runner = FakeRunner(fail=True)
    failed = run_all(settings, questions, "t", ResearchParams(), runner_factory=lambda s: runner, log=lambda m: None)
    assert failed == ["q1", "q2"]
    run = json.loads((settings.storage_dir / "eval" / "t" / "runs" / "q1.json").read_text(encoding="utf-8"))
    assert run["error"] == "RuntimeError: model crashed"


def test_run_all_refuses_while_a_research_is_running(settings):
    import os

    repo = JobManager(settings).repo
    record = repo.create("Another question here", "en", ResearchParams())
    repo.update(record.research_id, pid=os.getpid())  # a live process
    with pytest.raises(RuntimeError, match="is running"):
        run_all(settings, [QUESTION], "t", ResearchParams(), runner_factory=lambda s: None, log=lambda m: None)


def test_score_all_reads_the_checkpoint_of_each_question(tmp_path):
    from tests.test_workflow import RuleLLM, always, runner_with

    question = QUESTION.model_copy(update={"key_points": QUESTION.key_points + [KeyPoint.model_validate(
        {"name": "alpha", "any_of": ["alpha"]})]})
    runner_with(RuleLLM(always(Status.VERIFIED)), tmp_path).start("Explain alpha and beta",
                                                                  research_id=research_id("t", question))
    settings = Settings(_env_file=None, storage_dir=tmp_path)
    results, summary = score_all(settings, [question, question.model_copy(update={"id": "missing"})], "t")
    assert results[0]["completed"] and "alpha" not in results[0]["coverage"]["missing"]
    assert not results[1]["completed"] and summary["completed"] == 1


def test_rerun_starts_a_new_research_and_keeps_the_old_one(settings):
    from tests.test_jobs import FakeRunner

    repo = JobManager(settings).repo
    run_all(settings, [QUESTION], "t", ResearchParams(), runner_factory=lambda s: FakeRunner(), log=lambda m: None)
    run_all(settings, [QUESTION], "t", ResearchParams(), runner_factory=lambda s: FakeRunner(), log=lambda m: None,
            rerun={"q1"})
    run = json.loads((settings.storage_dir / "eval" / "t" / "runs" / "q1.json").read_text(encoding="utf-8"))
    assert run["research_id"] == "eval-t-q1-2" and run["replaced"] == ["eval-t-q1"]
    assert repo.get("eval-t-q1-2").status == repo.get("eval-t-q1").status == ResearchStatus.COMPLETED
    assert current_id(settings, "t", QUESTION) == "eval-t-q1-2"
