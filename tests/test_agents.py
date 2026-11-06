"""Analyst and Writer."""

from app.agents.analyst import AnalysisDraft, Analyst, Claim, Finding, SubQuestionAnalysis
from app.agents.fact_checker import CheckedClaim, Status
from app.agents.planner import SubQuestion
from app.agents.writer import ReportDraft, ReportSection, Writer, assign_labels
from tests.fakes import FakeLLM, make_passage, make_source

SQ = SubQuestion(id="Q1", question="How did battery prices change?", search_queries=["battery prices"])
PASSAGES = [make_passage("a", 0, "Pack prices fell 89% between 2010 and 2020.", page=3),
            make_passage("b", 1, "Prices rose in 2022.")]


# --- Analyst -----------------------------------------------------------------------------


def test_findings_become_claims_with_resolved_citations():
    llm = FakeLLM(AnalysisDraft(summary="Prices fell, then rose.", findings=[
        Finding(statement="Prices fell 89% from 2010 to 2020.", passage_ids=["P1", "[P1]"]),
        Finding(statement="Prices rose in 2022.", passage_ids=["P2", "P7"]),
        Finding(statement="Batteries will be free by 2030.", passage_ids=[]),
    ], disagreements=[], gaps=["No data after 2022."]))
    result = Analyst(llm).run(SQ, PASSAGES, language="fr", claim_offset=4)
    assert [c.claim_id for c in result.claims] == ["C5", "C6", "C7"]
    assert result.claims[0].chunk_ids == ["a-0000"] and result.claims[0].source_ids == ["a"]
    assert result.claims[1].chunk_ids == ["b-0001"]  # P7 does not exist
    assert result.invalid_citations == 1
    assert not result.claims[2].cited  # kept, so the fact checker can mark it unverified
    assert 'location="p. 3"' in llm.prompts[0][1] and "French" in llm.prompts[0][1]


def test_no_passages_gives_a_gap_without_calling_the_model():
    llm = FakeLLM()
    result = Analyst(llm).run(SQ, [])
    assert result.claims == [] and result.gaps and llm.prompts == []


# --- Writer ------------------------------------------------------------------------------


def checked(claim_id, text, status, sources):
    claim = Claim(claim_id=claim_id, sub_question_id="Q1", text=text, chunk_ids=[], source_ids=sources)
    return CheckedClaim(claim=claim, status=status, supporting_chunk_ids=[], contradicting_chunk_ids=[],
                        supporting_source_ids=sources, justification="")


CHECKED = [checked("C1", "Prices fell 89%.", Status.VERIFIED, ["a"]),
           checked("C2", "Prices rose in 2022.", Status.PARTIALLY_VERIFIED, ["b", "a"]),
           checked("C3", "Batteries will be free by 2030.", Status.UNVERIFIED, [])]
ANALYSES = [SubQuestionAnalysis(sub_question_id="Q1", question=SQ.question, summary="", claims=[c.claim for c in CHECKED],
                                disagreements=[], gaps=["No data after 2022."], passages_used=2)]
SOURCES = {"a": make_source("a", "BNEF survey", published_date="2023-11-26"), "b": make_source("b", "IEA report")}


def test_labels_follow_first_citation_order():
    assert assign_labels(CHECKED) == {"a": "S1", "b": "S2"}


def test_report_keeps_valid_citations_and_builds_the_bibliography():
    llm = FakeLLM(ReportDraft(
        title="Battery prices", executive_summary="Prices fell 89% [S1] but rose in 2022 [S2] [S9].",
        sections=[ReportSection(heading="Trends", body="Costs dropped [S1]. A study [S7] says otherwise.")],
        limitations=["Few sources."]))
    report = Writer(llm).run("Why are batteries cheaper?", ANALYSES, CHECKED, SOURCES, language="en")
    assert report.executive_summary == "Prices fell 89% [S1] but rose in 2022 [S2]."
    assert report.sections[0].body == "Costs dropped [S1]. A study says otherwise."
    assert report.removed_citations == 2
    assert [(b.label, b.title, b.published_date) for b in report.bibliography] == [
        ("S1", "BNEF survey", "2023-11-26"), ("S2", "IEA report", None)]
    assert report.uncertainties == ["[PARTIALLY_VERIFIED] Prices rose in 2022.",
                                    "[UNVERIFIED] Batteries will be free by 2030."]
    assert report.status_counts == {"VERIFIED": 1, "PARTIALLY_VERIFIED": 1, "CONFLICTING": 0, "UNVERIFIED": 1}


def test_writer_sees_statuses_and_labels_but_no_web_text():
    llm = FakeLLM(ReportDraft(title="t", executive_summary="s", sections=[], limitations=[]))
    Writer(llm).run("Why are batteries cheaper?", ANALYSES, CHECKED, SOURCES)
    prompt = llm.prompts[0][1]
    assert "[VERIFIED] Prices fell 89%. [S1]" in prompt
    assert "[UNVERIFIED] Batteries will be free by 2030." in prompt
    assert "[GAP] No data after 2022." in prompt
    assert "<document" not in prompt


def test_uncited_sources_are_not_listed():
    llm = FakeLLM(ReportDraft(title="t", executive_summary="Only S2 here [S2].", sections=[], limitations=[]))
    report = Writer(llm).run("q", ANALYSES, CHECKED, SOURCES)
    assert [b.label for b in report.bibliography] == ["S2"]
