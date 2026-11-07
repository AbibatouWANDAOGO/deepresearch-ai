"""Analyst and Writer."""

from app.agents.analyst import AnalysisDraft, Analyst, Claim, Finding, SubQuestionAnalysis
from app.agents.base import strip_passage_refs
from app.agents.fact_checker import CheckedClaim, Status
from app.agents.planner import SubQuestion
from app.agents.writer import ReportDraft, SummarySentence, Writer, assign_labels, best_excerpt
from app.rag.chunking import Chunk
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


def test_passage_ids_in_statements_are_removed_and_used_as_citations():
    llm = FakeLLM(AnalysisDraft(summary="Prices fell (P1).", findings=[
        Finding(statement="Prices fell 89%. This is stated in passage P1, which says pack prices fell.",
                passage_ids=[]),
        Finding(statement="Selon le passage P2, les prix ont augmenté en 2022.", passage_ids=["P2"]),
    ], disagreements=["Passages P1 and P2 disagree on 2022."], gaps=[]))
    result = Analyst(llm).run(SQ, PASSAGES)
    assert [c.text for c in result.claims] == ["Prices fell 89%.", "Les prix ont augmenté en 2022."]
    assert result.claims[0].chunk_ids == ["a-0000"]  # P1 was only written in the text
    assert result.summary == "Prices fell." and result.disagreements == ["Disagree on 2022."]


def test_passage_references_seen_in_a_real_run_are_cleaned():
    text = ("Les pompes à chaleur sont limitées par temps très froid. Cela est mentionné dans le passage P1 "
            "qui indique que la performance chute.")
    assert strip_passage_refs(text) == ("Les pompes à chaleur sont limitées par temps très froid.", ["P1"])
    assert strip_passage_refs("Le COP vaut 3 [P2] (P3, P4).") == ("Le COP vaut 3.", ["P2", "P3", "P4"])
    assert strip_passage_refs("PAC model P10X is quiet.")[0] == "PAC model P10X is quiet."


# --- Writer ------------------------------------------------------------------------------


def checked(claim_id, text, status, sources, contradicting=(), sq="Q1"):
    claim = Claim(claim_id=claim_id, sub_question_id=sq, text=text, chunk_ids=[], source_ids=sources)
    return CheckedClaim(claim=claim, status=status, supporting_chunk_ids=[f"{s}-0000" for s in sources],
                        contradicting_chunk_ids=list(contradicting), supporting_source_ids=sources,
                        justification="because")


SQ2 = SubQuestion(id="Q2", question="What comes next?", search_queries=["future"])
CHECKED = [checked("C1", "Prices fell 89%.", Status.VERIFIED, ["a"]),
           checked("C2", "Prices rose in 2022.", Status.PARTIALLY_VERIFIED, ["b", "a"]),
           checked("C3", "Batteries will be free by 2030.", Status.UNVERIFIED, []),
           checked("C4", "Sodium cells will win.", Status.CONFLICTING, ["b"], contradicting=["c-0000"], sq="Q2")]
ANALYSES = [SubQuestionAnalysis(sub_question_id="Q1", question=SQ.question, summary="",
                                claims=[c.claim for c in CHECKED[:3]], disagreements=[],
                                gaps=["No data after 2022."], passages_used=2)]
SOURCES = {"a": make_source("a", "BNEF survey", published_date="2023-11-26"), "b": make_source("b", "IEA report"),
           "c": make_source("c", "Sodium blog")}
CHUNKS = {"a-0000": Chunk(chunk_id="a-0000", source_id="a", research_id="r1", chunk_index=0, n_tokens=20,
                          text="Intro sentence here. Pack prices fell 89% between 2010 and 2020.\fNext page text.",
                          start_char=1000, end_char=1081, page_start=4, page_end=5)}


def draft(*sentences, title="Battery prices"):
    return ReportDraft(title=title, summary=[SummarySentence(text=t, claim_ids=ids) for t, ids in sentences])


def write(llm, checked_claims=CHECKED, language="en", stop_reason=None):
    return Writer(llm).run("Why are batteries cheaper?", [SQ, SQ2], ANALYSES, checked_claims, SOURCES, CHUNKS,
                           language=language, stop_reason=stop_reason)


def test_labels_follow_first_citation_order_including_contradicting_sources():
    assert assign_labels(CHECKED) == {"a": "S1", "b": "S2", "c": "S3"}
    assert assign_labels(CHECKED, known={"b", "c"}) == {"b": "S1", "c": "S2"}


def test_statements_are_built_by_code_with_citations_and_status_wording():
    report = write(FakeLLM(draft(("Prices fell sharply.", ["C1"]))))
    q1, q2 = report.sections
    assert q1.body.splitlines() == ["- Prices fell 89% [S1].", "- Partially supported: Prices rose in 2022 [S1] [S2]."]
    assert q2.body == "- Disputed: Sodium cells will win [S2]; contradicted by [S3]."
    assert [b.label for b in report.bibliography] == ["S1", "S2", "S3"]
    assert report.status_counts == {"VERIFIED": 1, "PARTIALLY_VERIFIED": 1, "CONFLICTING": 1, "UNVERIFIED": 1}


def test_unverified_claims_never_reach_the_body():
    report = write(FakeLLM(draft(("Prices fell.", ["C1"]))))
    body = " ".join(s.body for s in report.sections) + report.executive_summary
    assert "free by 2030" not in body
    assert [u.text for u in report.uncertainties] == ["Batteries will be free by 2030."]


def test_summary_sentences_must_name_reportable_claims():
    report = write(FakeLLM(draft(("Prices fell sharply [S9] (P3) (C1).", ["C1"]),
                                 ("Batteries will be free.", ["C3"]),  # unverified: dropped
                                 ("A vague remark.", []),               # no claim: dropped
                                 ("Prices fell, then rose in 2022 (C2, C1).", ["C2", "C1", "C99"]))))
    assert report.executive_summary == "Prices fell sharply [S1]. Prices fell, then rose in 2022 [S1] [S2]."
    assert report.removed_sentences == 2 and report.summary_origin == "model"


def test_summary_falls_back_to_verified_claims():
    report = write(FakeLLM(draft(("Nothing usable.", []))))
    assert report.summary_origin == "fallback"
    assert report.executive_summary == "Prices fell 89% [S1]."  # Q2 has only a disputed claim


def test_failed_writer_call_still_gives_a_cited_report():
    from app.llm import LLMError

    class SlowLLM(FakeLLM):
        def structured(self, system, user, schema):
            raise LLMError("timed out")

    report = write(SlowLLM())
    assert report.title == "Why are batteries cheaper?" and report.summary_origin == "fallback"
    assert report.sections[0].statements[0].labels == ["S1"]


def test_no_reportable_claim_means_no_model_call():
    llm = FakeLLM()
    report = write(llm, [CHECKED[2]], language="fr")
    assert llm.prompts == [] and report.bibliography == []
    assert report.executive_summary.startswith("Les sources collectées ne permettent pas")
    assert report.sections[0].body.startswith("Aucune affirmation")


def test_writer_sees_claim_ids_and_statuses_but_no_web_text():
    llm = FakeLLM(draft(("s", ["C1"])))
    write(llm)
    prompt = llm.prompts[0][1]
    assert "- C1 [VERIFIED] Prices fell 89%." in prompt
    assert "Unverified claims (do not state them):\n- Batteries will be free by 2030." in prompt
    assert "No data after 2022." in prompt and "<document" not in prompt and "[S1]" not in prompt


def test_claim_ids_written_in_summary_text_are_removed():
    report = write(FakeLLM(draft(("C1, C2 : Prices fell then rose.", ["C1", "C2"]), ("Prices fell (claims C1).", ["C1"]),
                                 title="Battery prices (C1)")))
    assert report.executive_summary == "Prices fell then rose [S1] [S2]. Prices fell [S1]."
    assert report.title == "Battery prices"


def test_complete_research_with_every_part_answered_has_no_research_limitation():
    full = [CHECKED[0], checked("C5", "Sodium cells exist.", Status.VERIFIED, ["a", "b"], sq="Q2")]
    report = write(FakeLLM(draft(("s", ["C1"]))), full, stop_reason="every sub-question is supported by evidence")
    assert report.limitations == ["1 claim(s) of the report rest on a single source.", "No data after 2022."]


def test_limitations_are_stated_by_code_and_french_wording_is_used():
    report = write(FakeLLM(draft(("s", ["C2"]))), language="fr", stop_reason="time limit reached")
    assert report.limitations == [
        "La recherche s'est arrêtée avant que chaque sous-question soit étayée (durée maximale atteinte).",
        ("1 affirmation(s) n'ont pas pu être vérifiées : elles sont listées à part et ne sont pas présentées "
        "comme des faits."),
        "2 affirmation(s) du rapport ne reposent que sur une seule source.",
        "No data after 2022.",
    ]
    assert report.stop_reason == "time limit reached"
    assert report.sections[0].body.splitlines()[1] == "- Partiellement confirmé : Prices rose in 2022 [S1] [S2]."
    assert report.sections[1].body.endswith(" ; contredit par [S3].")


def test_citations_point_to_the_closest_sentence_with_offsets_and_page():
    report = write(FakeLLM(draft(("s", ["C1"]))))
    citation = report.sections[0].statements[0].citations[0]
    assert citation.quote == "Pack prices fell 89% between 2010 and 2020."
    assert (citation.start_char, citation.page, citation.location) == (1021, 4, "p. 4")
    text = CHUNKS["a-0000"].text
    assert text[citation.start_char - 1000:citation.end_char - 1000] == citation.quote
    disputed = report.sections[1].statements[0].citations
    assert [(c.label, c.role) for c in disputed] == [("S2", "supports"), ("S3", "contradicts")]
    assert disputed[0].quote == ""  # chunk not available: the citation is kept without a quote


def test_excerpt_page_follows_page_breaks_inside_the_chunk():
    quote, start, _, page = best_excerpt(CHUNKS["a-0000"], "next page")
    assert quote.startswith("Next page text") and page == 5
    assert start == 1000 + CHUNKS["a-0000"].text.index("Next")


def test_passage_ids_in_justifications_become_report_labels():
    item = checked("C1", "Prices fell 89%.", Status.VERIFIED, ["a"])
    item.justification = "Passage P1 states it; P2 is about something else. Documents P1, P2 et P9 agree."
    item.passage_map = {"P1": "a-0000", "P2": "zz-0004"}
    report = write(FakeLLM(draft(("s", ["C1"]))), [item])
    assert report.sections[0].statements[0].justification == (
        "Passage [S1] states it; is about something else. Documents [S1] agree.")


def test_summary_sentences_cannot_go_beyond_the_claims_they_name():
    # seen in a real run: an unverified figure attached to an unrelated verified claim
    report = write(FakeLLM(draft(("Prices fell by 89%.", ["C1"]),
                                 ("Prices fell 95%.", ["C1"]),                        # number not in C1
                                 ("Heat pumps cover most thermal losses of homes.", ["C1"]))))  # unrelated words
    assert report.executive_summary == "Prices fell by 89% [S1]."
    assert report.removed_sentences == 2
