"""Markdown and PDF export of reports."""

import pymupdf

from app.agents.fact_checker import Status
from app.agents.writer import (
    BibliographyEntry,
    Citation,
    CitedStatement,
    Report,
    ReportSection,
)
from app.reports.export import load_report, md_escape, save_report, to_html, to_markdown

HOSTILE = "Prices fell ![x](https://evil.example/pixel.png) <script>alert(1)</script> [click](javascript:x)"


def statement(claim_id, text, status, labels, citations=(), justification="", sq="Q1"):
    return CitedStatement(claim_id=claim_id, sub_question_id=sq, text=text, status=status, labels=labels,
                          citations=list(citations), justification=justification)


def report(language="fr", claim_text="Le COP moyen est de 3,2.", title="Pompes à chaleur"):
    verified = statement("C1", claim_text, Status.VERIFIED, ["S1"], [
        Citation(label="S1", source_id="a", chunk_id="a-0003", role="supports", location="p. 4",
                 quote="Un COP de 3,2 a été mesuré.", start_char=1200, end_char=1227, page=4)],
        justification="Le passage donne la valeur.")
    unverified = statement("C2", "Les PAC durent 40 ans.", Status.UNVERIFIED, [],
                           justification="not checked: time limit")
    return Report(
        question="Quels sont les avantages des pompes à chaleur ?", language=language, title=title,
        executive_summary="Les PAC sont efficaces [S1].",
        sections=[ReportSection(sub_question_id="Q1", heading="Efficacité", statements=[verified],
                                body="- Le COP moyen est de 3,2 [S1]."),
                  ReportSection(sub_question_id="Q2", heading="Coûts", statements=[], body="Aucune affirmation.")],
        uncertainties=[unverified], limitations=["Peu de sources récentes."],
        bibliography=[BibliographyEntry(label="S1", source_id="a", title="Étude ADEME", url="https://ademe.fr/pac",
                                        site_name="ADEME", published_date="2024-03-01", collected_at="2026-11-07")],
        status_counts={"VERIFIED": 1, "PARTIALLY_VERIFIED": 0, "CONFLICTING": 0, "UNVERIFIED": 1},
        research_id="r-1", stop_reason="time limit reached",
    )


def test_markdown_has_every_part_of_the_report():
    md = to_markdown(report())
    for expected in ["# Pompes à chaleur", "## Résumé", "Les PAC sont efficaces [S1].", "## Efficacité",
                     "- Le COP moyen est de 3,2 [S1].", "## Coûts", "Aucune affirmation.",
                     "## Points non vérifiés", "- Les PAC durent 40 ans. — *non vérifiée : durée maximale atteinte*",
                     "## Limites", "## Traçabilité des affirmations", "**C1 · Vérifié**",
                     "- [S1], p. 4 (soutient) : « Un COP de 3,2 a été mesuré. »",
                     "## Sources", ("- [S1] Étude ADEME. ADEME. publié le 2024-03-01. consulté le 2026-11-07. "
                                   "<https://ademe.fr/pac>"),
                     "arrêt : durée maximale atteinte", "vérifié : 1, non vérifié : 1"]:
        assert expected in md, expected


def test_english_labels():
    md = to_markdown(report(language="en"))
    assert "## Summary" in md and "## Unverified points" in md and '(supports): "Un COP' in md


def test_web_text_cannot_inject_markdown_or_html():
    md = to_markdown(report(claim_text=HOSTILE, title="Title](https://evil.example)"))
    import re

    assert not re.search(r"(?<!\\)\]\(", md)  # every "](" that could open a link or image is escaped
    assert "<script>" not in md and "\\<script\\>" in md
    assert md_escape("a [b](c)") == r"a \[b\](c)"
    assert md_escape("# Title") == r"\# Title" and md_escape("<b>") == r"\<b\>"
    page = to_html(report(claim_text=HOSTILE))
    assert "<script>" not in page and "&lt;script&gt;" in page


def test_unsafe_urls_are_not_linked():
    r = report()
    r.bibliography[0].url = "javascript:alert(1)"
    assert "javascript:" not in to_markdown(r)


def test_saved_report_round_trips_and_the_pdf_is_readable(tmp_path):
    files, error = save_report(report(), tmp_path / "r-1")
    assert error is None and set(files) == {"json", "md", "pdf"}
    assert load_report(files["json"]) == report().model_copy(update={"generated_at": load_report(files["json"]).generated_at})
    with pymupdf.open(files["pdf"]) as doc:
        text = " ".join(page.get_text() for page in doc)
    assert "Pompes à chaleur" in " ".join(text.split()) and "« Un COP de 3,2 a été mesuré. »" in " ".join(text.split())
    assert "1 / " in text  # page numbers


def test_a_pdf_failure_keeps_json_and_markdown(tmp_path, monkeypatch):
    from app.reports import export

    def broken(report, path):
        raise RuntimeError("layout")
    monkeypatch.setattr(export, "to_pdf", broken)
    files, error = save_report(report(), tmp_path)
    assert set(files) == {"json", "md"} and error == "PDF export failed: layout"


def test_titles_ending_with_a_question_mark_get_no_extra_period():
    r = report()
    r.bibliography[0].title = "Comment fonctionne une PAC ?"
    assert "- [S1] Comment fonctionne une PAC ? ADEME. publié le" in to_markdown(r)


def test_english_reports_keep_reasons_unchanged():
    assert "stopped: time limit reached" in to_markdown(report(language="en"))
