"""Dashboard pages, rendered with Streamlit's test runner (no browser, no research process)."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from app.config import Settings
from app.reports.export import save_report
from tests.test_export import HOSTILE, report

DASHBOARD = str(Path(__file__).resolve().parents[1] / "frontend" / "dashboard.py")


@pytest.fixture
def app(tmp_path, monkeypatch):
    for name in Settings.model_fields:
        monkeypatch.delenv(name.upper(), raising=False)
    monkeypatch.setenv("STORAGE_DIR", str(tmp_path))
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://127.0.0.1:9")  # nothing listens: fails at once
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("ALLOW_PAID_APIS", "false")
    return AppTest.from_file(DASHBOARD, default_timeout=60)


def saved(tmp_path, research_id="cli-1", **changes):
    r = report().model_copy(update={"research_id": research_id, **changes})
    save_report(r, tmp_path / "reports" / research_id)
    return r


def text_of(at) -> str:
    return " ".join(str(e.value) for e in [*at.markdown, *at.caption, *at.info, *at.warning, *at.error,
                                           *at.success, *at.header, *at.subheader])


def test_new_research_page_has_the_form_and_warns_when_ollama_is_down(app):
    app.run()
    assert not app.exception
    assert app.header[0].value == "New research"
    assert app.text_area[0].label == "Question" and app.slider[0].value == 12
    depth = next(r for r in app.radio if r.label == "Depth")
    assert list(depth.options) == ["Quick", "Standard", "Deep"] and depth.value == "quick"
    assert "Ollama is not reachable" in text_of(app)
    assert "Paid APIs disabled" in text_of(app)


def test_a_too_short_question_is_refused(app):
    app.run()
    app.text_area[0].input("why?")
    app.button[0].click().run()
    assert "Write a full question" in text_of(app)


def test_history_lists_researches_from_the_command_line(app, tmp_path):
    saved(tmp_path)
    app.run()
    app.radio(key="page").set_value("History").run()
    assert not app.exception
    rows = app.dataframe[0].value
    assert list(rows["Question"]) == ["Quels sont les avantages des pompes à chaleur ?"]
    assert list(rows["Status"]) == ["completed"]


def test_research_page_shows_the_cited_report_and_its_traceability(app, tmp_path):
    saved(tmp_path)
    app.run()
    app.session_state["selected"] = "cli-1"
    app.session_state["goto"] = "Research"
    app.run()
    assert not app.exception
    page = text_of(app)
    assert ":green-badge[Completed]" in page
    assert {m.label: m.value for m in app.metric}["Verified"] == "1"
    assert "Le COP moyen est de 3,2 [S1]." in page
    assert "Un COP de 3,2 a été mesuré." in page  # the quote, in the Claims tab
    assert "Points non vérifiés" in page and "non vérifiée : durée maximale atteinte" in page


def test_web_text_is_escaped_in_the_dashboard(app, tmp_path):
    r = report(claim_text=HOSTILE)
    save_report(r.model_copy(update={"research_id": "evil"}), tmp_path / "reports" / "evil")
    app.run()
    app.session_state["selected"] = "evil"
    app.session_state["goto"] = "Research"
    app.run()
    assert not app.exception
    rendered = " ".join(str(m.value) for m in app.markdown)
    assert "![x](https://evil" not in rendered and "<script>" not in rendered
