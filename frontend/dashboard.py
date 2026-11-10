"""DeepResearch AI dashboard.

    streamlit run frontend/dashboard.py

Starts researches in the background (see app/jobs.py), follows their progress step by step,
and shows the history, the collected sources, the cited report with its verification
statuses, the trace from each claim to its passages, and the Markdown / PDF downloads.

Everything that comes from the web (titles, excerpts, quotes, claims) is escaped before
being rendered as Markdown, and links are only made from http(s) URLs.
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime
from pathlib import Path

import httpx
import streamlit as st
from pydantic import ValidationError

# Streamlit puts this file's folder on sys.path, not the project root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import __version__
from app.agents.fact_checker import Status
from app.agents.writer import CitedStatement, Report, render_statement
from app.config import Settings
from app.database.connection import connect
from app.database.models import SourceRepository
from app.jobs import (
    DEPTHS,
    JobManager,
    ResearchBusyError,
    ResearchParams,
    ResearchRecord,
    ResearchStatus,
    read_state,
    register_existing,
    step_states,
)
from app.reports.export import load_report, md_cited, md_escape, safe_url
from app.reports.i18n import localize

st.set_page_config(page_title="DeepResearch AI", page_icon="🔎", layout="wide")

STATUS_BADGE = {
    Status.VERIFIED: ":green-badge[Verified]",
    Status.PARTIALLY_VERIFIED: ":orange-badge[Partially verified]",
    Status.CONFLICTING: ":red-badge[Disputed]",
    Status.UNVERIFIED: ":gray-badge[Unverified]",
}
RUN_BADGE = {
    ResearchStatus.RUNNING: ":blue-badge[Running]",
    ResearchStatus.COMPLETED: ":green-badge[Completed]",
    ResearchStatus.FAILED: ":red-badge[Failed]",
    ResearchStatus.CANCELLED: ":gray-badge[Cancelled]",
    ResearchStatus.INTERRUPTED: ":orange-badge[Interrupted]",
}
STEP_ICON = {"done": "✅", "current": "⏳", "stopped": "⏸️", "pending": "▫️"}
DEPTH_HELP = {
    "quick": "2 sub-questions, no extra cycle, about 15 minutes on a laptop CPU",
    "standard": "3 sub-questions, 1 extra cycle if evidence is missing, up to 30 minutes",
    "deep": "5 sub-questions, up to 3 extra cycles, up to 60 minutes",
}
LANGUAGES = {"fr": "Français", "en": "English"}


# --- Resources ---------------------------------------------------------------------------


def load_settings() -> Settings | None:
    try:
        return Settings()
    except ValidationError as exc:
        st.error("The configuration is invalid. Fix your .env file and reload the page.")
        for error in exc.errors():
            location = ".".join(str(part) for part in error["loc"]) or "settings"
            st.write(f"- `{location}`: {error['msg']}")
        return None


@st.cache_resource
def get_manager(storage_dir: str) -> JobManager:
    settings = Settings(storage_dir=Path(storage_dir))
    settings.ensure_storage()
    manager = JobManager(settings)
    register_existing(settings, manager.repo)
    return manager


@st.cache_data(ttl=30, show_spinner=False)
def model_status(provider: str, base_url: str, model: str) -> tuple[bool, str]:
    """Is the local model available? Checked only for Ollama, cached for 30 s."""
    if provider != "ollama":
        return True, f"{provider} ({model})"
    try:
        tags = httpx.get(base_url.rstrip("/") + "/api/tags", timeout=2).json()
    except (httpx.HTTPError, ValueError):
        return False, f"Ollama is not reachable at {base_url}: start it before a research."
    names = {m.get("name") for m in tags.get("models", [])}
    if model not in names and f"{model}:latest" not in names:
        return False, f"Model {model} is not installed: run `ollama pull {model}`."
    return True, f"Ollama · {model}"


def select(research_id: str | None) -> None:
    """Open a research. The page switch is applied at the start of the next run, before the
    navigation widget exists (Streamlit forbids changing a widget's value after it is drawn)."""
    st.session_state["selected"] = research_id
    if research_id:
        st.session_state["goto"] = "Research"


def elapsed(record: ResearchRecord) -> str:
    end = record.finished_at or (datetime.now(UTC) if record.status == ResearchStatus.RUNNING else record.updated_at)
    seconds = int((end - record.created_at).total_seconds())
    if seconds <= 0:
        return "—"
    return f"{seconds // 60} min {seconds % 60:02d} s" if seconds >= 60 else f"{seconds} s"


# --- Sidebar -----------------------------------------------------------------------------


def sidebar(settings: Settings, manager: JobManager) -> None:
    with st.sidebar:
        st.title("DeepResearch AI")
        st.caption(f"Version {__version__}")
        st.radio("Page", ["New research", "History", "Research"], key="page", label_visibility="collapsed")
        st.divider()
        ok, message = model_status(settings.llm_provider, settings.ollama_base_url, settings.ollama_model)
        (st.success if ok else st.warning)(message, icon="🧠")
        st.caption("Search: " + settings.search_provider + (f", fallback {settings.search_fallback}"
                                                             if settings.search_fallback else ""))
        if not settings.allow_paid_apis:
            st.caption("Paid APIs disabled: free, local services only.")
        running = manager.running()
        if running:
            st.info(f"Running: {running.question[:80]}", icon="⏳")
            st.button("Follow it", on_click=select, args=(running.research_id,), width="stretch")


# --- New research ------------------------------------------------------------------------


def new_research_page(settings: Settings, manager: JobManager) -> None:
    st.header("New research")
    st.write("Ask a question: the agent plans the research, searches the web, reads pages and PDFs, checks every "
             "important claim against its sources and writes a report where each statement is cited.")
    running = manager.running()
    if running:
        st.info("A research is already running: only one runs at a time on a local model.")
        st.button("Follow the running research", on_click=select, args=(running.research_id,))
    with st.form("new_research"):
        question = st.text_area("Question", max_chars=500, height=90,
                                placeholder="Quels sont les avantages et les limites des pompes à chaleur ?")
        left, middle, right = st.columns(3)
        language = left.selectbox("Report language", list(LANGUAGES), format_func=LANGUAGES.get)
        depth = middle.radio("Depth", list(DEPTHS), index=0, format_func=str.capitalize,
                             captions=[DEPTH_HELP[d] for d in DEPTHS])
        max_sources = right.slider("Maximum sources", 3, 30, 12)
        submitted = st.form_submit_button("Start the research", type="primary", disabled=running is not None)
    if submitted:
        if len(question.strip()) < 10:
            st.error("Write a full question (at least 10 characters).")
            return
        try:
            record = manager.start(question, language, ResearchParams(depth=depth, max_sources=max_sources))
        except ResearchBusyError as exc:
            st.error(str(exc))
        except OSError as exc:
            st.error(f"The research could not be started: {exc}")
        else:
            select(record.research_id)
            st.rerun()


# --- History -----------------------------------------------------------------------------


def history_page(manager: JobManager) -> None:
    st.header("History")
    records = manager.refresh()
    if not records:
        st.info("No research yet.")
        return
    rows = [{"Question": r.question, "Status": r.status.value, "Language": r.language, "Depth": r.params.depth,
             "Started": r.created_at.astimezone().strftime("%Y-%m-%d %H:%M"), "Duration": elapsed(r),
             "Stopped because": r.stop_reason or ""} for r in records]
    event = st.dataframe(rows, hide_index=True, width="stretch", on_select="rerun",
                         selection_mode="single-row", key="history_table")
    picked = event.selection.rows if event and event.selection else []
    if picked:
        select(records[picked[0]].research_id)
        st.rerun()
    st.caption("Select a row to open the research.")


# --- One research ------------------------------------------------------------------------


def research_page(settings: Settings, manager: JobManager) -> None:
    research_id = st.session_state.get("selected")
    records = manager.refresh()
    if not research_id and records:
        research_id = records[0].research_id
    record = manager.repo.get(research_id) if research_id else None
    if record is None:
        st.info("Start a research or pick one in the history.")
        return

    st.header(md_escape(record.question))
    st.markdown(f"{RUN_BADGE[record.status]} &nbsp; `{record.research_id}` · {LANGUAGES.get(record.language)} · "
                f"depth {record.params.depth} · max {record.params.max_sources} sources · {elapsed(record)}")
    actions(manager, record)
    if record.status == ResearchStatus.RUNNING:
        live_progress(settings, manager, record.research_id)
    else:
        progress(settings, record)
        details(settings, manager, record)


def actions(manager: JobManager, record: ResearchRecord) -> None:
    if record.status == ResearchStatus.RUNNING:
        if st.button("Cancel", help="Stops the research. Completed steps are kept and it can be resumed."):
            manager.cancel(record.research_id)
            st.rerun()
    elif record.status.resumable:
        if record.error:
            st.error(f"The research stopped with an error: {record.error}")
        if st.button("Resume", type="primary", help="Continues from the last completed step."):
            try:
                manager.resume(record.research_id)
            except (ResearchBusyError, ValueError) as exc:
                st.error(str(exc))
            else:
                st.rerun()


@st.fragment(run_every=3)
def live_progress(settings: Settings, manager: JobManager, research_id: str) -> None:
    manager.refresh()
    record = manager.repo.get(research_id)
    if record.status != ResearchStatus.RUNNING:
        st.rerun()  # finished, cancelled or crashed: redraw the whole page with the results
    progress(settings, record)
    st.caption("This page refreshes every 3 seconds. You can close it: the research continues in the background.")


def progress(settings: Settings, record: ResearchRecord) -> None:
    state = read_state(settings, record.research_id)
    columns = st.columns(len(step_states(record)))
    for column, (_, label, step) in zip(columns, step_states(record), strict=True):
        column.markdown(f"{STEP_ICON[step]} **{label}**" if step == "current" else f"{STEP_ICON[step]} {label}")
    cycle = state.get("cycle", 0)
    if cycle:
        st.caption(f"Extra research cycles: {cycle}")
    events = state.get("events", [])
    if events and record.status == ResearchStatus.RUNNING:
        with st.container(border=True):
            for e in events[-8:]:
                st.markdown(f"`{e['at'][11:19]}` **{e['node']}** {md_escape(e['message'])}")


def details(settings: Settings, manager: JobManager, record: ResearchRecord) -> None:
    report_path = settings.reports_dir / record.research_id / "report.json"
    report = load_report(report_path) if report_path.exists() else None
    tabs = st.tabs(["Report", "Claims", "Sources", "Log"])
    with tabs[0]:
        report_tab(settings, report)
    with tabs[1]:
        claims_tab(report)
    with tabs[2]:
        sources_tab(settings, record.research_id, report)
    with tabs[3]:
        log_tab(settings, manager, record)


def report_tab(settings: Settings, report: Report | None) -> None:
    if report is None:
        st.info("No report yet: it is written at the end of the research.")
        return
    folder = settings.reports_dir / report.research_id
    downloads = st.columns([1, 1, 4])
    if (folder / "report.md").exists():
        downloads[0].download_button("Markdown", (folder / "report.md").read_bytes(), "report.md", "text/markdown",
                                     icon="⬇️")
    if (folder / "report.pdf").exists():
        downloads[1].download_button("PDF", (folder / "report.pdf").read_bytes(), "report.pdf", "application/pdf",
                                     icon="⬇️")

    counts = report.status_counts
    metrics = st.columns(5)
    metrics[0].metric("Verified", counts.get("VERIFIED", 0))
    metrics[1].metric("Partially verified", counts.get("PARTIALLY_VERIFIED", 0))
    metrics[2].metric("Disputed", counts.get("CONFLICTING", 0))
    metrics[3].metric("Unverified", counts.get("UNVERIFIED", 0))
    metrics[4].metric("Sources cited", len(report.bibliography))

    st.subheader(md_escape(report.title))
    st.markdown(md_cited(report.executive_summary))
    if report.summary_origin == "fallback":
        st.caption("Summary built from the verified claims (the model's summary was not usable).")
    for section in report.sections:
        st.markdown(f"#### {md_escape(section.heading)}")
        if not section.statements:
            st.markdown(f"*{md_escape(section.body)}*")
        for s in section.statements:
            st.markdown(f"{STATUS_BADGE[s.status]} {md_cited(render_statement(s, report.language))}")
    if report.uncertainties:
        st.markdown("#### " + ("Points non vérifiés" if report.language == "fr" else "Unverified points"))
        for s in report.uncertainties:
            reason = localize(s.justification, report.language) if s.justification else ""
            st.markdown(f"{STATUS_BADGE[s.status]} {md_escape(s.text)}" + (f" — *{md_escape(reason)}*" if reason else ""))
    if report.limitations:
        st.markdown("#### " + ("Limites" if report.language == "fr" else "Limitations"))
        st.markdown("\n".join(f"- {md_escape(x)}" for x in report.limitations))
    if report.bibliography:
        st.markdown("#### Sources")
        for b in report.bibliography:
            url = safe_url(b.url)
            title = f"[{md_escape(b.title)}]({url})" if url else md_escape(b.title)
            extra = " · ".join(md_escape(x) for x in (b.site_name, b.published_date) if x)
            st.markdown(f"**[{b.label}]** {title}" + (f" — {extra}" if extra else ""))


def claim_view(s: CitedStatement, language: str) -> None:
    with st.expander(f"{s.claim_id} · {s.text[:110]}{'…' if len(s.text) > 110 else ''}"):
        st.markdown(f"{STATUS_BADGE[s.status]} {md_escape(s.text)}")
        if s.justification:
            st.markdown(f"*{md_cited(localize(s.justification, language))}*")
        for c in s.citations:
            role = "supports" if c.role == "supports" else ":red[contradicts]"
            st.markdown(f"**[{c.label}]** {role} · {md_escape(c.location)} · `{c.chunk_id}`")
            if c.quote:
                st.markdown(f"> {md_escape(c.quote)}")


def claims_tab(report: Report | None) -> None:
    if report is None:
        st.info("Claims appear here once the report is written.")
        return
    st.caption("Each claim with its verification status, the justification and the exact passages that support "
               "or contradict it (page or character offsets in the extracted source text).")
    choice = st.segmented_control("Show", ["All", *[s.value for s in Status]], default="All", key="claim_filter")
    statements = [s for section in report.sections for s in section.statements] + report.uncertainties
    for s in statements:
        if choice in (None, "All") or s.status.value == choice:
            claim_view(s, report.language)


def sources_tab(settings: Settings, research_id: str, report: Report | None) -> None:
    sources = SourceRepository(connect(settings.documents_db_path)).list(research_id)
    if not sources:
        st.info("No source collected yet.")
        return
    labels = {b.source_id: b.label for b in report.bibliography} if report else {}
    usable = sum(s.usable for s in sources)
    st.caption(f"{len(sources)} URLs tried, {usable} usable, {len(labels)} cited in the report.")
    rows = [{"Cited": labels.get(s.source_id, ""), "Title": s.title or "", "URL": safe_url(s.url),
             "Type": s.source_type or "", "Status": s.access_status.value, "Published": s.published_date or "",
             "Excerpt": (s.excerpt or s.error or "")[:300]} for s in sources]
    rows.sort(key=lambda r: (r["Cited"] == "", int(r["Cited"][1:]) if r["Cited"] else 0))
    st.dataframe(rows, hide_index=True, width="stretch",
                 column_config={"URL": st.column_config.LinkColumn("URL", width="medium"),
                                "Excerpt": st.column_config.TextColumn("Excerpt", width="large")})


def log_tab(settings: Settings, manager: JobManager, record: ResearchRecord) -> None:
    events = read_state(settings, record.research_id).get("events", [])
    if record.stop_reason:
        st.markdown(f"**Stopped because:** {md_escape(record.stop_reason)}")
    if events:
        st.dataframe([{"Time": e["at"][11:19], "Step": e["node"], "Message": e["message"]} for e in events],
                     hide_index=True, width="stretch")
    log = manager.log_path(record.research_id)
    if log.exists():
        with st.expander("Process output"):
            st.code("\n".join(log.read_text(encoding="utf-8", errors="replace").splitlines()[-60:]) or "(empty)")


# --- Page --------------------------------------------------------------------------------


def main() -> None:
    settings = load_settings()
    if settings is None:
        return
    manager = get_manager(str(settings.storage_dir))
    st.session_state.setdefault("page", "New research")
    if "goto" in st.session_state:
        st.session_state["page"] = st.session_state.pop("goto")
    sidebar(settings, manager)
    page = st.session_state["page"]
    if page == "New research":
        new_research_page(settings, manager)
    elif page == "History":
        history_page(manager)
    else:
        research_page(settings, manager)


main()
