import os

import pytest

from app.config import Settings
from app.database.connection import connect
from app.jobs import (
    JobManager,
    ResearchBusyError,
    ResearchParams,
    ResearchRepository,
    ResearchStatus,
    pid_alive,
    read_state,
    register_existing,
    research_settings,
    run_job,
    step_states,
)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in Settings.model_fields:
        monkeypatch.delenv(name.upper(), raising=False)


@pytest.fixture
def settings(tmp_path):
    s = Settings(_env_file=None, storage_dir=tmp_path)
    s.ensure_storage()
    return s


class FakeProcesses:
    def __init__(self):
        self.alive, self.started, self.stopped, self.next_pid = set(), [], [], 1000

    def spawn(self, command, research_id, log_path):
        self.next_pid += 1
        self.started.append((command, research_id))
        self.alive.add(self.next_pid)
        return self.next_pid

    def is_alive(self, pid):
        return pid in self.alive

    def stop(self, pid):
        self.stopped.append(pid)
        self.alive.discard(pid)


def manager_with(settings):
    procs = FakeProcesses()
    return JobManager(settings, spawn=procs.spawn, alive=procs.is_alive, stop=procs.stop), procs


def test_depth_narrows_the_configuration(settings):
    quick = research_settings(settings, ResearchParams(depth="quick", max_sources=5))
    assert (quick.max_sub_questions, quick.max_extra_research_cycles, quick.max_research_minutes) == (2, 0, 15.0)
    assert quick.max_sources_per_research == 5 and quick.storage_dir == settings.storage_dir
    deep = research_settings(settings, ResearchParams(depth="deep"))
    assert deep.max_extra_research_cycles == 3 and deep.max_claims_per_sub_question == 5


def test_start_records_the_research_and_launches_a_process(settings):
    manager, procs = manager_with(settings)
    record = manager.start("Why are heat pumps efficient?", "en", ResearchParams(depth="quick"))
    assert procs.started == [("run", record.research_id)]
    stored = manager.repo.get(record.research_id)
    assert stored.status == ResearchStatus.RUNNING and stored.pid == 1001 and stored.params.depth == "quick"
    assert stored.current_step == "plan"


def test_only_one_research_runs_at_a_time(settings):
    manager, _ = manager_with(settings)
    manager.start("First question here", "en", ResearchParams())
    with pytest.raises(ResearchBusyError):
        manager.start("Second question here", "en", ResearchParams())


def test_a_dead_process_is_marked_interrupted_and_can_be_resumed(settings):
    manager, procs = manager_with(settings)
    record = manager.start("Why are heat pumps efficient?", "fr", ResearchParams())
    procs.alive.clear()  # the machine rebooted
    assert manager.refresh()[0].status == ResearchStatus.INTERRUPTED
    assert manager.running() is None
    manager.repo.update(record.research_id, current_step="analyze")
    manager.resume(record.research_id)
    assert procs.started[-1] == ("resume", record.research_id)
    stored = manager.repo.get(record.research_id)
    assert stored.status == ResearchStatus.RUNNING
    assert stored.current_step == "analyze"  # progress keeps showing where the run stopped


def test_cancel_stops_the_process_and_keeps_the_research_resumable(settings):
    manager, procs = manager_with(settings)
    record = manager.start("Why are heat pumps efficient?", "en", ResearchParams())
    manager.cancel(record.research_id)
    stored = manager.repo.get(record.research_id)
    assert procs.stopped == [1001] and stored.status == ResearchStatus.CANCELLED and stored.pid is None
    assert stored.status.resumable


def test_a_completed_research_cannot_be_resumed(settings):
    manager, _ = manager_with(settings)
    record = manager.repo.create("Done question", "en", ResearchParams(), status=ResearchStatus.COMPLETED)
    with pytest.raises(ValueError, match="nothing to resume"):
        manager.resume(record.research_id)


def test_a_failed_launch_is_recorded(settings):
    def broken(command, research_id, log_path):
        raise OSError("no python")
    manager = JobManager(settings, spawn=broken, alive=lambda pid: False, stop=lambda pid: None)
    with pytest.raises(OSError):
        manager.start("Why are heat pumps efficient?", "en", ResearchParams())
    record = manager.repo.list()[0]
    assert record.status == ResearchStatus.FAILED and "no python" in record.error


def test_step_states_follow_the_current_step(settings):
    repo = ResearchRepository(connect(settings.documents_db_path))
    record = repo.create("q", "en", ResearchParams())
    record.current_step = "analyze"
    states = [s for _, _, s in step_states(record)]
    assert states == ["done", "done", "done", "done", "current", "pending", "pending"]
    record.current_step, record.status = "next_cycle", ResearchStatus.CANCELLED
    assert [s for _, _, s in step_states(record)][:3] == ["done", "stopped", "pending"]
    record.status = ResearchStatus.COMPLETED
    assert {s for _, _, s in step_states(record)} == {"done"}


def test_pid_alive():
    assert pid_alive(os.getpid())
    assert not pid_alive(None) and not pid_alive(999_999_999)


# --- Job process -------------------------------------------------------------------------


class FakeRunner:
    def __init__(self, fail=False):
        self.fail, self.calls = fail, []

    def stream(self, question, language, research_id):
        self.calls.append(("stream", question, language))
        yield "plan", {"events": [{"at": "2026-11-07T10:00:00", "node": "plan", "message": "2 sub-questions"}]}
        if self.fail:
            raise RuntimeError("model crashed")
        yield "write", {"events": []}

    def resume_stream(self, research_id):
        self.calls.append(("resume",))
        yield "write", {"events": []}

    def pending(self, research_id):
        return ()

    def state(self, research_id):
        return {"stop_reason": "time limit reached"}


def test_run_job_updates_the_record_until_completion(settings):
    repo = ResearchRepository(connect(settings.documents_db_path))
    record = repo.create("Why are heat pumps efficient?", "fr", ResearchParams(depth="quick"))
    seen = {}

    def factory(job_settings):
        seen["minutes"] = job_settings.max_research_minutes
        return runner
    runner = FakeRunner()
    assert run_job(record.research_id, base=settings, runner_factory=factory) == 0
    stored = repo.get(record.research_id)
    assert stored.status == ResearchStatus.COMPLETED and stored.stop_reason == "time limit reached"
    assert stored.finished_at is not None and stored.pid is None
    assert seen["minutes"] == 15.0 and runner.calls[0] == ("stream", "Why are heat pumps efficient?", "fr")


def test_run_job_records_failures(settings):
    repo = ResearchRepository(connect(settings.documents_db_path))
    record = repo.create("Why are heat pumps efficient?", "en", ResearchParams())
    with pytest.raises(RuntimeError):
        run_job(record.research_id, base=settings, runner_factory=lambda s: FakeRunner(fail=True))
    stored = repo.get(record.research_id)
    assert stored.status == ResearchStatus.FAILED and stored.error == "RuntimeError: model crashed"


def test_resume_job_uses_the_checkpoint(settings):
    repo = ResearchRepository(connect(settings.documents_db_path))
    record = repo.create("q?", "en", ResearchParams(), status=ResearchStatus.INTERRUPTED)
    runner = FakeRunner()
    run_job(record.research_id, resume=True, base=settings, runner_factory=lambda s: runner)
    assert runner.calls == [("resume",)]


# --- Reading progress and history -----------------------------------------------------------


def test_progress_is_read_from_the_checkpoint_without_the_graph(tmp_path):
    from app.agents.fact_checker import Status
    from tests.test_workflow import RuleLLM, always, runner_with

    runner_with(RuleLLM(always(Status.VERIFIED)), tmp_path).start("Explain alpha and beta", research_id="r1")
    state = read_state(Settings(_env_file=None, storage_dir=tmp_path), "r1")
    assert state["events"][0]["node"] == "plan" and state["report"]["title"] == "Report"
    assert read_state(Settings(_env_file=None, storage_dir=tmp_path), "unknown") == {}


def test_reports_from_the_command_line_join_the_history(settings):
    from app.reports.export import save_report
    from tests.test_export import report

    save_report(report().model_copy(update={"research_id": "cli-1"}), settings.reports_dir / "cli-1")
    repo = ResearchRepository(connect(settings.documents_db_path))
    assert register_existing(settings, repo) == 1 and register_existing(settings, repo) == 0
    record = repo.get("cli-1")
    assert record.status == ResearchStatus.COMPLETED and record.language == "fr"
    assert record.stop_reason == "time limit reached"
    assert record.created_at == record.finished_at  # start unknown: no negative duration
