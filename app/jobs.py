"""Research jobs: history, background runs, cancellation and resume.

A research takes minutes to an hour on a local model, so the dashboard never runs it in its
own process: it starts ``python -m app.jobs run <id>`` in the background and reads the
progress from the database and the LangGraph checkpoints. Closing the page does not stop
the research. Cancelling stops the process; the last completed step is kept in the
checkpoint, so a cancelled or interrupted research can be resumed.

Only one research runs at a time: the local vector index (Qdrant embedded mode) accepts a
single process, and the local model uses the whole CPU anyway.

    python -m app.jobs run <research_id>
    python -m app.jobs resume <research_id>
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sqlite3
import subprocess
import sys
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from app.config import Settings
from app.database.connection import connect

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ResearchStatus(StrEnum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    INTERRUPTED = "interrupted"  # the process ended without finishing (crash, reboot)

    @property
    def resumable(self) -> bool:
        return self in (ResearchStatus.FAILED, ResearchStatus.CANCELLED, ResearchStatus.INTERRUPTED)


Depth = Literal["quick", "standard", "deep"]

# Depth sets how much work a research may do. "quick" fits a laptop CPU in about 15 minutes.
DEPTHS: dict[str, dict] = {
    "quick": {"max_sub_questions": 2, "max_extra_research_cycles": 0, "max_claims_per_sub_question": 2,
              "max_research_minutes": 15.0},
    "standard": {"max_sub_questions": 3, "max_extra_research_cycles": 1, "max_claims_per_sub_question": 3,
                 "max_research_minutes": 30.0},
    "deep": {"max_sub_questions": 5, "max_extra_research_cycles": 3, "max_claims_per_sub_question": 5,
             "max_research_minutes": 60.0},
}


class ResearchParams(BaseModel):
    depth: Depth = "standard"
    max_sources: int = Field(default=12, ge=1, le=100)


def research_settings(base: Settings, params: ResearchParams) -> Settings:
    """The settings of one research: the configuration, narrowed by the chosen depth."""
    overrides = {**DEPTHS[params.depth], "max_sources_per_research": params.max_sources}
    return Settings.model_validate({**base.model_dump(), **overrides})


class ResearchRecord(BaseModel):
    research_id: str
    question: str
    language: str
    status: ResearchStatus
    params: ResearchParams
    created_at: datetime
    updated_at: datetime
    finished_at: datetime | None = None
    current_step: str | None = None
    stop_reason: str | None = None
    error: str | None = None
    pid: int | None = None


def now() -> datetime:
    return datetime.now(UTC)


class ResearchRepository:
    def __init__(self, connection: sqlite3.Connection):
        self._db = connection

    def create(self, question: str, language: str, params: ResearchParams, research_id: str | None = None,
               status: ResearchStatus = ResearchStatus.RUNNING) -> ResearchRecord:
        stamp = now()
        record = ResearchRecord(research_id=research_id or f"r-{uuid.uuid4().hex[:10]}", question=question.strip(),
                                language=language, status=status, params=params, created_at=stamp, updated_at=stamp)
        self._db.execute(
            "INSERT INTO researches (research_id, question, language, status, params, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (record.research_id, record.question, language, status.value, params.model_dump_json(),
             stamp.isoformat(), stamp.isoformat()))
        self._db.commit()
        return record

    def get(self, research_id: str) -> ResearchRecord | None:
        row = self._db.execute("SELECT * FROM researches WHERE research_id = ?", (research_id,)).fetchone()
        return self._to_record(row) if row else None

    def list(self, limit: int = 100) -> list[ResearchRecord]:
        rows = self._db.execute("SELECT * FROM researches ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
        return [self._to_record(r) for r in rows]

    def update(self, research_id: str, **fields) -> None:
        fields["updated_at"] = now()
        values = {k: (v.isoformat() if isinstance(v, datetime) else v.value if isinstance(v, StrEnum) else v)
                  for k, v in fields.items()}
        assignments = ", ".join(f"{k} = :{k}" for k in values)
        self._db.execute(f"UPDATE researches SET {assignments} WHERE research_id = :rid", {**values, "rid": research_id})
        self._db.commit()

    @staticmethod
    def _to_record(row: sqlite3.Row) -> ResearchRecord:
        data = dict(row)
        data["params"] = json.loads(data["params"])
        return ResearchRecord.model_validate(data)


# --- Processes ---------------------------------------------------------------------------


def pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    if sys.platform == "win32":
        import ctypes

        # os.kill(pid, 0) would terminate the process on Windows: ask for its exit code instead.
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        code = ctypes.c_ulong()
        ok = kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
        kernel32.CloseHandle(handle)
        return bool(ok) and code.value == 259  # STILL_ACTIVE
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def spawn_job(command: str, research_id: str, log_path: Path) -> int:
    """Start ``python -m app.jobs <command> <id>`` detached from the dashboard."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    options: dict = {}
    if sys.platform == "win32":
        options["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
    else:
        options["start_new_session"] = True
    with open(log_path, "ab") as log:
        process = subprocess.Popen([sys.executable, "-m", "app.jobs", command, research_id], cwd=PROJECT_ROOT,
                                   stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, **options)
    return process.pid


def stop_process(pid: int) -> None:
    try:
        os.kill(pid, signal.SIGTERM)  # TerminateProcess on Windows
    except (ProcessLookupError, OSError):
        pass


class ResearchBusyError(RuntimeError):
    """Another research is running."""


class JobManager:
    def __init__(self, settings: Settings, connection: sqlite3.Connection | None = None,
                 spawn: Callable[[str, str, Path], int] = spawn_job, alive: Callable[[int | None], bool] = pid_alive,
                 stop: Callable[[int], None] = stop_process):
        self.settings = settings
        self.repo = ResearchRepository(connection or connect(settings.documents_db_path))
        self._spawn, self._alive, self._stop = spawn, alive, stop

    def log_path(self, research_id: str) -> Path:
        return self.settings.storage_dir / "jobs" / f"{research_id}.log"

    def refresh(self) -> list[ResearchRecord]:
        """The history, with runs whose process has died marked as interrupted."""
        records = self.repo.list()
        for record in records:
            if record.status == ResearchStatus.RUNNING and not self._alive(record.pid):
                self.repo.update(record.research_id, status=ResearchStatus.INTERRUPTED, pid=None)
                record.status, record.pid = ResearchStatus.INTERRUPTED, None
        return records

    def running(self) -> ResearchRecord | None:
        return next((r for r in self.refresh() if r.status == ResearchStatus.RUNNING), None)

    def start(self, question: str, language: str, params: ResearchParams) -> ResearchRecord:
        if not question.strip():
            raise ValueError("the question is empty")
        busy = self.running()
        if busy:
            raise ResearchBusyError(f"research {busy.research_id} is still running")
        record = self.repo.create(question, language, params)
        self._launch("run", record.research_id)
        return self.repo.get(record.research_id)

    def resume(self, research_id: str) -> ResearchRecord:
        record = self.repo.get(research_id)
        if record is None:
            raise KeyError(research_id)
        if not record.status.resumable:
            raise ValueError(f"research {research_id} is {record.status.value}: nothing to resume")
        busy = self.running()
        if busy:
            raise ResearchBusyError(f"research {busy.research_id} is still running")
        self.repo.update(research_id, status=ResearchStatus.RUNNING, error=None, finished_at=None)
        self._launch("resume", research_id)
        return self.repo.get(research_id)

    def cancel(self, research_id: str) -> None:
        record = self.repo.get(research_id)
        if record is None or record.status != ResearchStatus.RUNNING:
            return
        if record.pid:
            self._stop(record.pid)
        self.repo.update(research_id, status=ResearchStatus.CANCELLED, pid=None, finished_at=now())

    def _launch(self, command: str, research_id: str) -> None:
        try:
            pid = self._spawn(command, research_id, self.log_path(research_id))
        except OSError as exc:
            self.repo.update(research_id, status=ResearchStatus.FAILED, error=f"could not start: {exc}")
            raise
        self.repo.update(research_id, pid=pid, **({"current_step": "plan"} if command == "run" else {}))


# --- Progress, read without loading any model ---------------------------------------------

STEPS = [("plan", "Plan"), ("research", "Search"), ("collect", "Collect"), ("index", "Index"),
         ("analyze", "Analyze"), ("fact_check", "Fact-check"), ("write", "Write report")]
StepState = Literal["done", "current", "stopped", "pending"]


def step_states(record: ResearchRecord) -> list[tuple[str, str, StepState]]:
    """Each workflow step as done, current (running now), stopped (where a run halted) or pending."""
    if record.status == ResearchStatus.COMPLETED:
        return [(node, label, "done") for node, label in STEPS]
    order = [node for node, _ in STEPS]
    current = "research" if record.current_step == "next_cycle" else record.current_step
    position = order.index(current) if current in order else 0
    here: StepState = "current" if record.status == ResearchStatus.RUNNING else "stopped"
    return [(node, label, "done" if i < position else here if i == position else "pending")
            for i, (node, label) in enumerate(STEPS)]



def read_state(settings: Settings, research_id: str) -> dict:
    """The last checkpointed state of a research (plain JSON), or {} if none."""
    from langgraph.checkpoint.sqlite import SqliteSaver

    if not settings.checkpoints_db_path.exists():
        return {}
    connection = sqlite3.connect(settings.checkpoints_db_path, timeout=30, check_same_thread=False)
    try:
        saved = SqliteSaver(connection).get_tuple({"configurable": {"thread_id": research_id}})
    finally:
        connection.close()
    return dict(saved.checkpoint.get("channel_values", {})) if saved else {}


# --- Job process -------------------------------------------------------------------------


def run_job(research_id: str, resume: bool = False, base: Settings | None = None, runner_factory=None) -> int:
    """Body of the background process: run the research and keep its record up to date."""
    from app.config import get_settings
    from app.graph.workflow import build_runner

    base = base or get_settings()
    repo = ResearchRepository(connect(base.documents_db_path))
    record = repo.get(research_id)
    if record is None:
        print(f"unknown research {research_id}", file=sys.stderr)
        return 2
    repo.update(research_id, status=ResearchStatus.RUNNING, pid=os.getpid(), error=None)
    try:
        runner = (runner_factory or build_runner)(research_settings(base, record.params))
        if resume:
            pending = runner.pending(research_id)
            if pending:
                repo.update(research_id, current_step=pending[0])  # where the run picks up again
            steps = runner.resume_stream(research_id)
        else:
            steps = runner.stream(record.question, record.language, research_id)
        for node, update in steps:
            for e in (update or {}).get("events", []):
                print(f"{e['node']:<10} {e['message']}", flush=True)
            pending = runner.pending(research_id)
            repo.update(research_id, current_step=pending[0] if pending else node)
        state = runner.state(research_id)
    except Exception as exc:
        repo.update(research_id, status=ResearchStatus.FAILED, error=f"{type(exc).__name__}: {exc}"[:500],
                    pid=None, finished_at=now())
        raise
    repo.update(research_id, status=ResearchStatus.COMPLETED, current_step=None, pid=None, finished_at=now(),
                stop_reason=state.get("stop_reason"))
    return 0


def register_existing(settings: Settings, repo: ResearchRepository) -> int:
    """Add to the history the reports of researches started from the command line."""
    from app.reports.export import load_report

    added = 0
    if not settings.reports_dir.exists():
        return 0
    for path in sorted(settings.reports_dir.glob("*/report.json")):
        research_id = path.parent.name
        if repo.get(research_id):
            continue
        try:
            report = load_report(path)
        except ValueError:
            continue
        repo.create(report.question, report.language, ResearchParams(), research_id=research_id,
                    status=ResearchStatus.COMPLETED)
        repo.update(research_id, created_at=report.generated_at, finished_at=report.generated_at,
                    stop_reason=report.stop_reason)  # start unknown: no duration shown
        added += 1
    return added


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a research in the background (used by the dashboard).")
    parser.add_argument("command", choices=["run", "resume"])
    parser.add_argument("research_id")
    args = parser.parse_args(argv)
    return run_job(args.research_id, resume=args.command == "resume")


if __name__ == "__main__":
    raise SystemExit(main())
