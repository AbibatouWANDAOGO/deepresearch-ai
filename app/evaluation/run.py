"""Run the evaluation questions as researches, then score them.

    python -m app.evaluation.run run --name baseline --depth quick
    python -m app.evaluation.run score --name baseline --judge
    python -m app.evaluation.run status --name baseline

``run`` is resumable: a completed question is skipped, an interrupted one resumes from its
checkpoint. Researches appear in the dashboard history (ids ``eval-<name>-<question>``), and
the dashboard refuses to start another research while one runs. ``score`` needs no research
to run again; ``--judge`` asks the model to audit every supported claim (cached).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from app.agents.base import strip_passage_refs
from app.agents.fact_checker import CheckedClaim, check_evidence
from app.config import Settings
from app.database.connection import connect
from app.database.models import ChunkRepository, SourceRepository
from app.evaluation.metrics import (
    VERDICT_SCORE,
    EvalQuestion,
    aggregate,
    avg,
    checked_claims,
    score_question,
)
from app.jobs import (
    JobManager,
    ResearchParams,
    ResearchRepository,
    ResearchStatus,
    read_state,
    run_job,
)
from app.rag.chunking import Chunk

PROJECT_ROOT = Path(__file__).resolve().parents[2]
QUESTIONS = PROJECT_ROOT / "eval" / "questions.json"
ANNOTATIONS = PROJECT_ROOT / "eval" / "annotations.json"
RESULTS = PROJECT_ROOT / "eval" / "results"


def load_questions(path: Path = QUESTIONS) -> list[EvalQuestion]:
    return [EvalQuestion.model_validate(q) for q in json.loads(path.read_text(encoding="utf-8"))["questions"]]


def research_id(name: str, question: EvalQuestion, attempt: int = 1) -> str:
    base = f"eval-{name}-{question.id}"
    return base if attempt == 1 else f"{base}-{attempt}"


def eval_dir(settings: Settings, name: str) -> Path:
    return settings.storage_dir / "eval" / name


def read_json(path: Path, default):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def run_path(settings: Settings, name: str, question: EvalQuestion) -> Path:
    return eval_dir(settings, name) / "runs" / f"{question.id}.json"


def current_id(settings: Settings, name: str, question: EvalQuestion) -> str:
    """The research that counts for a question: the latest attempt."""
    return read_json(run_path(settings, name, question), {}).get("research_id") or research_id(name, question)


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


# --- Running -------------------------------------------------------------------------------


class TimedRunner:
    """Wraps a ResearchRunner to time each step and count the model's tokens."""

    def __init__(self, runner):
        self.runner, self.steps = runner, {}

    def __getattr__(self, name):
        return getattr(self.runner, name)

    def _timed(self, steps):
        last = time.perf_counter()
        for node, update in steps:
            now = time.perf_counter()
            self.steps[node] = round(self.steps.get(node, 0.0) + now - last, 1)
            last = now
            yield node, update

    def stream(self, *args, **kwargs):
        return self._timed(self.runner.stream(*args, **kwargs))

    def resume_stream(self, research_id_: str):
        return self._timed(self.runner.resume_stream(research_id_))

    def usage(self) -> dict:
        tracker = getattr(getattr(getattr(self.runner, "deps", None), "analyst", None), "llm", None)
        return tracker.tracker.usage.model_dump() if tracker is not None else {}


def _add(old: dict, new: dict) -> dict:
    return {k: round(old.get(k, 0) + new.get(k, 0), 4) for k in set(old) | set(new)}


def run_all(settings: Settings, questions: list[EvalQuestion], name: str, params: ResearchParams,
            runner_factory=None, log=print, rerun: set[str] = frozenset()) -> list[str]:
    """Run every question not completed yet; returns the ids that failed.

    Questions in ``rerun`` start again as a new research (``eval-<name>-<id>-2``...): the
    previous attempt stays in the history and is listed as replaced.
    """
    from app.graph.workflow import build_runner

    manager = JobManager(settings)
    busy = manager.running()
    if busy is not None:
        raise RuntimeError(f"research {busy.research_id} is running: wait for it or cancel it first")
    repo, built, failed = manager.repo, {}, []

    def factory(job_settings):  # one runner for every question: the embedded index opens once
        if "runner" not in built:
            built["runner"] = (runner_factory or build_runner)(job_settings)
        timed = TimedRunner(built["runner"])
        built["timed"] = timed
        return timed

    for i, question in enumerate(questions, start=1):
        path = run_path(settings, name, question)
        previous = read_json(path, {})
        rid = current_id(settings, name, question)
        if question.id in rerun and repo.get(rid) is not None:
            replaced = [*previous.get("replaced", []), rid]
            rid, previous = research_id(name, question, len(replaced) + 1), {"replaced": replaced}
            write_json(path, {"research_id": rid, **previous})
        record = repo.get(rid)
        if record is not None and record.status == ResearchStatus.COMPLETED:
            log(f"[{i}/{len(questions)}] {question.id}: already done")
            continue
        resume = record is not None and record.status.resumable
        if record is None:
            repo.create(question.question, question.language, params, research_id=rid,
                        status=ResearchStatus.INTERRUPTED)
        log(f"[{i}/{len(questions)}] {question.id}: {'resuming' if resume else 'starting'}")
        before = built["timed"].usage() if "timed" in built else {}
        started, error = time.perf_counter(), None
        try:
            run_job(rid, resume=resume, base=settings, runner_factory=factory)
        except Exception as exc:  # noqa: BLE001 - recorded by run_job; go on with the next question
            error = f"{type(exc).__name__}: {exc}"[:500]
            failed.append(question.id)
        timed = built.get("timed")
        usage = {k: v - before.get(k, 0) for k, v in timed.usage().items()} if timed else {}
        write_json(path, {
            "research_id": rid, "replaced": previous.get("replaced", []), "error": error, "resumed": resume or previous.get("resumed", False),
            "seconds": round(previous.get("seconds", 0) + time.perf_counter() - started, 1),
            "steps": _add(previous.get("steps", {}), timed.steps if timed else {}),
            "usage": _add(previous.get("usage", {}), usage),
        })
        log(f"    {'failed: ' + error if error else 'done'} in {time.perf_counter() - started:.0f} s")
    return failed


# --- Scoring -------------------------------------------------------------------------------


def judge_claims(settings: Settings, name: str, questions: list[EvalQuestion], llm, log=print) -> dict:
    """Model verdicts for every supported claim, cached in storage/eval/<name>/judgments.json."""
    from app.evaluation.judge import judge
    from app.evaluation.metrics import SUPPORTED_STATUSES
    from app.llm import LLMError

    path = eval_dir(settings, name) / "judgments.json"
    cache = read_json(path, {})
    chunks_repo = ChunkRepository(connect(settings.documents_db_path))
    for question in questions:
        rid = current_id(settings, name, question)
        state = read_state(settings, rid)
        if not state.get("report"):
            continue
        chunks = {c.chunk_id: c for c in chunks_repo.list(rid)}
        done = cache.setdefault(rid, {})
        for c in checked_claims(state):
            if c.skipped or c.status not in SUPPORTED_STATUSES or c.claim.claim_id in done:
                continue
            texts = [chunks[cid].text for cid in c.supporting_chunk_ids if cid in chunks]
            try:
                verdict = judge(llm, c.claim.text, texts)
            except LLMError as exc:  # one failed audit must not lose the others
                log(f"    {rid} {c.claim.claim_id}: judge failed ({exc})")
                continue
            done[c.claim.claim_id] = verdict.model_dump()
            log(f"    {rid} {c.claim.claim_id}: {verdict.verdict}")
            write_json(path, cache)
    return {rid: {cid: j["verdict"] for cid, j in claims.items()} for rid, claims in cache.items()}


# How a Fact Checker status reads against an annotation: VERIFIED claims the passages state it all.
STATUS_AS_VERDICT = {"VERIFIED": "SUPPORTED", "PARTIALLY_VERIFIED": "PARTIAL", "CONFLICTING": "PARTIAL"}


def annotation_agreement(results: list[dict], annotations: dict) -> dict:
    pairs, annotated, statuses = [], [], []
    for r in results:
        notes = annotations.get(r["research_id"], {})
        for d in r["citations"]["details"]:
            label = notes.get(d["claim_id"])
            if label in VERDICT_SCORE:
                annotated.append(VERDICT_SCORE[label])
                statuses.append(STATUS_AS_VERDICT.get(d["status"]) == label)
                if d.get("verdict"):
                    pairs.append(d["verdict"] == label)
    return {"annotated": len(annotated), "citation_precision_annotated": avg(annotated),
            "judge_agreement": avg([float(p) for p in pairs]),
            "fact_checker_agreement": avg([float(x) for x in statuses])}


def replay_evidence_rules(state: dict, chunks: dict[str, Chunk]) -> dict:
    """The research's checked claims as the current text rules would leave them (reference
    cleaning, passage filter, numbers rule), from the model verdicts already stored: the
    Fact Checker's model is not called again."""
    checked = {}
    for sq_id, claims in state.get("checked", {}).items():
        replayed = []
        for c in claims:
            item = CheckedClaim.model_validate(c)
            item.claim.text = strip_passage_refs(item.claim.text)[0] or item.claim.text
            ids = [x for x in item.supporting_chunk_ids if x in chunks]
            if not item.skipped and ids:
                status, ids = check_evidence(item.claim.text, item.status, ids, {x: chunks[x].text for x in ids})
                item.adjusted = item.adjusted or status != item.status
                item.status, item.supporting_chunk_ids = status, ids
                item.supporting_source_ids = list(dict.fromkeys(chunks[x].source_id for x in ids))
            replayed.append(item.model_dump(mode="json"))
        checked[sq_id] = replayed
    return state | {"checked": checked}


def score_all(settings: Settings, questions: list[EvalQuestion], name: str, verdicts: dict | None = None,
              annotations: dict | None = None, replay_rules: bool = False) -> tuple[list[dict], dict]:
    connection = connect(settings.documents_db_path)
    chunks_repo, sources_repo = ChunkRepository(connection), SourceRepository(connection)
    results = []
    for question in questions:
        rid = current_id(settings, name, question)
        run = read_json(run_path(settings, name, question), {})
        state = read_state(settings, rid) or {"research_id": rid}
        chunks = {c.chunk_id: c for c in chunks_repo.list(rid)}
        if replay_rules:
            state = replay_evidence_rules(state, chunks)
        results.append(score_question(question, state, chunks, sources_repo.list(rid), run,
                                      (verdicts or {}).get(rid)))
    summary = aggregate(results)
    if annotations:
        summary |= annotation_agreement(results, annotations)
    return results, summary


def pct(value: float | None) -> str:
    return "—" if value is None else f"{value * 100:.0f} %"


def minutes(seconds: float | None) -> str:
    return "—" if not seconds else f"{seconds / 60:.1f} min"


def to_markdown(name: str, params: dict, summary: dict, results: list[dict]) -> str:
    s = summary
    lines = [
        f"# Evaluation `{name}`", "",
        (f"{s['completed']} of {s['questions']} researches completed, depth `{params.get('depth')}`, "
        f"at most {params.get('max_sources')} sources each."), "",
        "| Metric | Value |", "|---|---|",
        f"| Key points in the report (coverage) | {pct(s['coverage_report'])} |",
        f"| Key points found anywhere in the research | {pct(s['coverage_research'])} |",
        f"| Sub-questions answered with evidence | {pct(s['answered_sub_questions'])} |",
        f"| Pages fetched successfully | {pct(s['fetch_rate'])} |",
        f"| Sources cited per report | {s['sources_cited']} |",
        f"| Claims checked / supported | {s['claims_checked']} / {s['claims_supported']} |",
        f"| Retrieval: origin chunk among passages checked (recall) | {pct(s['origin_recall'])} |",
        f"| Retrieval: mean reciprocal rank of the origin chunk | {s['origin_mrr']} |",
        f"| Citations: every number of the claim is in the passage | {pct(s['numbers_ok'])} |",
        f"| Citations: share of the claim's words in the passage | {pct(s['word_overlap'])} |",
        f"| Citations: supported by another source than the one it came from | {pct(s['independent'])} |",
        f"| Citations: passages cited per claim | {s['passages_per_claim'] or 0:.1f} |",
        f"| Citations: cited passages that contain most of the claim's words | {pct(s['passage_precision'])} |",
        f"| Citation precision, model judge ({s['judged']} claims) | {pct(s['citation_precision'])} |",
    ]
    if "annotated" in s:
        lines += [(f"| Citation precision, manual annotation ({s['annotated']} claims) | "
                  f"{pct(s['citation_precision_annotated'])} |"),
                  f"| Judge agrees with the annotation | {pct(s['judge_agreement'])} |",
                  f"| Fact Checker status agrees with the annotation | {pct(s['fact_checker_agreement'])} |"]
    lines += [
        f"| Duration per research, mean / max | {minutes(s['seconds_mean'])} / {minutes(s['seconds_max'])} |",
        f"| Model calls, tokens in / out | {s['llm_calls']}, {s['input_tokens']:,} / {s['output_tokens']:,} |",
        f"| Cost | ${s['cost_usd']:.2f} |", "",
        "## Time per step (mean)", "", "| Step | Time |", "|---|---|",
    ]
    lines += [f"| {step} | {minutes(sec)} |" for step, sec in s["seconds_by_step"].items()]
    lines += ["", "## Per question", "",
              "| Question | Coverage | Sources cited | Checked / supported | Citation precision | Duration | Missing |",
              "|---|---|---|---|---|---|---|"]
    for r in results:
        if not r["completed"]:
            lines.append(f"| {r['id']} | not completed: {r['error'] or 'not run'} | | | | | |")
            continue
        lines.append(f"| {r['id']} | {pct(r['coverage']['report'])} | {r['sources']['cited']} | "
                     f"{r['claims']['checked']} / {r['claims']['supported']} | {pct(r['citations']['precision'])} | "
                     f"{minutes(r['latency']['total_seconds'])} | {', '.join(r['coverage']['missing']) or '—'} |")
    return "\n".join(lines) + "\n"


# --- Command line --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    from app.config import get_settings

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["run", "score", "status"])
    parser.add_argument("--name", default="baseline", help="evaluation run name")
    parser.add_argument("--depth", choices=["quick", "standard", "deep"], default="quick")
    parser.add_argument("--max-sources", type=int, default=12)
    parser.add_argument("--only", help="comma-separated question ids")
    parser.add_argument("--rerun", default="", help="comma-separated question ids to run again from scratch")
    parser.add_argument("--judge", action="store_true", help="audit supported claims with the model")
    parser.add_argument("--replay-rules", action="store_true",
                        help="score with the current Fact Checker text rules applied to the stored verdicts")
    args = parser.parse_args(argv)

    settings = get_settings()
    settings.ensure_storage()
    questions = load_questions()
    if args.only:
        wanted = set(args.only.split(","))
        questions = [q for q in questions if q.id in wanted]
    params_path = eval_dir(settings, args.name) / "params.json"

    if args.command == "run":
        params = ResearchParams(depth=args.depth, max_sources=args.max_sources)
        write_json(params_path, params.model_dump())
        failed = run_all(settings, questions, args.name, params, log=lambda m: print(m, flush=True),
                         rerun=set(filter(None, args.rerun.split(","))))
        print(f"finished, {len(failed)} failed" + (f": {', '.join(failed)}" if failed else ""))
        return 1 if failed else 0

    if args.command == "status":
        repo = ResearchRepository(connect(settings.documents_db_path))
        for q in questions:
            record = repo.get(current_id(settings, args.name, q))
            print(f"{q.id:<24} {record.status.value if record else 'not started':<12} "
                  f"{(record.current_step or '') if record else ''}")
        return 0

    verdicts = None
    if args.judge:
        from app.llm import build_llm
        verdicts = judge_claims(settings, args.name, questions, build_llm(settings),
                                log=lambda m: print(m, flush=True))
    else:
        cached = read_json(eval_dir(settings, args.name) / "judgments.json", {})
        verdicts = {rid: {cid: j["verdict"] for cid, j in claims.items()} for rid, claims in cached.items()}
    results, summary = score_all(settings, questions, args.name, verdicts, read_json(ANNOTATIONS, {}),
                                 replay_rules=args.replay_rules)
    params = read_json(params_path, {})
    out = f"{args.name}-rules" if args.replay_rules else args.name
    write_json(RESULTS / f"{out}.json", {"name": out, "params": params, "summary": summary, "questions": results})
    markdown = to_markdown(out, params, summary, results)
    (RESULTS / f"{out}.md").write_text(markdown, encoding="utf-8")
    print(markdown)
    return 0


if __name__ == "__main__":
    sys.exit(main())
