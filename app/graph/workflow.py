"""LangGraph workflow of a research, with checkpointing.

    python -m app.graph.workflow "Why have lithium-ion battery prices fallen?" --language en
    python -m app.graph.workflow --resume <research_id>

plan -> research -> collect -> index -> analyze -> fact_check -> (next_cycle -> research ...) -> write

After fact checking, the run loops back for sub-questions whose claims are not supported
enough, at most MAX_EXTRA_RESEARCH_CYCLES times and within MAX_RESEARCH_MINUTES. The state
is saved after every node (SQLite checkpoints, one thread per research id), so an
interrupted run resumes from its last completed step.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
import time
import uuid

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from app.graph.nodes import Dependencies, Nodes
from app.graph.state import ResearchState

# Each cycle runs 6 nodes: plenty of room for the maximum of 3 extra cycles.
RECURSION_LIMIT = 100


def build_graph(nodes: Nodes) -> StateGraph:
    graph = StateGraph(ResearchState)
    for name in ("plan", "research", "collect", "index", "analyze", "fact_check", "next_cycle", "write"):
        graph.add_node(name, getattr(nodes, name))
    graph.add_edge(START, "plan")
    graph.add_edge("plan", "research")
    graph.add_edge("research", "collect")
    graph.add_edge("collect", "index")
    graph.add_edge("index", "analyze")
    graph.add_edge("analyze", "fact_check")
    graph.add_conditional_edges("fact_check", nodes.route, {"next_cycle": "next_cycle", "write": "write"})
    graph.add_edge("next_cycle", "research")
    graph.add_edge("write", END)
    return graph


class ResearchRunner:
    def __init__(self, deps: Dependencies, checkpointer):
        self.deps = deps
        self.app = build_graph(Nodes(deps)).compile(checkpointer=checkpointer)

    @staticmethod
    def config(research_id: str) -> dict:
        return {"configurable": {"thread_id": research_id}, "recursion_limit": RECURSION_LIMIT}

    def start(self, question: str, language: str = "en", research_id: str | None = None) -> ResearchState:
        research_id = research_id or f"r-{uuid.uuid4().hex[:10]}"
        initial: ResearchState = {"research_id": research_id, "question": question.strip(),
                                  "language": language, "started_at": self.deps.clock(), "events": []}
        return self.app.invoke(initial, self.config(research_id))

    def stream(self, question: str, language: str = "en", research_id: str | None = None):
        """Yield (node name, state update) as each step finishes, for progress display."""
        research_id = research_id or f"r-{uuid.uuid4().hex[:10]}"
        initial: ResearchState = {"research_id": research_id, "question": question.strip(),
                                  "language": language, "started_at": self.deps.clock(), "events": []}
        for chunk in self.app.stream(initial, self.config(research_id), stream_mode="updates"):
            yield from chunk.items()

    def resume(self, research_id: str) -> ResearchState:
        if not self.pending(research_id):
            raise ValueError(f"research {research_id!r} has nothing left to run")
        # The duration limit applies to each session: restart the clock for the resumed run.
        self.app.update_state(self.config(research_id), {"started_at": self.deps.clock()})
        return self.app.invoke(None, self.config(research_id))

    def resume_stream(self, research_id: str):
        """Like ``resume``, yielding (node name, state update) as each step finishes."""
        if not self.pending(research_id):
            raise ValueError(f"research {research_id!r} has nothing left to run")
        self.app.update_state(self.config(research_id), {"started_at": self.deps.clock()})
        for chunk in self.app.stream(None, self.config(research_id), stream_mode="updates"):
            yield from chunk.items()

    def state(self, research_id: str) -> ResearchState:
        return self.app.get_state(self.config(research_id)).values

    def pending(self, research_id: str) -> tuple[str, ...]:
        """Nodes that would run next: empty when the research is finished or unknown."""
        return tuple(self.app.get_state(self.config(research_id)).next)


def build_runner(settings) -> ResearchRunner:
    from app.agents.analyst import Analyst
    from app.agents.fact_checker import FactChecker
    from app.agents.planner import Planner
    from app.agents.researcher import Researcher
    from app.agents.writer import Writer
    from app.llm import build_llm
    from app.rag.retriever import build_retriever
    from app.tools.document_parser import build_collector
    from app.tools.web_search import build_web_search

    settings.ensure_storage()
    llm = build_llm(settings)
    deps = Dependencies(
        planner=Planner(llm),
        researcher=Researcher(llm, build_web_search(settings), settings.max_results_per_query),
        collector=build_collector(settings),
        retriever=build_retriever(settings),
        analyst=Analyst(llm),
        fact_checker=FactChecker(llm),
        writer=Writer(llm),
        settings=settings,
    )
    connection = sqlite3.connect(settings.checkpoints_db_path, check_same_thread=False)
    return ResearchRunner(deps, SqliteSaver(connection))


def print_report(state: ResearchState) -> None:
    from app.agents.writer import Report
    from app.reports.export import to_markdown

    if state.get("report"):
        print()
        print(to_markdown(Report.model_validate(state["report"])))


def main(argv: list[str] | None = None) -> int:
    from app.config import get_settings

    parser = argparse.ArgumentParser(description="Run or resume a full research.")
    parser.add_argument("question", nargs="?")
    parser.add_argument("--language", default="en", choices=["en", "fr"])
    parser.add_argument("--research-id", default=None)
    parser.add_argument("--resume", metavar="RESEARCH_ID", default=None)
    args = parser.parse_args(argv)
    if not args.question and not args.resume:
        parser.error("give a question or --resume RESEARCH_ID")

    runner = build_runner(get_settings())
    started = time.perf_counter()
    research_id = args.resume or args.research_id or f"r-{uuid.uuid4().hex[:10]}"
    if args.resume:
        print(f"Resuming research {research_id} at: {', '.join(runner.pending(research_id)) or 'nothing'}")
        steps = runner.resume_stream(research_id)
    else:
        print(f"Research {research_id} (resume with --resume {research_id} if interrupted)")
        steps = runner.stream(args.question, args.language, research_id)
    for _, update in steps:
        for e in (update or {}).get("events", []):
            print(f"[{time.perf_counter() - started:6.0f}s] {e['node']:<10} {e['message']}", flush=True)
    state = runner.state(research_id)
    print_report(state)
    usage = runner.deps.planner.llm.tracker.usage
    print(f"\nStopped because: {state.get('stop_reason')}. {usage.calls} LLM calls, "
          f"{time.perf_counter() - started:.0f} s, ${usage.cost_usd:.4f}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
