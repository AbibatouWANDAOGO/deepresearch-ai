"""Evaluation metrics, computed from what a research leaves behind: its checkpointed state
(claims, verdicts, passages shown), its report, its sources and chunks, and the timings and
token usage recorded while it ran. No metric needs to run the research again.

- Coverage: share of the expected key points found in the report body (only claims that were
  supported end up there), and anywhere in the research (unverified claims included).
- Retrieval: for each checked claim, is the chunk it was extracted from among the passages
  retrieved to check it? (recall and mean reciprocal rank). The origin chunk is a free,
  objective relevance label.
- Citations: for each claim the Fact Checker supported, does the cited passage really state
  it? Three signals: every number of the claim appears in the passage; share of the claim's
  words found in the best passage; share of the cited passages that contain most of the claim's
  words (over-citation lowers it); the verdict of a stricter judge (model or manual annotation).
- Latency and cost: seconds per step, model calls and tokens.
"""

from __future__ import annotations

import re
import unicodedata
from statistics import mean
from typing import Literal
from urllib.parse import urlparse

from pydantic import BaseModel

from app.agents.evidence import numbers, word_overlap
from app.agents.fact_checker import CheckedClaim, Status
from app.database.models import Source
from app.rag.chunking import Chunk

Verdict = Literal["SUPPORTED", "PARTIAL", "NOT_SUPPORTED"]
VERDICT_SCORE: dict[str, float] = {"SUPPORTED": 1.0, "PARTIAL": 0.5, "NOT_SUPPORTED": 0.0}
# A cited passage "states" a claim when it contains most of its words.
STATES_CLAIM_OVERLAP = 0.6
SUPPORTED_STATUSES = {Status.VERIFIED, Status.PARTIALLY_VERIFIED, Status.CONFLICTING}


class KeyPoint(BaseModel):
    name: str
    any_of: list[str]


class EvalQuestion(BaseModel):
    id: str
    question: str
    language: str = "en"
    key_points: list[KeyPoint]


# --- Text helpers --------------------------------------------------------------------------


def normalize(text: str) -> str:
    """Lowercase, no accents, straight apostrophes, single spaces."""
    text = unicodedata.normalize("NFKD", text.replace("’", "'").replace(" ", " ").replace(" ", " "))
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", text.lower())


def term_found(term: str, text: str) -> bool:
    """``term`` at the start of a word of ``text`` (both normalized): 'chemistr' matches
    'chemistry', 'cop' does not match 'scope'."""
    return re.search(r"(?<![a-z0-9])" + re.escape(normalize(term)), text) is not None


def ratio(part: float, whole: float) -> float | None:
    return round(part / whole, 3) if whole else None


def avg(values: list[float]) -> float | None:
    return round(mean(values), 3) if values else None


# --- Per question --------------------------------------------------------------------------


class ClaimScore(BaseModel):
    claim_id: str
    text: str
    status: str
    numbers_ok: bool
    word_overlap: float
    independent: bool  # supported by a passage from another source than the one it came from
    passages_cited: int
    passages_stating: int  # cited passages that contain most of the claim's words
    verdict: Verdict | None = None


def coverage(question: EvalQuestion, text: str) -> tuple[list[str], list[str]]:
    text = normalize(text)
    found = [kp.name for kp in question.key_points if any(term_found(t, text) for t in kp.any_of)]
    return found, [kp.name for kp in question.key_points if kp.name not in found]


def report_text(report: dict) -> str:
    parts = [report.get("executive_summary", "")]
    parts += [s["text"] for section in report.get("sections", []) for s in section.get("statements", [])]
    return "\n".join(parts)


def research_text(state: dict) -> str:
    """Everything the research found, verified or not."""
    parts = [report_text(state.get("report") or {})]
    for analysis in state.get("analyses", {}).values():
        parts.append(analysis.get("summary", ""))
        parts += [c["text"] for c in analysis.get("claims", [])]
    return "\n".join(parts)


def checked_claims(state: dict) -> list[CheckedClaim]:
    return [CheckedClaim.model_validate(c) for claims in state.get("checked", {}).values() for c in claims]


def score_claim(claim: CheckedClaim, chunks: dict[str, Chunk], verdict: str | None = None) -> ClaimScore:
    texts = [chunks[cid].text for cid in claim.supporting_chunk_ids if cid in chunks]
    cited = numbers(" ".join(texts))
    overlaps = [word_overlap(claim.claim.text, t) for t in texts]
    return ClaimScore(
        claim_id=claim.claim.claim_id, text=claim.claim.text, status=claim.status.value,
        numbers_ok=numbers(claim.claim.text) <= cited,
        word_overlap=round(max(overlaps, default=0.0), 3),
        independent=bool(set(claim.supporting_source_ids) - set(claim.claim.source_ids)),
        passages_cited=len(texts), passages_stating=sum(o >= STATES_CLAIM_OVERLAP for o in overlaps),
        verdict=verdict,
    )


def retrieval(claims: list[CheckedClaim]) -> dict:
    """Where the chunk a claim was extracted from ranks among the passages shown to check it."""
    reciprocal = []
    for c in claims:
        if c.skipped or not c.claim.chunk_ids:
            continue
        ranks = [int(label[1:]) for label, cid in c.passage_map.items() if cid in c.claim.chunk_ids]
        reciprocal.append(1 / min(ranks) if ranks else 0.0)
    return {"claims": len(reciprocal), "origin_recall": ratio(sum(r > 0 for r in reciprocal), len(reciprocal)),
            "origin_mrr": avg(reciprocal)}


def score_question(question: EvalQuestion, state: dict, chunks: dict[str, Chunk], sources: list[Source],
                   run: dict | None = None, verdicts: dict[str, str] | None = None) -> dict:
    run, verdicts = run or {}, verdicts or {}
    report = state.get("report") or {}
    claims = checked_claims(state)
    checked = [c for c in claims if not c.skipped]
    supported = [score_claim(c, chunks, verdicts.get(c.claim.claim_id)) for c in checked
                 if c.status in SUPPORTED_STATUSES and c.supporting_chunk_ids]
    judged = [VERDICT_SCORE[s.verdict] for s in supported if s.verdict]
    found, missing = coverage(question, report_text(report))
    found_anywhere, _ = coverage(question, research_text(state))
    usable = [s for s in sources if s.usable]
    sections = report.get("sections", [])
    statuses = {s.value: sum(c.status == s for c in checked) for s in Status}
    return {
        "id": question.id,
        "research_id": state.get("research_id"),
        "language": question.language,
        "completed": bool(report),
        "stop_reason": state.get("stop_reason"),
        "error": run.get("error"),
        "coverage": {"report": ratio(len(found), len(question.key_points)),
                     "research": ratio(len(found_anywhere), len(question.key_points)),
                     "missing": missing},
        "answered_sub_questions": ratio(sum(any(s["status"] != "UNVERIFIED" for s in sec["statements"])
                                            for sec in sections), len(sections)),
        "sources": {"tried": len(sources), "usable": len(usable), "fetch_rate": ratio(len(usable), len(sources)),
                    "domains": len({urlparse(s.final_url or s.url).netloc for s in usable}),
                    "chunks": len(chunks), "cited": len(report.get("bibliography", []))},
        "claims": {"extracted": len(claims), "checked": len(checked), "skipped": len(claims) - len(checked),
                   **statuses, "supported": len(supported)},
        "retrieval": retrieval(claims),
        "citations": {"supported_claims": len(supported),
                      "numbers_ok": ratio(sum(s.numbers_ok for s in supported), len(supported)),
                      "word_overlap": avg([s.word_overlap for s in supported]),
                      "independent": ratio(sum(s.independent for s in supported), len(supported)),
                      "passage_precision": ratio(sum(s.passages_stating for s in supported),
                                                 sum(s.passages_cited for s in supported)),
                      "judged": len(judged), "precision": avg(judged),
                      "details": [s.model_dump() for s in supported]},
        "report": {"summary_origin": report.get("summary_origin"),
                   "removed_summary_sentences": report.get("removed_sentences", 0)},
        "latency": {"total_seconds": run.get("seconds"), "by_step": run.get("steps", {})},
        "cost": run.get("usage", {}),
    }


# --- Whole run -----------------------------------------------------------------------------


def aggregate(results: list[dict]) -> dict:
    done = [r for r in results if r["completed"]]
    details = [d for r in done for d in r["citations"]["details"]]
    judged = [VERDICT_SCORE[d["verdict"]] for d in details if d.get("verdict")]
    by_step: dict[str, list[float]] = {}
    for r in done:
        for step, seconds in r["latency"]["by_step"].items():
            by_step.setdefault(step, []).append(seconds)
    seconds = [r["latency"]["total_seconds"] for r in done if r["latency"]["total_seconds"]]
    retrieval_claims = sum(r["retrieval"]["claims"] for r in done)

    def mean_of(path: tuple[str, str]) -> float | None:
        return avg([r[path[0]][path[1]] for r in done if r[path[0]][path[1]] is not None])

    def total(key: str) -> float:
        return sum(r["cost"].get(key, 0) for r in results)

    return {
        "questions": len(results), "completed": len(done),
        "coverage_report": mean_of(("coverage", "report")),
        "coverage_research": mean_of(("coverage", "research")),
        "answered_sub_questions": avg([r["answered_sub_questions"] for r in done
                                       if r["answered_sub_questions"] is not None]),
        "fetch_rate": mean_of(("sources", "fetch_rate")),
        "sources_cited": avg([r["sources"]["cited"] for r in done]),
        "claims_checked": sum(r["claims"]["checked"] for r in done),
        "claims_supported": len(details),
        "origin_recall": ratio(sum((r["retrieval"]["origin_recall"] or 0) * r["retrieval"]["claims"] for r in done),
                               retrieval_claims),
        "origin_mrr": mean_of(("retrieval", "origin_mrr")),
        "numbers_ok": ratio(sum(d["numbers_ok"] for d in details), len(details)),
        "word_overlap": avg([d["word_overlap"] for d in details]),
        "independent": ratio(sum(d["independent"] for d in details), len(details)),
        "passages_per_claim": avg([d["passages_cited"] for d in details]),
        "passage_precision": ratio(sum(d["passages_stating"] for d in details),
                                   sum(d["passages_cited"] for d in details)),
        "citation_precision": avg(judged), "judged": len(judged),
        "seconds_mean": avg(seconds), "seconds_max": max(seconds) if seconds else None,
        "seconds_by_step": {k: avg(v) for k, v in by_step.items()},
        "llm_calls": total("calls"), "input_tokens": total("input_tokens"),
        "output_tokens": total("output_tokens"), "cost_usd": round(total("cost_usd"), 4),
    }
