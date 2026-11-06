"""Test doubles shared by agent tests."""

from datetime import UTC, datetime

from app.database.models import AccessStatus, Source
from app.llm import CostTracker
from app.rag.chunking import Chunk
from app.rag.retriever import Passage


class FakeLLM:
    """Returns scripted answers in order and records every prompt."""

    name, model = "fake", "fake-model"

    def __init__(self, *answers):
        self.answers = list(answers)
        self.prompts: list[tuple[str, str, type]] = []
        self.tracker = CostTracker(0.0)

    def structured(self, system, user, schema):
        self.prompts.append((system, user, schema))
        answer = self.answers.pop(0)
        return answer if isinstance(answer, schema) else schema.model_validate(answer)


def make_source(source_id: str, title: str | None = None, url: str | None = None, **extra) -> Source:
    return Source(source_id=source_id, research_id="r1", url=url or f"https://{source_id}.org",
                  final_url=url or f"https://{source_id}.org", title=title or source_id.title(),
                  collected_at=datetime(2026, 11, 7, tzinfo=UTC), source_type="web_page",
                  access_status=AccessStatus.OK, **extra)


def make_passage(source_id: str, index: int, text: str, page: int | None = None) -> Passage:
    chunk = Chunk(chunk_id=f"{source_id}-{index:04d}", source_id=source_id, research_id="r1", chunk_index=index,
                  text=text, n_tokens=len(text.split()), start_char=0, end_char=len(text),
                  page_start=page, page_end=page)
    return Passage(chunk=chunk, source=make_source(source_id), rrf_score=0.5)
