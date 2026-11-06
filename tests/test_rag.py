import hashlib
import itertools
import math
from datetime import UTC, datetime
from pathlib import Path

import pytest
from qdrant_client import QdrantClient

from app.config import Settings
from app.database.connection import connect
from app.database.models import AccessStatus, ChunkRepository, Source, SourceRepository
from app.rag.chunking import chunk_text, whitespace_token_count
from app.rag.embeddings import check_chunk_fits
from app.rag.retriever import BM25, HybridRetriever, bm25_tokens, reciprocal_rank_fusion
from app.rag.vector_store import VectorStore, point_id

# --- Chunking ----------------------------------------------------------------------------

TEXT = " ".join(f"Sentence number {i} talks about topic {i % 3}." for i in range(1, 41))


def test_chunks_respect_size_and_cover_the_text():
    chunks = chunk_text(TEXT, "s1", "r1", chunk_size=30, overlap=8)
    assert len(chunks) > 3
    assert all(c.n_tokens <= 30 for c in chunks)
    assert chunks[0].start_char == 0 and chunks[-1].end_char == len(TEXT)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    assert chunks[1].chunk_id == "s1-0001"


def test_offsets_point_to_the_exact_passage():
    for c in chunk_text(TEXT, "s1", "r1", chunk_size=30, overlap=8):
        assert TEXT[c.start_char:c.end_char] == c.text


def test_consecutive_chunks_overlap_by_whole_sentences():
    chunks = chunk_text(TEXT, "s1", "r1", chunk_size=30, overlap=8)
    for previous, current in itertools.pairwise(chunks):
        assert current.start_char < previous.end_char  # overlap exists
        overlap = TEXT[current.start_char:previous.end_char]
        assert overlap.startswith("Sentence") and overlap.endswith(".")  # whole sentences
        assert whitespace_token_count(overlap) <= 8


def test_no_overlap_means_contiguous_chunks():
    chunks = chunk_text(TEXT, "s1", "r1", chunk_size=30, overlap=0)
    for previous, current in itertools.pairwise(chunks):
        assert current.start_char > previous.end_char - 1


def test_long_sentence_is_split_on_words():
    long_sentence = " ".join(f"w{i}" for i in range(100))
    chunks = chunk_text(long_sentence, "s1", "r1", chunk_size=30, overlap=5)
    assert all(c.n_tokens <= 30 for c in chunks)
    assert " ".join(c.text for c in chunks).split()[:30] == long_sentence.split()[:30]


def test_pdf_chunks_know_their_pages():
    pages = ["First page sentence one. First page sentence two.",
             "Second page sentence one. Second page sentence two.",
             "Third page only sentence."]
    text = "\f".join(pages)
    chunks = chunk_text(text, "s1", "r1", chunk_size=9, overlap=0)
    assert chunks[0].page_start == 1
    assert chunks[-1].page_end == 3
    assert any(c.page_start != c.page_end for c in chunks) or len(chunks) >= 3
    assert all("\f" not in c.text for c in chunks)


def test_web_chunks_have_no_pages():
    assert chunk_text(TEXT, "s1", "r1", 30, 5)[0].page_start is None


def test_empty_text_and_invalid_overlap():
    assert chunk_text("   \n  ", "s1", "r1", 30, 5) == []
    with pytest.raises(ValueError):
        chunk_text(TEXT, "s1", "r1", chunk_size=10, overlap=10)


# --- BM25 and fusion ---------------------------------------------------------------------


def test_bm25_tokens_ignore_case_accents_and_stopwords():
    assert bm25_tokens("Les Énergies renouvelables et the Solar PANELS") == ["energies", "renouvelables", "solar", "panels"]


def test_bm25_ranks_the_document_with_the_rare_term_first():
    docs = ["solar panels convert sunlight", "wind turbines convert wind", "battery storage for solar energy",
            "the price of lithium fell by 89 percent"]
    scores = BM25(docs).scores("lithium price")
    assert max(range(4), key=lambda i: scores[i]) == 3
    assert scores[1] == 0


def test_bm25_prefers_shorter_documents_at_equal_term_frequency():
    scores = BM25(["lithium battery", "lithium battery " + "filler words here " * 20]).scores("lithium")
    assert scores[0] > scores[1]


def test_reciprocal_rank_fusion():
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["c", "a", "d"]], k=60)
    assert fused["a"] == pytest.approx(1 / 61 + 1 / 62)
    assert fused["d"] == pytest.approx(1 / 63)
    assert max(fused, key=fused.get) == "a"  # good in both lists beats first in one


# --- Fakes -------------------------------------------------------------------------------


class FakeEmbedder:
    """Bag-of-words vectors hashed into 64 dimensions: similar words, similar vectors."""

    dimension, max_tokens = 64, 512

    def count_tokens(self, text):
        return len(text.split())

    def _vector(self, text):
        v = [0.0] * self.dimension
        for token in bm25_tokens(text):
            v[int(hashlib.md5(token.encode()).hexdigest(), 16) % self.dimension] += 1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    def embed_passages(self, texts):
        return [self._vector(t) for t in texts]

    def embed_query(self, text):
        return self._vector(text)


class OverlapReranker:
    def score(self, query, passages):
        q = set(bm25_tokens(query))
        return [len(q & set(bm25_tokens(p))) for p in passages]


DOCS = {
    "batteries": "Lithium-ion battery pack prices fell by 89 percent between 2010 and 2020. "
                 "Cheaper cathode materials and larger factories drove the decline.",
    "solar": "Solar panel efficiency improved steadily. Module prices dropped as production scaled up in Asia.",
    "wind": "Offshore wind turbines became larger. Capacity factors increased with taller towers.",
}


@pytest.fixture
def rag(tmp_path, monkeypatch):
    for name in Settings.model_fields:
        monkeypatch.delenv(name.upper(), raising=False)
    settings = Settings(_env_file=None, storage_dir=tmp_path, chunk_size_tokens=50, chunk_overlap_tokens=5,
                        retrieval_top_k=10, rerank_top_k=3)
    db = connect(tmp_path / "docs.db")
    sources = SourceRepository(db)
    for name, text in DOCS.items():
        path = Path("documents") / "r1" / f"{name}.txt"
        (tmp_path / path).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / path).write_text(text, encoding="utf-8")
        sources.save(Source(source_id=name, research_id="r1", url=f"https://{name}.org", final_url=f"https://{name}.org",
                            title=name.title(), collected_at=datetime.now(UTC), source_type="web_page",
                            access_status=AccessStatus.OK, n_chars=len(text), text_path=path.as_posix()))
    sources.save(Source(source_id="broken", research_id="r1", url="https://broken.org", final_url="https://broken.org",
                        collected_at=datetime.now(UTC), access_status=AccessStatus.HTTP_ERROR))
    embedder = FakeEmbedder()
    store = VectorStore(QdrantClient(":memory:"), "chunks", embedder.dimension)
    return HybridRetriever(ChunkRepository(db), sources, store, embedder, OverlapReranker(), settings)


# --- Indexing and retrieval --------------------------------------------------------------


def test_indexing_covers_usable_sources_only_and_is_idempotent(rag):
    created = rag.index("r1")
    assert created >= 3
    assert rag.chunks.indexed_sources("r1") == {"batteries", "solar", "wind"}
    assert rag.store.count("r1") == created
    assert rag.index("r1") == 0  # nothing new on a second run


def test_retrieval_finds_the_relevant_source_first(rag):
    rag.index("r1")
    passages = rag.retrieve("Why did lithium battery prices fall?", "r1")
    assert passages[0].source.source_id == "batteries"
    assert "89 percent" in passages[0].chunk.text
    assert passages[0].dense_rank is not None and passages[0].bm25_rank is not None
    assert passages[0].rerank_score is not None


def test_retrieved_passages_carry_citation_metadata(rag):
    rag.index("r1")
    p = rag.retrieve("offshore wind turbines", "r1")[0]
    assert p.source.url == "https://wind.org" and p.source.title == "Wind"
    assert p.location.startswith("chars ")
    assert len(rag.retrieve("anything", "r1")) <= 3  # rerank_top_k


def test_research_isolation(rag):
    rag.index("r1")
    assert rag.retrieve("lithium battery prices", "other-research") == []


def test_retrieval_without_reranker_uses_fusion_order(rag):
    rag.index("r1")
    rag.reranker = None
    passages = rag.retrieve("solar module prices", "r1")
    assert passages[0].source.source_id == "solar" and passages[0].rerank_score is None
    assert passages == sorted(passages, key=lambda p: -p.rrf_score)


def test_exact_number_found_by_bm25(rag):
    rag.index("r1")
    passages = rag.retrieve("2010 2020", "r1")
    assert passages[0].source.source_id == "batteries" and passages[0].bm25_rank == 1


# --- Safety checks -----------------------------------------------------------------------


def test_chunk_size_larger_than_the_model_is_refused():
    embedder = FakeEmbedder()
    check_chunk_fits(480, embedder)
    with pytest.raises(ValueError, match="silently truncated"):
        check_chunk_fits(800, embedder)


def test_collection_with_another_dimension_is_refused():
    client = QdrantClient(":memory:")
    VectorStore(client, "chunks", 64)
    with pytest.raises(ValueError, match="64-dim"):
        VectorStore(client, "chunks", 768)


def test_point_ids_are_stable_uuids():
    assert point_id("abc-0001") == point_id("abc-0001") != point_id("abc-0002")
