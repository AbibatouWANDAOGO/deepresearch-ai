"""Hybrid retrieval: dense vectors + BM25, Reciprocal Rank Fusion, cross-encoder reranking.

    python -m app.rag.retriever --research-id demo-batteries --query "Why did battery prices fall?"

Dense search finds passages that mean the same thing in other words; BM25 catches exact
terms (names, figures, acronyms) that embeddings can blur. Their rankings are merged with
Reciprocal Rank Fusion, which only uses ranks, so the two incompatible score scales never
have to be compared. A cross-encoder then reorders the best candidates.
"""

from __future__ import annotations

import argparse
import atexit
import math
import re
import sys
import unicodedata
from collections import Counter

from pydantic import BaseModel

from app.config import Settings, get_settings
from app.database.connection import connect
from app.database.models import ChunkRepository, Source, SourceRepository
from app.rag.chunking import Chunk, chunk_text
from app.rag.embeddings import Embedder, check_chunk_fits
from app.rag.reranker import Reranker
from app.rag.vector_store import VectorStore
from app.tools.document_parser import load_text

STOPWORDS = {"a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "have", "in", "is", "it", "its", "of", "on", "or", "that", "the", "this", "to", "was", "were", "will", "with", "au", "aux", "avec", "ce", "ces", "dans", "de", "des", "du", "elle", "en", "est", "et", "il", "ils", "la", "le", "les", "leur", "mais", "ne", "ou", "par", "pas", "pour", "qu", "que", "qui", "sa", "se", "ses", "son", "sont", "sur", "un", "une"}


def bm25_tokens(text: str) -> list[str]:
    """Lowercase, accent-insensitive word tokens without stop words or single characters."""
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return [t for t in re.findall(r"\w+", text) if len(t) > 1 and t not in STOPWORDS]


class BM25:
    """Okapi BM25 over a fixed list of documents."""

    def __init__(self, documents: list[str], k1: float = 1.5, b: float = 0.75):
        self._docs = [Counter(bm25_tokens(d)) for d in documents]
        self._lengths = [sum(d.values()) for d in self._docs]
        self._avg = (sum(self._lengths) / len(self._lengths)) if self._docs else 0.0
        df = Counter(term for d in self._docs for term in d)
        n = len(self._docs)
        self._idf = {t: math.log((n - f + 0.5) / (f + 0.5) + 1) for t, f in df.items()}
        self._k1, self._b = k1, b

    def scores(self, query: str) -> list[float]:
        terms = bm25_tokens(query)
        out = []
        for doc, length in zip(self._docs, self._lengths):
            score = 0.0
            for t in terms:
                tf = doc.get(t, 0)
                if tf:
                    norm = tf + self._k1 * (1 - self._b + self._b * length / (self._avg or 1))
                    score += self._idf[t] * tf * (self._k1 + 1) / norm
            out.append(score)
        return out


def reciprocal_rank_fusion(rankings: list[list[str]], k: int = 60) -> dict[str, float]:
    """score(d) = sum over rankings of 1 / (k + rank of d), ranks starting at 1."""
    fused: dict[str, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            fused[item] = fused.get(item, 0.0) + 1.0 / (k + rank)
    return fused


class Passage(BaseModel):
    """A retrieved chunk with everything needed to cite it."""

    chunk: Chunk
    source: Source
    dense_rank: int | None = None
    bm25_rank: int | None = None
    rrf_score: float
    rerank_score: float | None = None

    @property
    def location(self) -> str:
        if self.chunk.page_start is None:
            return f"chars {self.chunk.start_char}-{self.chunk.end_char}"
        if self.chunk.page_start == self.chunk.page_end:
            return f"p. {self.chunk.page_start}"
        return f"pp. {self.chunk.page_start}-{self.chunk.page_end}"


class HybridRetriever:
    def __init__(self, chunks: ChunkRepository, sources: SourceRepository, store: VectorStore,
                 embedder: Embedder, reranker: Reranker | None, settings: Settings):
        self.chunks, self.sources, self.store = chunks, sources, store
        self.embedder, self.reranker, self.settings = embedder, reranker, settings

    def chunk_map(self, research_id: str) -> dict[str, Chunk]:
        """Every chunk of a research by id, to trace citations back to their passage."""
        return {c.chunk_id: c for c in self.chunks.list(research_id)}

    def index(self, research_id: str) -> int:
        """Chunk, embed and store every usable source not indexed yet. Returns new chunks."""
        check_chunk_fits(self.settings.chunk_size_tokens, self.embedder)
        done = self.chunks.indexed_sources(research_id)
        created = 0
        for source in self.sources.list(research_id):
            if not source.usable or source.source_id in done:
                continue
            pieces = chunk_text(
                load_text(source, self.settings), source.source_id, research_id,
                self.settings.chunk_size_tokens, self.settings.chunk_overlap_tokens,
                count=self.embedder.count_tokens, paged=source.source_type == "pdf",
            )
            vectors = self.embedder.embed_passages([c.text for c in pieces])
            self.store.upsert(pieces, vectors)
            self.chunks.save_many(pieces)  # saved last: a crash leaves the source "not indexed"
            created += len(pieces)
        return created

    def retrieve(self, query: str, research_id: str) -> list[Passage]:
        chunks = self.chunks.list(research_id)
        if not chunks or not query.strip():
            return []
        by_id = {c.chunk_id: c for c in chunks}
        top_k = self.settings.retrieval_top_k

        dense = [cid for cid, _ in self.store.search(self.embedder.embed_query(query), research_id, top_k)
                 if cid in by_id]
        bm25_scores = BM25([c.text for c in chunks]).scores(query)
        lexical = [chunks[i].chunk_id for i in sorted(range(len(chunks)), key=lambda i: -bm25_scores[i])
                   if bm25_scores[i] > 0][:top_k]

        fused = reciprocal_rank_fusion([dense, lexical], self.settings.rrf_k)
        candidates = sorted(fused, key=lambda cid: -fused[cid])[:top_k]
        dense_rank = {cid: r for r, cid in enumerate(dense, 1)}
        lexical_rank = {cid: r for r, cid in enumerate(lexical, 1)}
        sources = {s.source_id: s for s in self.sources.list(research_id)}

        passages = [
            Passage(chunk=by_id[cid], source=sources[by_id[cid].source_id], dense_rank=dense_rank.get(cid),
                    bm25_rank=lexical_rank.get(cid), rrf_score=fused[cid])
            for cid in candidates
        ]
        if self.reranker is not None and self.settings.rerank_enabled:
            scores = self.reranker.score(query, [p.chunk.text for p in passages])
            for passage, score in zip(passages, scores):
                passage.rerank_score = score
            passages.sort(key=lambda p: -(p.rerank_score or 0.0))
        return passages[: self.settings.rerank_top_k]


def build_retriever(settings: Settings) -> HybridRetriever:
    from qdrant_client import QdrantClient

    from app.rag.embeddings import E5Embedder
    from app.rag.reranker import CrossEncoderReranker

    embedder = E5Embedder(settings.embedding_model, settings.embedding_batch_size)
    reranker = CrossEncoderReranker(settings.reranker_model) if settings.rerank_enabled else None
    settings.qdrant_dir.mkdir(parents=True, exist_ok=True)
    client = QdrantClient(path=str(settings.qdrant_dir))
    atexit.register(client.close)  # close before interpreter shutdown, not in a late __del__
    store = VectorStore(client, settings.qdrant_collection, embedder.dimension)
    db = connect(settings.documents_db_path)
    return HybridRetriever(ChunkRepository(db), SourceRepository(db), store, embedder, reranker, settings)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Index a research's sources and retrieve passages.")
    parser.add_argument("--research-id", required=True)
    parser.add_argument("--query", required=True)
    args = parser.parse_args(argv)

    settings = get_settings()
    retriever = build_retriever(settings)
    try:
        created = retriever.index(args.research_id)
    except ValueError as exc:
        print(f"Indexing failed: {exc}", file=sys.stderr)
        return 1
    total = len(retriever.chunks.list(args.research_id))
    print(f"Indexed {created} new chunks ({total} in research {args.research_id})")
    for i, p in enumerate(retriever.retrieve(args.query, args.research_id), start=1):
        rerank = f"{p.rerank_score:+.2f}" if p.rerank_score is not None else "-"
        print(f"\n{i}. rerank {rerank} | dense #{p.dense_rank or '-'} | bm25 #{p.bm25_rank or '-'} | "
              f"{p.source.title or p.source.url} ({p.location})")
        print("   " + p.chunk.text[:300].replace("\n", " "))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
