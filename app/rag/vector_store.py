"""Chunk vectors in Qdrant (local, embedded mode: no server, data under storage/qdrant)."""

from __future__ import annotations

import uuid

from qdrant_client import QdrantClient, models

from app.rag.chunking import Chunk


def point_id(chunk_id: str) -> str:
    """Qdrant ids must be integers or UUIDs: derive a stable UUID from the chunk id."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"deepresearch:{chunk_id}"))


class VectorStore:
    def __init__(self, client: QdrantClient, collection: str, dimension: int):
        self._client, self._collection = client, collection
        if not client.collection_exists(collection):
            client.create_collection(
                collection,
                vectors_config=models.VectorParams(size=dimension, distance=models.Distance.COSINE),
            )
        else:
            existing = client.get_collection(collection).config.params.vectors.size
            if existing != dimension:
                raise ValueError(
                    f"collection {collection!r} stores {existing}-dim vectors, the model produces "
                    f"{dimension}: use another QDRANT_COLLECTION after changing the embedding model"
                )

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("one vector per chunk is required")
        points = [
            models.PointStruct(
                id=point_id(c.chunk_id),
                vector=v,
                payload={"chunk_id": c.chunk_id, "source_id": c.source_id, "research_id": c.research_id},
            )
            for c, v in zip(chunks, vectors)
        ]
        if points:
            self._client.upsert(self._collection, points=points)

    def search(self, vector: list[float], research_id: str, limit: int) -> list[tuple[str, float]]:
        """(chunk_id, cosine similarity), best first, restricted to one research."""
        hits = self._client.query_points(
            self._collection,
            query=vector,
            query_filter=models.Filter(must=[
                models.FieldCondition(key="research_id", match=models.MatchValue(value=research_id))
            ]),
            limit=limit,
            with_payload=True,
        ).points
        return [(h.payload["chunk_id"], float(h.score)) for h in hits]

    def count(self, research_id: str) -> int:
        return self._client.count(
            self._collection,
            count_filter=models.Filter(must=[
                models.FieldCondition(key="research_id", match=models.MatchValue(value=research_id))
            ]),
        ).count
