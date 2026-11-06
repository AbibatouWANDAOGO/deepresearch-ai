"""Text embeddings with a multilingual E5 model (runs locally on CPU).

E5 models expect a prefix: "query: " for questions and "passage: " for documents.
Vectors are L2-normalized, so cosine similarity equals the dot product.
"""

from __future__ import annotations

from typing import Protocol

QUERY_PREFIX = "query: "
PASSAGE_PREFIX = "passage: "


class Embedder(Protocol):
    dimension: int
    max_tokens: int

    def count_tokens(self, text: str) -> int: ...

    def embed_passages(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class E5Embedder:
    def __init__(self, model_name: str, batch_size: int = 16, device: str = "cpu"):
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name, device=device)
        self._batch_size = batch_size
        self.dimension = int(self._model.get_embedding_dimension())
        self.max_tokens = int(self._model.max_seq_length)

    def count_tokens(self, text: str) -> int:
        return len(self._model.tokenizer.encode(text, add_special_tokens=False))

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(
            [PASSAGE_PREFIX + t for t in texts], batch_size=self._batch_size,
            normalize_embeddings=True, show_progress_bar=False,
        )
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        return self._model.encode(QUERY_PREFIX + text, normalize_embeddings=True).tolist()


def check_chunk_fits(chunk_size: int, embedder: Embedder) -> None:
    """Refuse chunks the model would truncate (prefix and special tokens included)."""
    budget = embedder.max_tokens - embedder.count_tokens(PASSAGE_PREFIX) - 2
    if chunk_size > budget:
        raise ValueError(
            f"CHUNK_SIZE_TOKENS={chunk_size} exceeds what the embedding model reads "
            f"({budget} tokens after the prefix): chunks would be silently truncated"
        )
