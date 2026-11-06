"""Cross-encoder reranking: reads the question and each passage together, so it judges
relevance more precisely than vector similarity, at a higher cost (used on few candidates)."""

from __future__ import annotations

from typing import Protocol


class Reranker(Protocol):
    def score(self, query: str, passages: list[str]) -> list[float]: ...


class CrossEncoderReranker:
    def __init__(self, model_name: str, device: str = "cpu"):
        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(model_name, device=device)

    def score(self, query: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        scores = self._model.predict([(query, p) for p in passages], show_progress_bar=False)
        return [float(s) for s in scores]
