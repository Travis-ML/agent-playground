"""EmbeddingProvider protocol."""

from __future__ import annotations

from typing import Protocol


class EmbeddingProvider(Protocol):
    dim: int
    model_id: str

    def embed(self, text: str) -> list[float]: ...

    def embed_many(self, texts: list[str]) -> list[list[float]]: ...
