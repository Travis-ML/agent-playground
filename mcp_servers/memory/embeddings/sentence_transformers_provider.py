"""Local, in-process embeddings via sentence-transformers."""

from __future__ import annotations


class SentenceTransformersProvider:
    def __init__(self, model_id: str = "nomic-ai/nomic-embed-text-v1.5") -> None:
        from sentence_transformers import SentenceTransformer

        self.model_id = model_id
        self._model = SentenceTransformer(model_id, trust_remote_code=True)
        self.dim = int(self._model.get_sentence_embedding_dimension())

    def embed(self, text: str) -> list[float]:
        return [float(x) for x in self._model.encode(text, normalize_embeddings=True)]

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        arr = self._model.encode(texts, normalize_embeddings=True)
        return [[float(x) for x in row] for row in arr]
