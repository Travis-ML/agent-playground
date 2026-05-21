"""Embedding provider for any OpenAI-compatible /v1/embeddings endpoint (vLLM)."""

from __future__ import annotations

import os

import httpx


class OpenAICompatibleEmbeddingProvider:
    def __init__(
        self,
        *,
        base_url: str | None = None,
        model_id: str,
        dim: int,
        api_key: str | None = None,
    ) -> None:
        self.base_url = (base_url or os.getenv("LOCAL_BASE_URL", "")).rstrip("/")
        if not self.base_url:
            raise ValueError("base_url required (or set LOCAL_BASE_URL)")
        self.model_id = model_id
        self.dim = dim
        self._api_key = api_key or os.getenv("OPENAI_API_KEY", "not-needed")

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._api_key}"}

    def embed(self, text: str) -> list[float]:
        return self.embed_many([text])[0]

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        resp = httpx.post(
            f"{self.base_url}/embeddings",
            json={"model": self.model_id, "input": texts},
            headers=self._headers(),
            timeout=60.0,
        )
        resp.raise_for_status()
        data = resp.json()["data"]
        data.sort(key=lambda d: d["index"])
        return [list(d["embedding"]) for d in data]
