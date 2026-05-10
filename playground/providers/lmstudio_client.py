"""LM Studio provider — OpenAI-compatible local endpoint."""

from __future__ import annotations

import os

import httpx

from playground.providers.openai_client import OpenAIClient


class LMStudioClient(OpenAIClient):
    name = "lmstudio"

    def __init__(self, model: str, base_url: str | None = None) -> None:
        url = base_url or os.getenv("LMSTUDIO_BASE_URL", "http://localhost:1234/v1")
        # api_key required by SDK but ignored by LM Studio — placeholder is fine.
        super().__init__(model=model, api_key="lm-studio", base_url=url)


def discover_lmstudio_models(base_url: str | None = None, timeout: float = 1.0) -> list[str]:
    """Hit /v1/models to discover what's loaded. Returns [] if unreachable."""
    url = base_url or os.getenv("LMSTUDIO_BASE_URL", "http://localhost:1234/v1")
    try:
        resp = httpx.get(f"{url.rstrip('/')}/models", timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        return [m["id"] for m in data.get("data", [])]
    except Exception:
        return []
