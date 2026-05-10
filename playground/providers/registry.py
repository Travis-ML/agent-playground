"""Provider registry — instantiate clients and detect availability."""

from __future__ import annotations

import os

from playground.providers.anthropic_client import AnthropicClient
from playground.providers.base import LLMClient
from playground.providers.lmstudio_client import LMStudioClient, discover_lmstudio_models
from playground.providers.openai_client import OpenAIClient


def get_client(provider: str, model: str, **overrides) -> LLMClient:
    if provider == "anthropic":
        return AnthropicClient(model=model, **overrides)
    if provider == "openai":
        return OpenAIClient(model=model, **overrides)
    if provider == "lmstudio":
        return LMStudioClient(model=model, **overrides)
    raise ValueError(f"Unknown provider: {provider!r}")


def list_available_providers(check_lmstudio: bool = True) -> list[str]:
    """Return providers that are likely usable right now."""
    out: list[str] = []
    if os.getenv("ANTHROPIC_API_KEY"):
        out.append("anthropic")
    if os.getenv("OPENAI_API_KEY"):
        out.append("openai")
    if os.getenv("LMSTUDIO_BASE_URL"):
        if not check_lmstudio or discover_lmstudio_models(timeout=0.5):
            out.append("lmstudio")
    return out


def list_models(provider: str, static_models: list[str]) -> list[str]:
    """Return models for a provider. For lmstudio, queries /v1/models."""
    if provider == "lmstudio":
        return discover_lmstudio_models() or static_models
    return list(static_models)
