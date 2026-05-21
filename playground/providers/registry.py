"""Provider registry — instantiate clients and detect availability."""

from __future__ import annotations

import os

from playground.providers.anthropic_client import AnthropicClient
from playground.providers.base import LLMClient
from playground.providers.local_client import LocalClient, discover_local_models
from playground.providers.openai_client import OpenAIClient


def get_client(provider: str, model: str, **overrides) -> LLMClient:
    if provider == "anthropic":
        return AnthropicClient(model=model, **overrides)
    if provider == "openai":
        return OpenAIClient(model=model, **overrides)
    if provider == "local":
        return LocalClient(model=model, **overrides)
    raise ValueError(f"Unknown provider: {provider!r}")


def list_available_providers(check_local: bool = True) -> list[str]:
    """Return providers that are likely usable right now."""
    out: list[str] = []
    if os.getenv("ANTHROPIC_API_KEY"):
        out.append("anthropic")
    if os.getenv("OPENAI_API_KEY"):
        out.append("openai")
    if os.getenv("LOCAL_BASE_URL"):
        if not check_local or discover_local_models(timeout=0.5):
            out.append("local")
    return out


def list_models(provider: str, static_models: list[str]) -> list[str]:
    """Return models for a provider. For local, queries /v1/models."""
    if provider == "local":
        return discover_local_models() or static_models
    return list(static_models)
