"""Loader for providers.toml."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

KNOWN_PROVIDERS = {"anthropic", "openai", "lmstudio"}


class UnknownProviderError(ValueError):
    """Raised when providers.toml contains a section we don't recognize."""


@dataclass(frozen=True)
class ProviderConfig:
    name: str
    models: list[str]
    default_model: str
    default_max_tokens: int
    default_temperature: float
    capabilities: list[str]


def load_providers_config(path: str | Path = "providers.toml") -> dict[str, ProviderConfig]:
    """Load and validate providers.toml. Returns {provider_name: ProviderConfig}."""
    p = Path(path)
    with p.open("rb") as f:
        data = tomllib.load(f)
    out: dict[str, ProviderConfig] = {}
    for name, section in data.items():
        if name not in KNOWN_PROVIDERS:
            raise UnknownProviderError(
                f"providers.toml has unknown provider {name!r}; "
                f"expected one of {sorted(KNOWN_PROVIDERS)}"
            )
        out[name] = ProviderConfig(
            name=name,
            models=list(section.get("models", [])),
            default_model=section["default_model"],
            default_max_tokens=int(section["default_max_tokens"]),
            default_temperature=float(section["default_temperature"]),
            capabilities=list(section.get("capabilities", [])),
        )
    return out
