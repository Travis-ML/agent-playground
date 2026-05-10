"""Tests for the providers.toml loader."""

from pathlib import Path

import pytest

from playground.providers.config import (
    ProviderConfig,
    UnknownProviderError,
    load_providers_config,
)


def test_load_known_anthropic_config(tmp_path: Path) -> None:
    cfg_path = tmp_path / "providers.toml"
    cfg_path.write_text(
        """
        [anthropic]
        models = ["claude-sonnet-4-6"]
        default_model = "claude-sonnet-4-6"
        default_max_tokens = 4096
        default_temperature = 1.0
        capabilities = ["tools", "streaming"]
        """
    )
    cfg = load_providers_config(cfg_path)
    assert "anthropic" in cfg
    anthropic = cfg["anthropic"]
    assert isinstance(anthropic, ProviderConfig)
    assert anthropic.default_model == "claude-sonnet-4-6"
    assert anthropic.default_max_tokens == 4096
    assert "streaming" in anthropic.capabilities


def test_unknown_provider_raises(tmp_path: Path) -> None:
    cfg_path = tmp_path / "providers.toml"
    cfg_path.write_text(
        """
        [made_up_provider]
        models = ["x"]
        default_model = "x"
        default_max_tokens = 1
        default_temperature = 1.0
        capabilities = []
        """
    )
    with pytest.raises(UnknownProviderError):
        load_providers_config(cfg_path)
