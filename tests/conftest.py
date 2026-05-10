"""Shared pytest fixtures for the playground."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest


@pytest.fixture
def tmp_conversations_root(tmp_path: Path) -> Path:
    """A fresh conversations/ root per test."""
    root = tmp_path / "conversations"
    root.mkdir()
    return root


@pytest.fixture
def fixtures_dir() -> Path:
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def canonical_conversation(fixtures_dir: Path) -> dict:
    """A v1-schema conversation used for round-trip tests."""
    with (fixtures_dir / "conversation_v1.json").open() as f:
        return json.load(f)
