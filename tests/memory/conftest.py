"""Shared fixtures for memory subsystem tests."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from mcp_servers.memory.db.connection import open_connection
from mcp_servers.memory.db.migrations import apply_migrations


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "memory.db"


@pytest.fixture
def conn(db_path: Path) -> Iterator[sqlite3.Connection]:
    c = open_connection(db_path)
    apply_migrations(c)
    yield c
    c.close()


@pytest.fixture
def fake_llm() -> MagicMock:
    """A MagicMock standing in for an `LLMClient` instance.

    Tests configure `.stream_chat.return_value = [TextDelta(...), MessageComplete(...)]`
    or override per-call.
    """
    return MagicMock()


@pytest.fixture
def fixed_embedder() -> Any:
    """Returns a deterministic 768-dim embedding for any input string."""

    class _Embedder:
        dim = 768

        def embed(self, text: str) -> list[float]:
            # Cheap, deterministic, distinct per input — not semantic.
            h = abs(hash(text))
            return [((h >> (i % 30)) & 1) - 0.5 for i in range(self.dim)]

        def embed_many(self, texts: list[str]) -> list[list[float]]:
            return [self.embed(t) for t in texts]

    return _Embedder()
