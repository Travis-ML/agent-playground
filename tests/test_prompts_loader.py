"""Tests for the prompt library loader."""

from pathlib import Path

import pytest

from playground.prompts.loader import (
    PromptNotFoundError,
    list_prompts,
    load_prompt,
)


def test_list_prompts_includes_default(tmp_path: Path) -> None:
    lib = tmp_path / "library"
    lib.mkdir()
    (lib / "default.md").write_text("hello")
    (lib / "other.md").write_text("world")
    names = list_prompts(library_dir=lib)
    assert sorted(names) == ["default", "other"]


def test_load_prompt_returns_text(tmp_path: Path) -> None:
    lib = tmp_path / "library"
    lib.mkdir()
    (lib / "default.md").write_text("system text\n")
    text = load_prompt("default", library_dir=lib)
    assert text == "system text"   # trimmed


def test_unknown_prompt_raises(tmp_path: Path) -> None:
    lib = tmp_path / "library"
    lib.mkdir()
    with pytest.raises(PromptNotFoundError):
        load_prompt("nope", library_dir=lib)
