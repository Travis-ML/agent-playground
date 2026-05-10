"""Loader for the on-disk prompt library."""

from __future__ import annotations

from pathlib import Path

DEFAULT_LIBRARY = Path(__file__).parent / "library"


class PromptNotFoundError(KeyError):
    """Raised when load_prompt() can't find the named prompt."""


def list_prompts(library_dir: str | Path = DEFAULT_LIBRARY) -> list[str]:
    p = Path(library_dir)
    if not p.exists():
        return []
    return sorted(f.stem for f in p.glob("*.md"))


def load_prompt(name: str, library_dir: str | Path = DEFAULT_LIBRARY) -> str:
    p = Path(library_dir) / f"{name}.md"
    if not p.exists():
        raise PromptNotFoundError(name)
    return p.read_text().strip()
