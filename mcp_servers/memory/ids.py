"""ULID-based id generators with kind prefixes."""

from __future__ import annotations

from ulid import ULID


def _new(prefix: str) -> str:
    return f"{prefix}_{ULID()}"


def new_raw_turn_id() -> str:
    return _new("rt")


def new_episode_id() -> str:
    return _new("ep")


def new_entity_id() -> str:
    return _new("en")


def new_fact_id() -> str:
    return _new("fa")


def new_reflection_id() -> str:
    return _new("re")


def new_hypothesis_id() -> str:
    return _new("hy")


def new_dream_run_id() -> str:
    return _new("dr")
