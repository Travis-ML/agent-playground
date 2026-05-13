"""Dataclasses for memory subsystem rows. Storage layer maps to/from these."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RawTurnRef:
    id: str
    conversation_id: str
    turn_index: int
    role: str               # 'user' | 'assistant' | 'tool'
    occurred_at: str        # ISO-8601 Z
    recorded_at: str
    extraction_status: str  # 'pending' | 'done' | 'failed' | 'poison'
    retry_count: int = 0
    last_error: str | None = None


@dataclass(frozen=True)
class Episode:
    id: str
    actor: str
    predicate: str
    subject_entity: str | None
    object_entity: str | None
    object_value: str | None
    summary: str
    importance: float
    occurred_at: str
    created_at: str
    status: str               # 'fresh' | 'consolidated' | 'archived'
    source_refs: list[dict]   # list of {"raw_turn_ref_id": "rt_..."}


@dataclass(frozen=True)
class Entity:
    id: str
    canonical_name: str
    kind: str          # 'person'|'project'|'concept'|'tool'|'file'|'place'|'other'
    aliases: list[str]
    summary: str | None
    first_seen: str
    last_seen: str
    importance: float


@dataclass(frozen=True)
class Fact:
    id: str
    subject_entity: str
    predicate: str
    object_entity: str | None
    object_value: str | None
    valid_from: str
    valid_to: str | None
    learned_at: str
    invalidated_at: str | None
    source_episode_ids: list[str]
    confidence: float
    supersedes: str | None
    superseded_by: str | None
    created_in_dream_run: str
