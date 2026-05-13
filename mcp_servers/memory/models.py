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
