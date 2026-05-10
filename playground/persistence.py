"""Conversation persistence — file-per-conversation JSON, atomic writes."""

from __future__ import annotations

import copy
import json
import os
import re
import secrets
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

SCHEMA_VERSION = 1
_TS_FILENAME_FMT = "%Y-%m-%dT%H-%M-%S"


@dataclass(frozen=True)
class ConversationSummary:
    id: str
    page: str
    started_at: datetime
    ended_at: datetime | None
    provider: str
    model: str
    message_count: int
    first_user_message: str


class Conversation:
    """An open, in-memory conversation that auto-saves on append."""

    def __init__(self, path: Path, data: dict) -> None:
        self.path = path
        self.data = data
        self._save()

    @property
    def id(self) -> str:
        return self.data["id"]

    def append_message(self, msg: dict) -> None:
        self.data["messages"].append(msg)
        self._save()

    def add_event(self, event: dict) -> None:
        self.data.setdefault("events", []).append(event)
        self._save()

    def end(self) -> None:
        self.data["ended_at"] = _now_iso()
        self._save()

    def _save(self) -> None:
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.data, indent=2, ensure_ascii=False))
        try:
            os.replace(tmp, self.path)
        except Exception:
            # Clean up the tmp file so we don't leave partial state behind.
            try:
                tmp.unlink()
            except OSError:
                pass
            raise


class ConversationStore:
    def __init__(self, root: str | Path = "conversations") -> None:
        self.root = Path(root)

    def new(self, page: str, config: dict) -> Conversation:
        page_dir = self.root / page
        page_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(UTC)
        short_id = secrets.token_hex(2)
        conv_id = f"{ts.strftime(_TS_FILENAME_FMT)}-{short_id}"
        data = {
            "schema_version": SCHEMA_VERSION,
            "id": conv_id,
            "page": page,
            "started_at": ts.replace(microsecond=0).isoformat().replace("+00:00", "Z"),
            "ended_at": None,
            "config": copy.deepcopy(config),
            "messages": [],
            "events": [],
        }
        return Conversation(page_dir / f"{conv_id}.json", data)

    def list(self, page: str | None = None) -> list[ConversationSummary]:
        roots: list[Path] = []
        if page is None:
            if not self.root.exists():
                return []
            roots = [p for p in self.root.iterdir() if p.is_dir()]
        else:
            page_dir = self.root / page
            if page_dir.exists():
                roots = [page_dir]

        out: list[ConversationSummary] = []
        for page_dir in roots:
            for jfile in page_dir.glob("*.json"):
                try:
                    data = json.loads(jfile.read_text())
                except Exception:
                    continue
                first_user = ""
                for m in data.get("messages", []):
                    if m.get("role") == "user":
                        for b in m.get("content", []):
                            if b.get("type") == "text":
                                first_user = b.get("text", "")[:80]
                                break
                        if first_user:
                            break
                started = _parse_iso(data["started_at"])
                ended = _parse_iso(data["ended_at"]) if data.get("ended_at") else None
                out.append(
                    ConversationSummary(
                        id=data["id"],
                        page=data["page"],
                        started_at=started,
                        ended_at=ended,
                        provider=data["config"]["provider"],
                        model=data["config"]["model"],
                        message_count=len(data.get("messages", [])),
                        first_user_message=first_user,
                    )
                )
        out.sort(key=lambda s: s.started_at, reverse=True)
        return out

    def load(self, conv_id: str) -> Conversation:
        for page_dir in self.root.iterdir() if self.root.exists() else []:
            candidate = page_dir / f"{conv_id}.json"
            if candidate.exists():
                return Conversation(candidate, json.loads(candidate.read_text()))
        raise FileNotFoundError(f"No conversation with id {conv_id!r}")


def _now_iso() -> str:
    return (
        datetime.now(UTC)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


_ISO_RE = re.compile(r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})(Z|[+-]\d{2}:?\d{2})?$")


def _parse_iso(s: str) -> datetime:
    m = _ISO_RE.match(s)
    if not m:
        raise ValueError(f"Bad ISO timestamp: {s!r}")
    base = datetime.fromisoformat(m.group(1))
    return base.replace(tzinfo=UTC)
