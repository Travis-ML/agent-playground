"""Current-time example tool."""

from __future__ import annotations

from datetime import UTC, datetime

from playground.tools import register_tool


@register_tool
def get_current_time(timezone_name: str = "UTC") -> str:
    """Return the current time as an ISO 8601 string. Timezone is informational only."""
    return datetime.now(UTC).isoformat() + f" ({timezone_name})"
