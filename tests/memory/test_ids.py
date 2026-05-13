import re

from mcp_servers.memory.ids import (
    new_dream_run_id,
    new_entity_id,
    new_episode_id,
    new_fact_id,
    new_hypothesis_id,
    new_raw_turn_id,
    new_reflection_id,
)

_ULID_RE = re.compile(r"^[0-9A-HJKMNP-TV-Z]{26}$")


def _assert_prefixed_ulid(s: str, prefix: str) -> None:
    assert s.startswith(prefix + "_"), s
    _, rest = s.split("_", 1)
    assert _ULID_RE.match(rest), f"not a ULID after prefix: {rest}"


def test_each_generator_has_distinct_prefix_and_yields_ulids() -> None:
    pairs = [
        (new_raw_turn_id(), "rt"),
        (new_episode_id(), "ep"),
        (new_entity_id(), "en"),
        (new_fact_id(), "fa"),
        (new_reflection_id(), "re"),
        (new_hypothesis_id(), "hy"),
        (new_dream_run_id(), "dr"),
    ]
    for value, prefix in pairs:
        _assert_prefixed_ulid(value, prefix)


def test_consecutive_ids_are_unique() -> None:
    assert new_episode_id() != new_episode_id()
