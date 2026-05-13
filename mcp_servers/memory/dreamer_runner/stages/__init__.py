"""Stage registry. Each stage is filled in by its own phase (8-13)."""

from __future__ import annotations


def all_stages() -> dict:
    # Real implementations get registered here as they land in Phases 8-13.
    # Until then the runner is callable with mock stages (tests inject them).
    from mcp_servers.memory.dreamer_runner.stages.stage_1_cluster import run as ingest_cluster
    from mcp_servers.memory.dreamer_runner.stages.stage_2_consolidate import run as consolidate
    from mcp_servers.memory.dreamer_runner.stages.stage_3_extract import run as extract
    from mcp_servers.memory.dreamer_runner.stages.stage_4_reflect import run as reflect
    from mcp_servers.memory.dreamer_runner.stages.stage_5_recombine import run as recombine
    from mcp_servers.memory.dreamer_runner.stages.stage_6_decay_reindex import run as decay_reindex

    return {
        "ingest_cluster": ingest_cluster,
        "consolidate":    consolidate,
        "extract":        extract,
        "reflect":        reflect,
        "recombine":      recombine,
        "decay_reindex":  decay_reindex,
    }
