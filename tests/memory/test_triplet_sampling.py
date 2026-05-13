from mcp_servers.memory.dreamer_runner.triplet_sampling import sample_triplets


def test_sample_triplets_deterministic_with_seed() -> None:
    candidates = [("episode", f"ep_{i}") for i in range(10)]
    a = sample_triplets(candidates=candidates, k=4, seed=42, link_lookup=lambda *_: [])
    b = sample_triplets(candidates=candidates, k=4, seed=42, link_lookup=lambda *_: [])
    assert a == b


def test_sample_triplets_returns_k_distinct_triplets() -> None:
    candidates = [("episode", f"ep_{i}") for i in range(12)]
    out = sample_triplets(candidates=candidates, k=6, seed=1, link_lookup=lambda *_: [])
    assert len(out) == 6
    for tr in out:
        assert len(set(tr)) == 3  # three distinct nodes per triplet


def test_sample_triplets_handles_small_pool() -> None:
    candidates = [("episode", "ep_1"), ("episode", "ep_2")]
    out = sample_triplets(candidates=candidates, k=5, seed=1, link_lookup=lambda *_: [])
    assert out == []
