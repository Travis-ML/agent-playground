"""Triplet sampler — biased toward high graph-distance triplets to make
recombination productive. Deterministic with `seed`."""

from __future__ import annotations

import random
from collections.abc import Callable


def sample_triplets(
    *,
    candidates: list[tuple[str, str]],
    k: int,
    seed: int,
    link_lookup: Callable[[tuple[str, str]], list[tuple[str, str]]],
    bias_distant: bool = True,
) -> list[tuple[tuple[str, str], tuple[str, str], tuple[str, str]]]:
    if len(candidates) < 3:
        return []
    rng = random.Random(seed)

    def _distant_score(a, b, c) -> float:
        if not bias_distant:
            return 1.0
        # cheap proxy for graph distance: count of direct links between any pair
        directly_linked = 0
        for x, y in [(a, b), (b, c), (a, c)]:
            x_neighbors = set(link_lookup(x))
            if y in x_neighbors:
                directly_linked += 1
        # prefer triplets where no pair is directly linked
        return 1.0 / (1 + directly_linked)

    pool = list(candidates)
    out: list = []
    seen: set[tuple] = set()
    attempts = 0
    while len(out) < k and attempts < 20 * k:
        attempts += 1
        a, b, c = rng.sample(pool, 3)
        key = tuple(sorted([a, b, c]))
        if key in seen:
            continue
        # weighted accept
        score = _distant_score(a, b, c)
        if rng.random() < score:
            out.append((a, b, c))
            seen.add(key)
    return out
