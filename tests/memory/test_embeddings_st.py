import pytest

from mcp_servers.memory.embeddings.sentence_transformers_provider import (
    SentenceTransformersProvider,
)


@pytest.mark.slow
def test_embed_returns_768_dim_vector() -> None:
    p = SentenceTransformersProvider(model_id="nomic-ai/nomic-embed-text-v1.5")
    vec = p.embed("hello world")
    assert len(vec) == 768
    assert all(isinstance(x, float) for x in vec)


@pytest.mark.slow
def test_embed_many_matches_embed_singletons() -> None:
    p = SentenceTransformersProvider(model_id="nomic-ai/nomic-embed-text-v1.5")
    a = p.embed("foo")
    b = p.embed("bar")
    both = p.embed_many(["foo", "bar"])
    assert both[0] == pytest.approx(a, rel=1e-4)
    assert both[1] == pytest.approx(b, rel=1e-4)
