from mcp_servers.memory.embeddings.base import EmbeddingProvider


def test_protocol_has_required_attrs() -> None:
    # All implementations must expose `dim` and `model_id`, plus
    # `embed(text)` and `embed_many(texts)`.
    assert "dim" in EmbeddingProvider.__annotations__
    assert "model_id" in EmbeddingProvider.__annotations__
    assert hasattr(EmbeddingProvider, "embed")
    assert hasattr(EmbeddingProvider, "embed_many")
