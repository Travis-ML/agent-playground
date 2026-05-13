import respx
from httpx import Response

from mcp_servers.memory.embeddings.openai_compatible_provider import (
    OpenAICompatibleEmbeddingProvider,
)


@respx.mock
def test_embed_calls_v1_embeddings_endpoint() -> None:
    respx.post("http://localhost:8000/v1/embeddings").mock(
        return_value=Response(
            200,
            json={
                "data": [{"embedding": [0.1] * 768, "index": 0}],
                "model": "BAAI/bge-base-en-v1.5",
            },
        )
    )
    p = OpenAICompatibleEmbeddingProvider(
        base_url="http://localhost:8000/v1",
        model_id="BAAI/bge-base-en-v1.5",
        dim=768,
    )
    vec = p.embed("hello")
    assert vec == [0.1] * 768


@respx.mock
def test_embed_many_batches_inputs() -> None:
    respx.post("http://localhost:8000/v1/embeddings").mock(
        return_value=Response(
            200,
            json={
                "data": [
                    {"embedding": [0.1] * 768, "index": 0},
                    {"embedding": [0.2] * 768, "index": 1},
                ],
                "model": "BAAI/bge-base-en-v1.5",
            },
        )
    )
    p = OpenAICompatibleEmbeddingProvider(
        base_url="http://localhost:8000/v1",
        model_id="BAAI/bge-base-en-v1.5",
        dim=768,
    )
    out = p.embed_many(["a", "b"])
    assert out == [[0.1] * 768, [0.2] * 768]
