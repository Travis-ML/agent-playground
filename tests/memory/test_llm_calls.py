import json
from unittest.mock import MagicMock

from mcp_servers.memory.dreamer_runner.llm_calls import call_json_llm
from mcp_servers.memory.providers.base import MessageComplete, TextDelta, Usage


def _stream(text):
    yield TextDelta(text=text)
    yield MessageComplete(usage=Usage(input_tokens=1, output_tokens=1),
                          stop_reason="end_turn")


def test_call_json_llm_parses_response() -> None:
    llm = MagicMock()
    llm.stream_chat.return_value = _stream(json.dumps({"groups": []}))
    out = call_json_llm(llm=llm, system="you are x", user="prompt body",
                        max_tokens=500)
    assert out == {"groups": []}


def test_call_json_llm_strips_code_fences() -> None:
    llm = MagicMock()
    llm.stream_chat.return_value = _stream("```json\n" + json.dumps({"k": 1}) + "\n```")
    out = call_json_llm(llm=llm, system="x", user="y", max_tokens=100)
    assert out == {"k": 1}
