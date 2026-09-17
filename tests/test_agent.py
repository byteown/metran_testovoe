import json

import pytest

from fin_agent.agent.llm import LLMError
from fin_agent.agent.loop import run_agent
from fin_agent.agent.registry import ToolContext
from fin_agent.schemas import Status


def _tool_call_response(name: str, arguments: dict) -> dict:
    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": name,
                                "arguments": json.dumps(arguments, ensure_ascii=False),
                            },
                        }
                    ],
                }
            }
        ],
        "usage": {"prompt_tokens": 100, "completion_tokens": 20},
    }


def _text_response(text: str) -> dict:
    return {
        "choices": [{"message": {"role": "assistant", "content": text}}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 20},
    }


class FakeLLM:
    def __init__(self, responses: list[dict], fail: Exception | None = None) -> None:
        self._responses = list(responses)
        self._fail = fail
        self.calls = 0

    def chat(
            self,
            messages: list[dict],
            tools: list[dict] | None = None,
            tool_choice: str | None = None,
    ) -> dict:
        self.calls += 1
        if self._fail is not None:
            raise self._fail
        if self._responses:
            return self._responses.pop(0)
        return _text_response("Готово.")

    @staticmethod
    def usage(response: dict) -> tuple[int, int]:
        usage = response.get("usage") or {}
        return usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)


@pytest.fixture
def ctx(conn, index) -> ToolContext:
    return ToolContext(conn=conn, index=index)


def test_numbers_come_from_tool_not_from_model(ctx):
    client = FakeLLM(
        [
            _tool_call_response("get_receivables", {"counterparty": "ООО Ромашка"}),
            _text_response("Задолженность составляет 999 рублей."),
        ]
    )

    result = run_agent("Сколько должна Ромашка?", ctx, client)

    assert result.answer.status == Status.OK
    assert result.answer.calculations[0].result == pytest.approx(2600080.15)
    assert sorted(result.answer.sources[0].record_ids) == ["СБ-00010", "СБ-00026"]
    assert result.trace.tool_calls[0]["name"] == "get_receivables"


def test_llm_failure_gives_honest_error(ctx):
    client = FakeLLM([], fail=LLMError("соединение отклонено"))

    result = run_agent("Сколько должна Ромашка?", ctx, client)

    assert result.answer.status == Status.ERROR
    assert result.answer.calculations == []
    assert result.answer.sources == []
    assert result.answer.warnings


def test_turn_limit_stops_the_loop(ctx):
    endless = [
        _tool_call_response("get_receivables", {"only_overdue": True}) for _ in range(50)
    ]
    client = FakeLLM(endless)

    result = run_agent("Сумма просрочки?", ctx, client)

    assert result.answer.status == Status.PARTIAL
    assert result.trace.turns <= 10  # верхняя граница AGENT_MAX_TURNS из конфига
    assert client.calls == result.trace.turns
    assert any("лимит витков" in w for w in result.answer.warnings)


def test_unknown_tool_does_not_crash(ctx):
    client = FakeLLM(
        [
            _tool_call_response("выдуманный_инструмент", {}),
            _text_response("Не удалось выполнить."),
        ]
    )

    result = run_agent("Что-нибудь?", ctx, client)

    assert result.answer.status == Status.NOT_FOUND
    assert any("не существует" in w for w in result.answer.warnings)
