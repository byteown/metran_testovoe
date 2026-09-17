import json
import logging
import time
from dataclasses import dataclass, field

from fin_agent.agent.llm import LLMClient, LLMError
from fin_agent.agent.registry import TOOLS_SPEC, ToolContext, call_tool
from fin_agent.config import get_settings
from fin_agent.schemas import Answer, Status
from fin_agent.tools.base import ToolResult

logger = logging.getLogger(__name__)

MAX_ROWS_TO_MODEL = 20

SYSTEM_PROMPT = """Ты — помощник финансовой службы ООО «ПромИнвест».
Отвечаешь на вопросы по учётным данным из 1С и по внутренним регламентам.

Отчётная дата — {report_date}. Все сроки и просрочки считаются на эту дату,
а не на сегодняшний день.

Правила:
1. Никогда не считай в уме. Любое число в ответе должно прийти из инструмента.
   Не складывай, не вычитай и не округляй значения самостоятельно.
2. Если вопрос про суммы, задолженность, счета или обороты — используй
   инструменты по данным. Если про порядок действий, правила и нормы —
   search_regulations.
3. Название контрагента, названное неточно, сначала уточни через find_counterparty.
4. Разделы регламента разбиты по диапазонам дней просрочки. Выбирай тот раздел,
   чей диапазон ПОКРЫВАЕТ число из вопроса: например, 45 дней попадают
   в диапазон «от 30 до 60 дней».
5. Если инструмент вернул пустой результат — так и скажи: данных нет.
   Не придумывай числа и не подставляй ноль вместо отсутствующих данных.
   Прежде чем заявить, что данных нет, обязательно проверь это инструментом:
   утверждать отсутствие без проверки нельзя.
6. Отвечай по-русски, коротко и по существу: одно-три предложения.
   Не перечисляй в тексте то, что уже есть в полях calculations и sources.
"""


@dataclass
class AgentTrace:
    question: str
    turns: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    duration_s: float = 0.0
    status: str = ""
    tool_calls: list[dict] = field(default_factory=list)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def as_dict(self) -> dict:
        return {
            "question": self.question,
            "turns": self.turns,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "duration_s": round(self.duration_s, 2),
            "status": self.status,
            "tool_calls": self.tool_calls,
        }


@dataclass
class AgentResult:
    answer: Answer
    trace: AgentTrace


def _tool_payload(result: ToolResult) -> str:
    payload = {
        "rows": result.rows[:MAX_ROWS_TO_MODEL],
        "calculations": [
            {"operation": c.operation, "result": c.result, "currency": c.currency}
            for c in result.calculations
        ],
        "warnings": result.warnings,
    }
    if len(result.rows) > MAX_ROWS_TO_MODEL:
        payload["note"] = f"показаны первые {MAX_ROWS_TO_MODEL} из {len(result.rows)} строк"
    return json.dumps(payload, ensure_ascii=False, default=str)


def _decide_status(tool_results: list[ToolResult], exhausted: bool) -> Status:
    if not tool_results:
        return Status.NOT_FOUND
    if exhausted:
        return Status.PARTIAL
    if any(r.rows for r in tool_results):
        return Status.OK
    return Status.NOT_FOUND


def run_agent(question: str, ctx: ToolContext, client: LLMClient) -> AgentResult:
    settings = get_settings()
    trace = AgentTrace(question=question)
    started = time.perf_counter()

    messages: list[dict] = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT.format(report_date=settings.data.report_date),
        },
        {"role": "user", "content": question},
    ]

    tool_results: list[ToolResult] = []
    final_text = ""
    exhausted = False

    try:
        for turn in range(settings.agent.max_turns):
            response = client.chat(messages, tools=TOOLS_SPEC)
            trace.turns = turn + 1
            prompt_tokens, completion_tokens = client.usage(response)
            trace.prompt_tokens += prompt_tokens
            trace.completion_tokens += completion_tokens

            message = response["choices"][0]["message"]
            tool_calls = message.get("tool_calls") or []

            if not tool_calls:
                final_text = (message.get("content") or "").strip()
                break

            messages.append(message)
            for call in tool_calls:
                result = _execute(call, ctx, trace)
                tool_results.append(result)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.get("id", ""),
                        "content": _tool_payload(result),
                    }
                )

            if trace.total_tokens >= settings.agent.token_budget:
                logger.warning(
                    "Бюджет токенов исчерпан: %d >= %d",
                    trace.total_tokens,
                    settings.agent.token_budget,
                )
                exhausted = True
                break
        else:
            exhausted = True

    except LLMError as exc:
        logger.error("Модель недоступна: %s", exc)
        trace.duration_s = time.perf_counter() - started
        trace.status = Status.ERROR
        logger.info("trace %s", json.dumps(trace.as_dict(), ensure_ascii=False))
        return AgentResult(
            answer=Answer(
                answer="Не удалось получить ответ: языковая модель недоступна",
                status=Status.ERROR,
                warnings=[str(exc)],
            ),
            trace=trace,
        )

    if not final_text:
        final_text = _summarize(tool_results, exhausted)

    status = _decide_status(tool_results, exhausted)
    answer = Answer(
        answer=final_text,
        status=status,
        calculations=_collect(tool_results, "calculations"),
        sources=_collect(tool_results, "sources"),
        warnings=_collect(tool_results, "warnings"),
    )
    if exhausted:
        answer.warnings.append(
            f"Достигнут лимит витков ({settings.agent.max_turns}) "
            f"или бюджет токенов ({settings.agent.token_budget}); ответ может быть неполным"
        )

    trace.duration_s = time.perf_counter() - started
    trace.status = status
    logger.info("trace %s", json.dumps(trace.as_dict(), ensure_ascii=False))
    return AgentResult(answer=answer, trace=trace)


def _execute(call: dict, ctx: ToolContext, trace: AgentTrace) -> ToolResult:
    function = call.get("function") or {}
    name = function.get("name", "")
    raw_arguments = function.get("arguments") or "{}"

    try:
        arguments = json.loads(raw_arguments) if isinstance(raw_arguments, str) else raw_arguments
    except json.JSONDecodeError:
        logger.warning("Модель прислала неразбираемые аргументы для %s: %r", name, raw_arguments)
        trace.tool_calls.append({"name": name, "error": "invalid_json"})
        return ToolResult(rows=[], warnings=[f"Аргументы для {name} не разобрались как JSON"])

    started = time.perf_counter()
    result = call_tool(name, arguments, ctx)
    trace.tool_calls.append(
        {
            "name": name,
            "arguments": arguments,
            "rows": len(result.rows),
            "duration_ms": round((time.perf_counter() - started) * 1000, 1),
            "warnings": result.warnings,
        }
    )
    return result


def _collect(results: list[ToolResult], attribute: str) -> list:
    collected: list = []
    for result in results:
        collected.extend(getattr(result, attribute))
    return collected


def _summarize(results: list[ToolResult], exhausted: bool) -> str:
    if exhausted:
        return "Не удалось завершить рассуждение в отведённых пределах; ниже то, что успели посчитать"
    if not results or not any(r.rows for r in results):
        return "По этому вопросу данных не нашлось"
    return "Результаты расчёта приведены в полях calculations и sources"
