import logging
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

from fin_agent.rag.index import DocumentIndex
from fin_agent.tools import finance, rag
from fin_agent.tools.base import ToolResult

logger = logging.getLogger(__name__)

ISO_DATE = r"^\d{4}-\d{2}-\d{2}$"


@dataclass(frozen=True)
class ToolContext:
    conn: sqlite3.Connection
    index: DocumentIndex


class FindCounterpartyArgs(BaseModel):
    name: str = Field(
        description="Название контрагента так, как его назвал пользователь, "
                    "в именительном падеже: «Ромашка», «ТехноСнаб», «Смирнов»."
    )


class GetReceivablesArgs(BaseModel):
    counterparty: str | None = Field(
        default=None,
        description="Точное название контрагента из справочника, например «ООО Ромашка». "
                    "Если пользователь назвал контрагента сокращённо или с ошибкой — "
                    "сначала вызови find_counterparty и возьми название оттуда.",
    )
    only_overdue: bool = Field(
        default=False,
        description="true — учитывать только просроченную задолженность, "
                    "false — всю дебиторскую задолженность.",
    )
    min_overdue_days: int | None = Field(
        default=None,
        ge=0,
        description="Минимальное число дней просрочки. Например, 90 — долги старше 90 дней.",
    )


class TopDebtorsArgs(BaseModel):
    limit: int = Field(default=5, ge=1, le=50, description="Сколько должников вернуть.")
    by: Literal["overdue", "total"] = Field(
        default="overdue",
        description="overdue — ранжировать по сумме просрочки, "
                    "total — по всей задолженности контрагента.",
    )


class SearchInvoicesArgs(BaseModel):
    direction: Literal["issued", "received"] = Field(
        description="issued — счета, выставленные покупателям (нам должны); "
                    "received — счета от поставщиков (должны мы)."
    )
    status: Literal["paid", "pending", "overdue"] | None = Field(
        default=None,
        description="paid — оплачен, pending — срок не наступил, overdue — просрочен.",
    )
    due_from: str | None = Field(
        default=None,
        pattern=ISO_DATE,
        description="Нижняя граница срока оплаты, формат ГГГГ-ММ-ДД.",
    )
    due_to: str | None = Field(
        default=None,
        pattern=ISO_DATE,
        description="Верхняя граница срока оплаты, формат ГГГГ-ММ-ДД.",
    )
    counterparty: str | None = Field(
        default=None, description="Точное название контрагента из справочника."
    )


class AccountTurnoverArgs(BaseModel):
    account: Literal["62", "62.01", "60", "60.01"] = Field(
        description="62.01 — расчёты с покупателями, 60.01 — расчёты с поставщиками."
    )
    date_from: str = Field(pattern=ISO_DATE, description="Начало периода, ГГГГ-ММ-ДД.")
    date_to: str = Field(pattern=ISO_DATE, description="Конец периода, ГГГГ-ММ-ДД.")


class SearchRegulationsArgs(BaseModel):
    question: str = Field(
        description="Вопрос о порядке действий или правилах учёта, "
                    "сформулированный своими словами."
    )


def _find_counterparty(ctx: ToolContext, args: FindCounterpartyArgs) -> ToolResult:
    return finance.find_counterparty(ctx.conn, args.name)


def _get_receivables(ctx: ToolContext, args: GetReceivablesArgs) -> ToolResult:
    return finance.get_receivables(
        ctx.conn,
        counterparty=args.counterparty,
        only_overdue=args.only_overdue,
        min_overdue_days=args.min_overdue_days,
    )


def _top_debtors(ctx: ToolContext, args: TopDebtorsArgs) -> ToolResult:
    return finance.top_debtors(ctx.conn, limit=args.limit, by=args.by)


def _search_invoices(ctx: ToolContext, args: SearchInvoicesArgs) -> ToolResult:
    return finance.search_invoices(
        ctx.conn,
        direction=args.direction,
        status=args.status,
        due_from=args.due_from,
        due_to=args.due_to,
        counterparty=args.counterparty,
    )


def _account_turnover(ctx: ToolContext, args: AccountTurnoverArgs) -> ToolResult:
    return finance.account_turnover(
        ctx.conn, account=args.account, date_from=args.date_from, date_to=args.date_to
    )


def _search_regulations(ctx: ToolContext, args: SearchRegulationsArgs) -> ToolResult:
    return rag.search_regulations(ctx.index, args.question)


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    args_model: type[BaseModel]
    handler: Callable[[ToolContext, BaseModel], ToolResult]


_SPECS: tuple[ToolSpec, ...] = (
    ToolSpec(
        name="find_counterparty",
        description=(
            "Находит контрагента в справочнике по неточному названию и возвращает "
            "его официальное наименование и ИНН. Вызывай ПЕРВЫМ, если пользователь "
            "назвал контрагента сокращённо, в другом падеже или с опечаткой. "
            "Если найдено несколько — переспроси пользователя, не выбирай наугад."
        ),
        args_model=FindCounterpartyArgs,
        handler=_find_counterparty,
    ),
    ToolSpec(
        name="get_receivables",
        description=(
            "Дебиторская задолженность покупателей по учётным данным: суммы долга, "
            "дни просрочки, номера счетов. Используй для вопросов «сколько должен X», "
            "«сумма просроченной дебиторки», «долги старше N дней»."
        ),
        args_model=GetReceivablesArgs,
        handler=_get_receivables,
    ),
    ToolSpec(
        name="top_debtors",
        description=(
            "Рейтинг контрагентов по сумме задолженности: топ-N должников. "
            "Используй, когда нужен список крупнейших должников, а не долг "
            "конкретного контрагента."
        ),
        args_model=TopDebtorsArgs,
        handler=_top_debtors,
    ),
    ToolSpec(
        name="search_invoices",
        description=(
            "Счета покупателям и от поставщиков с фильтром по сроку оплаты, статусу "
            "и контрагенту. Используй для вопросов о платёжном календаре: «сколько мы "
            "должны поставщикам со сроком в августе», «какие счета просрочены». "
            "Возвращает и сумму счетов, и остаток к оплате."
        ),
        args_model=SearchInvoicesArgs,
        handler=_search_invoices,
    ),
    ToolSpec(
        name="account_turnover",
        description=(
            "Обороты по счетам бухгалтерского учёта за период, посчитанные из первичных "
            "документов: 62.01 (расчёты с покупателями) и 60.01 (расчёты с поставщиками). "
            "Возвращает обороты по дебету и по кредиту. Данные есть только "
            "за 2026-01-01..2026-07-31."
        ),
        args_model=AccountTurnoverArgs,
        handler=_account_turnover,
    ),
    ToolSpec(
        name="search_regulations",
        description=(
            "Поиск по внутренним регламентам финансовой службы: регламент работы "
            "с дебиторской задолженностью и учётная политика. Отвечает на вопросы "
            "«что делать при просрочке N дней», «как формируется резерв», «каков порядок». "
            "НЕ содержит сумм и НЕ считает по учётным данным: если вопрос про конкретные "
            "цифры или конкретного контрагента — используй инструменты по данным. "
            "Возвращает разделы регламентов с дословными цитатами."
        ),
        args_model=SearchRegulationsArgs,
        handler=_search_regulations,
    ),
)

REGISTRY: dict[str, ToolSpec] = {spec.name: spec for spec in _SPECS}

TOOLS_SPEC: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": spec.name,
            "description": spec.description,
            "parameters": spec.args_model.model_json_schema(),
        },
    }
    for spec in _SPECS
]


def call_tool(name: str, arguments: dict, ctx: ToolContext) -> ToolResult:
    spec = REGISTRY.get(name)
    if spec is None:
        logger.warning("Модель запросила неизвестный инструмент %r", name)
        return ToolResult(
            rows=[],
            warnings=[
                (
                    f"Инструмента {name!r} не существует. "
                    f"Доступны: {', '.join(sorted(REGISTRY))}"
                )
            ],
        )

    try:
        args = spec.args_model(**arguments)
    except ValidationError as exc:
        details = "; ".join(
            f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()[:3]
        )
        logger.warning("Неверные аргументы для %s: %s", name, details)
        return ToolResult(rows=[], warnings=[f"Неверные аргументы для {name}: {details}"])
    except TypeError as exc:
        logger.warning("Аргументы для %s не являются объектом: %s", name, exc)
        return ToolResult(rows=[], warnings=[f"Аргументы для {name} должны быть объектом"])

    try:
        return spec.handler(ctx, args)
    except Exception:
        logger.exception("Инструмент %s завершился с ошибкой", name)
        return ToolResult(rows=[], warnings=[f"Инструмент {name} завершился с ошибкой"])
