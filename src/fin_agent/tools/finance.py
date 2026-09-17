import difflib
import re
import sqlite3

from fin_agent.schemas import Calculation, DataSource
from fin_agent.tools.base import ToolResult

ROW_LIMIT = 100
MAX_TERMS = 10
LEGAL_FORMS = {"ооо", "оао", "зао", "пао", "ао", "ип"}
ACCOUNT_ALIASES = {"62": "62.01", "62.01": "62.01", "60": "60.01", "60.01": "60.01"}
DATA_FROM, DATA_TO = "2026-01-01", "2026-07-31"


def _normalize(name: str) -> str:
    text = name.lower().replace("ё", "е")
    text = re.sub(r"[«»\"'.,]", " ", text)
    return " ".join(w for w in text.split() if w not in LEGAL_FORMS)


def _expression(amounts: list[float]) -> str:
    shown = " + ".join(f"{a:.2f}" for a in amounts[:MAX_TERMS])
    if len(amounts) > MAX_TERMS:
        shown += f" + ... (еще {len(amounts) - MAX_TERMS} позиций)"
    return shown


def get_receivables(
        conn: sqlite3.Connection,
        counterparty: str | None = None,
        only_overdue: bool = False,
        min_overdue_days: int | None = None,
) -> ToolResult:
    conditions: list[str] = []
    params: list = []

    if counterparty:
        conditions.append("counterparty = ?")
        params.append(counterparty)
    if only_overdue:
        conditions.append("status = 'overdue'")
    if min_overdue_days is not None:
        conditions.append("overdue_days >= ?")
        params.append(min_overdue_days)

    where = " AND ".join(conditions) if conditions else "1=1"
    sql = f"SELECT invoice, counterparty, amount, invoice_date, overdue_days, status FROM receivables WHERE {where} ORDER BY amount DESC LIMIT ?"
    rows = [dict(r) for r in conn.execute(sql, params + [ROW_LIMIT]).fetchall()]

    if not rows:
        return ToolResult(rows=[])

    amounts = [r["amount"] for r in rows]
    total = round(sum(amounts), 2)

    result = ToolResult(
        rows=rows,
        calculations=[
            Calculation(
                operation="sum",
                expression=_expression(amounts),
                result=total,
            )
        ],
        sources=[
            DataSource(
                file="receivables.csv",
                record_ids=[r["invoice"] for r in rows],
            )
        ],
    )
    if len(rows) == ROW_LIMIT:
        result.warnings.append(f"Показаны первые {ROW_LIMIT} записей, сумма неполная")
    return result


def search_invoices(
        conn: sqlite3.Connection,
        direction: str,
        status: str | None = None,
        due_from: str | None = None,
        due_to: str | None = None,
        counterparty: str | None = None,
) -> ToolResult:
    if direction == "issued":
        table = "invoices_issued"
    elif direction == "received":
        table = "invoices_received"
    else:
        return ToolResult(rows=[], warnings=[f"Неизвестное направление {direction!r}"])

    conditions: list[str] = []
    params: list = []

    if status:
        conditions.append("status = ?")
        params.append(status)
    if due_from:
        conditions.append("due_date >= ?")
        params.append(due_from)
    if due_to:
        conditions.append("due_date <= ?")
        params.append(due_to)
    if counterparty:
        conditions.append("counterparty = ?")
        params.append(counterparty)

    where = " AND ".join(conditions) if conditions else "1=1"
    sql = f"SELECT number, date, counterparty, amount, term_days, status, paid, due_date FROM {table} WHERE {where} ORDER BY due_date LIMIT ?"
    rows = [dict(r) for r in conn.execute(sql, params + [ROW_LIMIT]).fetchall()]

    if not rows:
        return ToolResult(rows=[])

    amounts = [r["amount"] for r in rows]
    remainders = [round(r["amount"] - r["paid"], 2) for r in rows]

    total = round(sum(amounts), 2)
    unpaid = round(sum(remainders), 2)

    result = ToolResult(
        rows=rows,
        calculations=[
            Calculation(operation="sum", expression=_expression(amounts), result=total),
            Calculation(
                operation="unpaid_sum",
                expression=_expression(remainders),
                result=unpaid,
            ),
        ],
        sources=[
            DataSource(file=f"{table}.csv", record_ids=[r["number"] for r in rows])
        ],
    )
    if len(rows) == ROW_LIMIT:
        result.warnings.append(f"Показаны первые {ROW_LIMIT} записей, суммы неполные")
    return result


def top_debtors(
        conn: sqlite3.Connection, limit: int = 5, by: str = "overdue"
) -> ToolResult:
    limit = max(1, min(limit, 50))

    if by == "overdue":
        where = "status = 'overdue'"
    elif by == "total":
        where = "1=1"
    else:
        return ToolResult(rows=[], warnings=[f"Неизвестный режим by={by!r}"])

    sql = f"SELECT counterparty, SUM(amount) AS total, COUNT(*) AS invoice_count, GROUP_CONCAT(invoice) AS invoices, GROUP_CONCAT(amount) AS amounts FROM receivables WHERE {where} GROUP BY counterparty ORDER BY total DESC LIMIT ?"
    rows = [dict(r) for r in conn.execute(sql, (limit,)).fetchall()]

    if not rows:
        return ToolResult(rows=[])

    calculations: list[Calculation] = []
    record_ids: list[str] = []

    for r in rows:
        invoices = r.pop("invoices").split(",")
        amounts = [float(a) for a in r.pop("amounts").split(",")]
        r["total"] = round(r["total"], 2)

        calculations.append(
            Calculation(
                operation="sum",
                expression=f"{r['counterparty']}: {_expression(amounts)}",
                result=r["total"],
            )
        )
        record_ids.extend(invoices)

    return ToolResult(
        rows=rows,
        calculations=calculations,
        sources=[DataSource(file="receivables.csv", record_ids=record_ids)],
    )


def find_counterparty(conn: sqlite3.Connection, name: str) -> ToolResult:
    query = _normalize(name)
    if not query:
        return ToolResult(rows=[], warnings=["Пустое название контрагента"])

    all_rows = [
        dict(r)
        for r in conn.execute(
            "SELECT name, inn, type, region, manager FROM counterparties"
        ).fetchall()
    ]
    index = {_normalize(r["name"]): r for r in all_rows}

    if query in index:
        matches = [index[query]]
    else:
        matches = [r for key, r in index.items() if query in key or key in query]

    if not matches:
        close = difflib.get_close_matches(query, list(index), n=3, cutoff=0.6)
        matches = [index[k] for k in close]

    if not matches:
        return ToolResult(rows=[])

    result = ToolResult(
        rows=matches,
        sources=[
            DataSource(
                file="counterparties.csv", record_ids=[r["inn"] for r in matches]
            )
        ],
    )
    if len(matches) > 1:
        result.warnings.append("Найдено несколько контрагентов, уточните название")
    return result


def account_turnover(
        conn: sqlite3.Connection, account: str, date_from: str, date_to: str
) -> ToolResult:
    def side(spec: tuple[str, str, str]) -> tuple[str, str, list[dict]]:
        file_name, label, sql = spec
        rows = [
            dict(r)
            for r in conn.execute(sql, (date_from, date_to, ROW_LIMIT)).fetchall()
        ]
        return file_name, label, rows

    key = ACCOUNT_ALIASES.get(account.strip())
    if key is None:
        return ToolResult(
            rows=[],
            warnings=[f"Счет {account!r} не поддерживается, доступные: 62.01, 60.01"],
        )
    if key == "62.01":
        debit = (
            "invoices_issued.csv",
            "выставлено покупателям",
            "SELECT number, date, counterparty, amount FROM invoices_issued WHERE date BETWEEN ? AND ? ORDER BY date LIMIT ?",
        )
        credit = (
            "payments.csv",
            "оплачено покупателями",
            "SELECT number, date, counterparty, amount FROM payments WHERE direction = 'in' AND date BETWEEN ? AND ? ORDER BY date LIMIT ?",
        )
    else:
        credit = (
            "invoices_issued.csv",
            "выставлено покупателям",
            "SELECT number, date, counterparty, amount FROM invoices_issued WHERE date BETWEEN ? AND ? ORDER BY date LIMIT ?",
        )
        debit = (
            "payments.csv",
            "оплачено покупателями",
            "SELECT number, date, counterparty, amount FROM payments WHERE direction = 'in' AND date BETWEEN ? AND ? ORDER BY date LIMIT ?",
        )

    debit_file, debit_label, debit_rows = side(debit)
    credit_file, credit_label, credit_rows = side(credit)

    debit_amounts = [r["amount"] for r in debit_rows]
    credit_amounts = [r["amount"] for r in credit_rows]

    result = ToolResult(
        rows=[
            {
                "account": key,
                "period": f"{date_from}..{date_to}",
                "turnover_debit": round(sum(debit_amounts), 2),
                "turnover_credit": round(sum(credit_amounts), 2),
                "debit_documents": len(debit_rows),
                "credit_documents": len(credit_rows),
            }
        ],
        calculations=[
            Calculation(
                operation="turnover_debit",
                expression=f"{debit_label}: {_expression(debit_amounts)}",
                result=round(sum(debit_amounts), 2),
            ),
            Calculation(
                operation="turnover_credit",
                expression=f"{credit_label}: {_expression(credit_amounts)}",
                result=round(sum(credit_amounts), 2),
            ),
        ],
        sources=[
            DataSource(file=debit_file, record_ids=[r["number"] for r in debit_rows]),
            DataSource(file=credit_file, record_ids=[r["number"] for r in credit_rows]),
        ],
    )

    if date_from < DATA_FROM or date_to > DATA_TO:
        result.warnings.append(
            "Запрошенный период выходит за границы, обороты неполные"
        )
    return result
