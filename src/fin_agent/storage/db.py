import csv
import logging
import sqlite3
from datetime import date, timedelta

from fin_agent.config import get_settings

logging.basicConfig(level=get_settings().log.log_level)
logger = logging.getLogger(__name__)

CHECK_TOLERANCE = 0.01  # допуск в одну копейку


def _issued_row(r: dict) -> tuple:
    due = date.fromisoformat(r["date"]) + timedelta(days=int(r["term_days"]))
    return (
        r["number"], r["date"], r["counterparty"], float(r["amount"]), int(r["term_days"]), r["status"],
        float(r["paid"]), due.isoformat(),
    )


def _payment_row(r: dict) -> tuple:
    invoice_no = r["purpose"].split()[-1]
    direction = "in" if invoice_no.startswith("СБ") else "out"
    return r["number"], r["date"], r["counterparty"], float(r["amount"]), r["purpose"], invoice_no, direction


def _scalar(conn: sqlite3.Connection, sql: str, params: tuple = ()) -> float | None:
    row = conn.execute(sql, params).fetchone()
    return None if row is None else row["v"]


def _verify_totals(conn: sqlite3.Connection) -> list[str]:
    problems: list[str] = []

    def compare(label: str, detail: float | None, expected: float | None) -> None:
        if detail is None or expected is None:
            problems.append(f"{label}: нет данных для сверки")
            return
        diff = detail - expected
        if abs(diff) > CHECK_TOLERANCE:
            problems.append(
                f"{label}: детали {detail:,.2f}, ОСВ {expected:,.2f} "
                f"(расхождение {diff:+,.2f})"
            )

    compare(
        "Обороты Дт 62.01 (выставлено покупателям)",
        _scalar(conn, "SELECT SUM(amount) AS v FROM invoices_issued"),
        _scalar(conn, "SELECT turnover_dt AS v FROM account_balances WHERE account = ?", ("62.01",)),
    )

    balance_62 = _scalar(
        conn, "SELECT balance_end_dt AS v FROM account_balances WHERE account = ?", ("62.01",)
    )
    compare(
        "Сальдо Дт 62.01 (остаток по счетам)",
        _scalar(conn, "SELECT SUM(amount - paid) AS v FROM invoices_issued"),
        balance_62,
    )

    compare(
        "Сальдо Дт 62.01 (дебиторка receivables)",
        _scalar(conn, "SELECT SUM(amount) AS v FROM receivables"),
        balance_62,
    )

    compare(
        "Обороты Кт 60.01 (счета поставщиков)",
        _scalar(conn, "SELECT SUM(amount) AS v FROM invoices_received"),
        _scalar(conn, "SELECT turnover_ct AS v FROM account_balances WHERE account = ?", ("60.01",)),
    )

    return problems


def load_database():
    path = get_settings().data.data_path

    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.row_factory = sqlite3.Row

    conn.execute("""
                 CREATE TABLE account_balances
                 (
                     account          TEXT PRIMARY KEY,
                     balance_start_dt REAL NOT NULL,
                     balance_start_ct REAL NOT NULL,
                     turnover_dt      REAL NOT NULL,
                     turnover_ct      REAL NOT NULL,
                     balance_end_dt   REAL NOT NULL,
                     balance_end_ct   REAL NOT NULL
                 )
                 """)
    with open(path / "account_balances.csv", encoding="utf-8", newline="") as f:
        rows = [
            (r["account"], float(r["balance_start_dt"]), float(r["balance_start_ct"]), float(r["turnover_dt"]),
             float(r["turnover_ct"]), float(r["balance_end_dt"]), float(r["balance_end_ct"]))
            for r in csv.DictReader(f)
        ]
    conn.executemany("INSERT INTO account_balances VALUES (?, ?, ?, ?, ?, ?, ?)", rows)
    conn.commit()

    conn.execute("""
                 CREATE TABLE counterparties
                 (
                     name    TEXT NOT NULL,
                     inn     TEXT PRIMARY KEY,
                     type    TEXT NOT NULL,
                     region  TEXT NOT NULL,
                     manager TEXT NOT NULL
                 )
                 """)
    with open(path / "counterparties.csv", encoding="utf-8", newline="") as f:
        rows = [
            (r["name"], r["inn"], r["type"], r["region"], r["manager"])
            for r in csv.DictReader(f)
        ]
    conn.executemany("INSERT INTO counterparties VALUES (?, ?, ?, ?, ?)", rows)
    conn.commit()

    conn.execute("""
                 CREATE TABLE invoices_issued
                 (
                     number       TEXT PRIMARY KEY,
                     date         TEXT    NOT NULL,
                     counterparty TEXT    NOT NULL,
                     amount       REAL    NOT NULL,
                     term_days    INTEGER NOT NULL,
                     status       TEXT    NOT NULL,
                     paid         REAL    NOT NULL,
                     due_date     TEXT    NOT NULL
                 )
                 """)
    with open(path / "invoices_issued.csv", encoding="utf-8", newline="") as f:
        rows = [
            _issued_row(r) for r in csv.DictReader(f)
        ]
    conn.executemany("INSERT INTO invoices_issued VALUES (?, ?, ?, ?, ?, ?, ?, ?)", rows)
    conn.commit()

    conn.execute("""
                 CREATE TABLE invoices_received
                 (
                     number       TEXT PRIMARY KEY,
                     date         TEXT    NOT NULL,
                     counterparty TEXT    NOT NULL,
                     amount       REAL    NOT NULL,
                     term_days    INTEGER NOT NULL,
                     status       TEXT    NOT NULL,
                     paid         REAL    NOT NULL,
                     due_date     TEXT    NOT NULL
                 )
                 """)
    with open(path / "invoices_received.csv", encoding="utf-8", newline="") as f:
        rows = [
            _issued_row(r) for r in csv.DictReader(f)
        ]
    conn.executemany("INSERT INTO invoices_received VALUES (?, ?, ?, ?, ?, ?, ?, ?)", rows)
    conn.commit()

    conn.execute("""
                 CREATE TABLE payments
                 (
                     number       TEXT PRIMARY KEY,
                     date         TEXT NOT NULL,
                     counterparty TEXT NOT NULL,
                     amount       REAL NOT NULL,
                     purpose      TEXT NOT NULL,
                     invoice_no   TEXT NOT NULL,
                     direction    TEXT NOT NULL
                 )
                 """)
    with open(path / "payments.csv", encoding="utf-8", newline="") as f:
        rows = [
            _payment_row(r) for r in csv.DictReader(f)
        ]
    conn.executemany("INSERT INTO payments VALUES (?, ?, ?, ?, ?, ?, ?)", rows)
    conn.commit()

    conn.execute("""
                 CREATE TABLE receivables
                 (
                     counterparty TEXT    NOT NULL,
                     contract     TEXT    NOT NULL,
                     invoice      TEXT PRIMARY KEY,
                     amount       REAL    NOT NULL,
                     invoice_date TEXT    NOT NULL,
                     overdue_days INTEGER NOT NULL,
                     status       TEXT    NOT NULL
                 )
                 """)
    with open(path / "receivables.csv", encoding="utf-8", newline="") as f:
        rows = [
            (r["counterparty"], r["contract"], r["invoice"], float(r["amount"]), r["invoice_date"],
             int(r["overdue_days"]), r["status"])
            for r in csv.DictReader(f)
        ]
    conn.executemany("INSERT INTO receivables VALUES (?, ?, ?, ?, ?, ?, ?)", rows)
    conn.commit()

    problems = _verify_totals(conn)
    for p in problems:
        logger.warning("Сверка данных: %s", p)

    conn.execute("PRAGMA query_only = ON")
    return conn, problems
