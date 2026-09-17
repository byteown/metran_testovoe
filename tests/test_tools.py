import sqlite3

import pytest

from fin_agent.tools import finance
from fin_agent.tools.rag import _is_verbatim, search_regulations


def test_receivables_for_counterparty(conn):
    result = finance.get_receivables(conn, counterparty="ООО Ромашка")

    assert result.calculations[0].result == pytest.approx(2600080.15)
    assert sorted(result.sources[0].record_ids) == ["СБ-00010", "СБ-00026"]
    assert result.warnings == []


def test_unknown_counterparty_returns_nothing(conn):
    result = finance.get_receivables(conn, counterparty="ООО Такого Нет")

    assert result.rows == []
    assert result.calculations == []
    assert result.sources == []


def test_database_is_read_only(conn):
    with pytest.raises(sqlite3.OperationalError, match="readonly"):
        conn.execute("DELETE FROM receivables")


def test_rag_answer_contains_verbatim_quote(index):
    result = search_regulations(index, "Что делать при просрочке дебиторки 45 дней?")
    assert result.sources
    for source in result.sources:
        assert _is_verbatim(source.quote, source.file)
