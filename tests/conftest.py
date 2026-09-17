import pytest

from fin_agent.storage.db import load_database


@pytest.fixture(scope="session")
def conn():
    connection, _ = load_database()
    yield connection
    connection.close()
