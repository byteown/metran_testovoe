import pytest

from fin_agent.config import get_settings
from fin_agent.rag.chunks import split_document
from fin_agent.rag.index import DocumentIndex
from fin_agent.storage.db import load_database


@pytest.fixture(scope="session")
def conn():
    connection, _ = load_database()
    yield connection
    connection.close()


@pytest.fixture(scope="session")
def index():
    settings = get_settings()
    chunks = [
        c
        for p in sorted(settings.data.docs_path.glob("*.md"))
        for c in split_document(p)
    ]
    return DocumentIndex(chunks, settings.rag.model)
