import logging

from fin_agent.agent.registry import ToolContext
from fin_agent.config import get_settings
from fin_agent.rag.chunks import split_document
from fin_agent.rag.index import DocumentIndex
from fin_agent.storage.db import load_database

logger = logging.getLogger(__name__)


def configure_logging() -> None:
    logging.basicConfig(
        level=get_settings().log.log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def build_context() -> tuple[ToolContext, list[str]]:
    settings = get_settings()

    conn, load_warnings = load_database()
    logger.info("База загружена, расхождений при сверке: %d", len(load_warnings))

    chunks = [
        chunk
        for path in sorted(settings.data.docs_path.glob("*.md"))
        for chunk in split_document(path)
    ]
    index = DocumentIndex(chunks, settings.rag.model)
    logger.info("Индекс регламентов построен: %d разделов", len(chunks))

    return ToolContext(conn=conn, index=index), load_warnings
