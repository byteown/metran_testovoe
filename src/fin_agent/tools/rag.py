import re
from functools import lru_cache
from pathlib import Path

import numpy as np

from fin_agent.config import get_settings
from fin_agent.rag.chunks import Chunk
from fin_agent.rag.index import DocumentIndex
from fin_agent.schemas import DocumentSource
from fin_agent.tools.base import ToolResult

BULLET = re.compile(r"^\s*[-*]\s+")
MIN_QUOTE_CHARS = 30


def _normalize_spaces(text: str) -> str:
    return " ".join(text.split())


@lru_cache(maxsize=8)
def _file_text(path: str) -> str:
    return _normalize_spaces(Path(path).read_text(encoding="utf-8"))


def _is_verbatim(quote: str, file: str) -> bool:
    return _normalize_spaces(quote) in _file_text(file)


def _split_fragments(text: str) -> list[str]:
    fragments: list[str] = []

    for paragraph in text.split("\n\n"):
        lines = paragraph.splitlines()
        if not any(BULLET.match(line) for line in lines):
            fragments.append(paragraph)
            continue

        current: list[str] = []
        for line in lines:
            if BULLET.match(line) and current:
                fragments.append("\n".join(current))
                current = [line]
            else:
                current.append(line)
        if current:
            fragments.append("\n".join(current))

    return [
        f.strip()
        for f in fragments
        if not f.strip().startswith("#")
           and len(_normalize_spaces(f)) >= MIN_QUOTE_CHARS
    ]


def _best_quote(index: DocumentIndex, question: str, chunk: Chunk) -> str | None:
    fragments = _split_fragments(chunk.text)
    if not fragments:
        return None
    vectors = index.encode(fragments)
    query = index.encode([question])[0]
    best = fragments[int(np.argmax(vectors @ query))]
    return BULLET.sub("", best).strip()


def search_regulations(index: DocumentIndex, question: str) -> ToolResult:
    rag = get_settings().rag
    hits = index.search(question, rag.top_k, rag.min_score)
    if not hits:
        return ToolResult(rows=[])

    rows, sources, warnings = [], [], []
    for chunk, score in hits:
        quote = _best_quote(index, question, chunk)
        if quote is None or not _is_verbatim(quote, chunk.file):
            warnings.append(
                f"Цитата из раздела {chunk.section} не подтвердилась, отброшена"
            )
            continue
        sources.append(
            DocumentSource(file=chunk.file, section=chunk.section, quote=quote)
        )
        rows.append(
            {
                "section": chunk.section,
                "breadcrumb": chunk.breadcrumb,
                "score": round(score, 3),
                "text": chunk.text,
            }
        )

    if not sources:
        return ToolResult(rows=[], warnings=warnings)
    return ToolResult(rows=rows, sources=sources, warnings=warnings)
