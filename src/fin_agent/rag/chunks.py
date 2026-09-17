import re
from dataclasses import dataclass
from pathlib import Path

HEADING = re.compile(r"^(#{2,4})\s+(.+?)\s*$", re.MULTILINE)


@dataclass(frozen=True)
class Chunk:
    file: str
    section: str
    breadcrumb: str
    text: str


def split_document(path: Path) -> list[Chunk]:
    text = path.read_text(encoding="utf-8")
    matches = list(HEADING.finditer(text))

    chunks: list[Chunk] = []
    parent = ""
    for i, m in enumerate(matches):
        level, title = len(m.group(1)), m.group(2)
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()

        if level <= 2:
            continue
        if level == 3:
            parent = title
            breadcrumb = title
        else:
            breadcrumb = f"{parent} › {title}"

        chunks.append(
            Chunk(
                file=f"docs/{path.name}",
                section=title,
                breadcrumb=breadcrumb,
                text=body,
            )
        )
    return chunks
