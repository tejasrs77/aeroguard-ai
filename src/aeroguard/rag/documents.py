"""Load Markdown guidance into traceable retrieval chunks."""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class KnowledgeChunk:
    """A section of one source document with a stable citation identifier."""

    chunk_id: str
    source: str
    document_title: str
    section: str
    text: str
    sha256: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "overview"


def _chunk_document(path: Path) -> list[KnowledgeChunk]:
    content = path.read_text(encoding="utf-8").strip()
    if not content:
        raise ValueError(f"Knowledge document is empty: {path}")

    lines = content.splitlines()
    document_title = path.stem.replace("_", " ").title()
    if lines and lines[0].startswith("# "):
        document_title = lines[0][2:].strip()

    sections: list[tuple[str, list[str]]] = []
    current_heading = "Overview"
    current_lines: list[str] = []
    for line in lines:
        if line.startswith("## "):
            if any(item.strip() for item in current_lines):
                sections.append((current_heading, current_lines))
            current_heading = line[3:].strip()
            current_lines = []
        elif not line.startswith("# "):
            current_lines.append(line)
    if any(item.strip() for item in current_lines):
        sections.append((current_heading, current_lines))

    chunks: list[KnowledgeChunk] = []
    for heading, body_lines in sections:
        body = "\n".join(body_lines).strip()
        if not body:
            continue
        text = f"{document_title}\n{heading}\n{body}"
        chunks.append(
            KnowledgeChunk(
                chunk_id=f"{path.stem}#{_slug(heading)}",
                source=path.name,
                document_title=document_title,
                section=heading,
                text=text,
                sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
            )
        )
    if not chunks:
        raise ValueError(f"No retrievable sections found in {path}")
    return chunks


def load_knowledge_chunks(knowledge_dir: Path) -> list[KnowledgeChunk]:
    """Load every Markdown source in stable filename and section order."""

    paths = sorted(knowledge_dir.glob("*.md"))
    if not paths:
        raise FileNotFoundError(f"No Markdown knowledge files found in {knowledge_dir}")
    chunks = [chunk for path in paths for chunk in _chunk_document(path)]
    identifiers = [chunk.chunk_id for chunk in chunks]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Knowledge chunk IDs must be unique")
    return chunks
