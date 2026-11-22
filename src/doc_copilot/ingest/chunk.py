"""Chunking strategies.

baseline:    each section becomes one stream of words (tables flattened into the stream) cut
             into fixed windows of `size` words with `overlap` words shared between windows.
             Simple and predictable, but a table can be cut in half, and the second half
             loses its column headers ("2024 | 2023"), so a number lands in a chunk that no
             longer says which year it belongs to.

table_aware: prose is windowed the same way, but windows never cross a table. Each table
             becomes its own chunk (split by rows if large), and every table chunk repeats
             the caption line before it and the table's header rows.

Both produce stable chunk IDs: {TICKER}-{YEAR}-{ITEM}-{b|t}{index:03d}.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from .parse import Block, Section

STRATEGIES = ("baseline", "table_aware")
_TOKEN = re.compile(r"\S+\s*")


@dataclass
class Chunk:
    id: str
    section: str
    section_title: str
    anchor: str | None
    chunk_index: int
    kind: str  # "text" or "table"
    text: str

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(f"{self.section_title}\n{self.text}".encode()).hexdigest()


def _windows(text: str, size: int, overlap: int) -> list[str]:
    tokens = _TOKEN.findall(text)
    if not tokens:
        return []
    if len(tokens) <= size:
        return ["".join(tokens).strip()]
    step = max(1, size - overlap)
    out = []
    for start in range(0, len(tokens), step):
        out.append("".join(tokens[start : start + size]).strip())
        if start + size >= len(tokens):
            break
    return out


def _caption_for(blocks: list[Block], i: int) -> str | None:
    """The short paragraph right before a table usually says what the table is."""
    if i == 0:
        return None
    prev = blocks[i - 1]
    if prev.kind != "text" or len(prev.text.split()) > 60:
        return None
    return prev.text


def _table_chunks(block: Block, caption: str | None, size: int) -> list[str]:
    rows = block.rows or block.text.split("\n")
    header = rows[: block.header_rows]
    body = rows[block.header_rows :]
    prefix = ([caption] if caption else []) + header
    prefix_words = sum(len(r.split()) for r in prefix)
    budget = max(size - prefix_words, 20)
    pieces, current, words = [], [], 0
    for r in body:
        n = len(r.split())
        if current and words + n > budget:
            pieces.append(current)
            current, words = [], 0
        current.append(r)
        words += n
    if current:
        pieces.append(current)
    return ["\n".join(prefix + p) for p in pieces] or ["\n".join(prefix)]


def chunk_section(
    section: Section, strategy: str, size: int, overlap: int
) -> list[tuple[str, str, str | None]]:
    """Return (kind, text, anchor) tuples for one section."""
    if strategy == "baseline":
        stream = "\n".join(b.text for b in section.blocks)
        has_table = any(b.kind == "table" for b in section.blocks)
        return [
            ("mixed" if has_table and "|" in w else "text", w, section.anchor)
            for w in _windows(stream, size, overlap)
        ]
    if strategy == "table_aware":
        out: list[tuple[str, str, str | None]] = []
        prose: list[str] = []

        def flush() -> None:
            if prose:
                out.extend(("text", w, section.anchor) for w in _windows("\n".join(prose), size, overlap))
                prose.clear()

        for i, b in enumerate(section.blocks):
            if b.kind == "table":
                flush()
                caption = _caption_for(section.blocks, i)
                out.extend(("table", t, b.anchor or section.anchor) for t in _table_chunks(b, caption, size))
            else:
                prose.append(b.text)
        flush()
        return out
    raise ValueError(f"Unknown chunking strategy {strategy!r}; use one of {STRATEGIES}")


def chunk_filing(
    sections: list[Section], ticker: str, year: int, strategy: str, size: int, overlap: int
) -> list[Chunk]:
    code = "b" if strategy == "baseline" else "t"
    chunks: list[Chunk] = []
    for s in sections:
        for idx, (kind, text, anchor) in enumerate(chunk_section(s, strategy, size, overlap)):
            chunks.append(
                Chunk(
                    id=f"{ticker}-{year}-{s.item}-{code}{idx:03d}",
                    section=s.item,
                    section_title=s.title,
                    anchor=anchor,
                    chunk_index=idx,
                    kind=kind,
                    text=text,
                )
            )
    return chunks
