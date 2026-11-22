"""Parse 10-K HTML into Items (sections) made of text and table blocks.

10-K HTML from EDGAR is messy: deeply nested divs and spans, a table of contents that
repeats every Item heading, tables whose "$", "%" and ")" sit in their own cells, and
inline XBRL headers. The approach here:

1. Drop scripts, styles, hidden elements and the inline XBRL header.
2. Walk the DOM in order and emit a flat list of blocks: a text block for each leaf
   block-level element, and a table block (with rows) for each table.
3. A short block that reads like "Item 1A. Risk Factors" starts a new section.
4. The table of contents repeats every heading, so when an Item appears more than once
   we keep the occurrence with the most content.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from bs4 import BeautifulSoup, NavigableString, Tag

BLOCK_TAGS = {
    "p", "div", "h1", "h2", "h3", "h4", "h5", "h6", "li", "ul", "ol", "section",
    "article", "center", "blockquote", "body", "html", "table", "tr", "td", "th",
    "tbody", "thead", "dl", "dt", "dd", "hr",
}

ITEM_RE = re.compile(
    r"^(?:part\s+[ivx]+\s*[,.\-–—:]?\s*)?item\s+(\d{1,2}[a-c]?)\s*[.:\-–—]?\s*(.{0,200})$",
    re.IGNORECASE,
)
NOISE_RE = re.compile(r"^(?:part\s+[ivx]+\.?|table of contents|\d{1,3}|page)$", re.IGNORECASE)
NUMBER_CELL_RE = re.compile(r"\d")
YEAR_RE = re.compile(r"^(?:fiscal\s+)?(?:19|20)\d{2}$", re.IGNORECASE)


@dataclass
class Block:
    kind: str  # "text" or "table"
    text: str
    anchor: str | None = None
    rows: list[str] | None = None  # tables only
    header_rows: int = 0  # tables only: how many leading rows are column headers


@dataclass
class Section:
    item: str  # "1", "1A", "7", ... or "0" for the cover page
    title: str
    anchor: str | None
    blocks: list[Block] = field(default_factory=list)

    @property
    def word_count(self) -> int:
        return sum(len(b.text.split()) for b in self.blocks)


def clean_text(s: str) -> str:
    s = s.replace("\xa0", " ").replace("​", "")
    s = s.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    return re.sub(r"\s+", " ", s).strip()


def _is_hidden(tag: Tag) -> bool:
    style = (tag.get("style") or "").replace(" ", "").lower()
    return "display:none" in style


def _has_block_child(tag: Tag) -> bool:
    return any(isinstance(c, Tag) and c.name in BLOCK_TAGS for c in tag.descendants)


def _table_rows(table: Tag) -> list[list[str]]:
    rows = []
    for tr in table.find_all("tr"):
        cells = [clean_text(td.get_text(" ")) for td in tr.find_all(["td", "th"])]
        merged: list[str] = []
        for c in cells:
            if not c:
                continue
            if c in {")", "%", ")%", "%)"} and merged:
                merged[-1] += c
            elif merged and merged[-1] in {"$", "(", "$("}:
                merged[-1] = (merged[-1] + " " + c) if merged[-1] == "$" else merged[-1] + c
            else:
                merged.append(c)
        if merged:
            rows.append(merged)
    return rows


def _is_data_row(cells: list[str]) -> bool:
    return len(cells) >= 2 and any(
        NUMBER_CELL_RE.search(c) and not YEAR_RE.match(c) for c in cells[1:]
    )


def table_block(table: Tag, anchor: str | None) -> Block | None:
    rows = _table_rows(table)
    if not rows:
        return None
    header = 0
    while header < min(3, len(rows)) and not _is_data_row(rows[header]):
        header += 1
    if header == len(rows):  # no data rows at all: layout table, keep it but no header
        header = 0
    lines = [" | ".join(r) for r in rows]
    return Block(kind="table", text="\n".join(lines), anchor=anchor, rows=lines, header_rows=header)


def extract_blocks(html: str) -> list[Block]:
    soup = BeautifulSoup(html, "lxml")
    for t in soup(["script", "style", "head", "noscript"]):
        t.decompose()
    for t in soup.find_all(lambda x: isinstance(x, Tag) and (x.name or "").startswith("ix:header")):
        t.decompose()
    for t in soup.find_all(lambda x: isinstance(x, Tag) and _is_hidden(x)):
        t.decompose()

    blocks: list[Block] = []
    state = {"anchor": None}

    def emit_text(text: str) -> None:
        text = clean_text(text)
        if text and not NOISE_RE.match(text):
            blocks.append(Block(kind="text", text=text, anchor=state["anchor"]))

    def walk(el: Tag) -> None:
        anchor = el.get("id") or el.get("name")
        if anchor:
            state["anchor"] = anchor
        for a in el.find_all("a", attrs={"name": True}, recursive=False):
            state["anchor"] = a["name"]
        if el.name == "table":
            b = table_block(el, state["anchor"])
            if b:
                blocks.append(b)
            return
        if not _has_block_child(el):
            emit_text(el.get_text(" "))
            return
        buf: list[str] = []
        for child in el.children:
            if isinstance(child, NavigableString):
                buf.append(str(child))
            elif isinstance(child, Tag):
                if child.name in BLOCK_TAGS or _has_block_child(child):
                    emit_text(" ".join(buf))
                    buf = []
                    walk(child)
                else:
                    child_anchor = child.get("id") or child.get("name")
                    if child_anchor:
                        state["anchor"] = child_anchor
                    buf.append(child.get_text(" "))
        emit_text(" ".join(buf))

    root = soup.body or soup
    walk(root)
    return blocks


def _heading(block: Block, next_block: Block | None) -> tuple[str, str] | None:
    """Return (item, title) if this block is an Item heading."""
    if block.kind == "table":
        # Some filings lay out the real heading as a one-row table. A TOC has many rows.
        if block.rows and len(block.rows) == 1 and len(block.text) < 200:
            m = ITEM_RE.match(block.rows[0].replace(" | ", " "))
            if m:
                return m.group(1).upper(), clean_text(m.group(2))
        return None
    if len(block.text) > 220:
        return None
    m = ITEM_RE.match(block.text)
    if not m:
        return None
    title = clean_text(m.group(2))
    if not title and next_block and next_block.kind == "text" and len(next_block.text) < 120:
        title = next_block.text
    # A sentence that merely starts with "Item 7 of this report..." is not a heading.
    if title and title[0].islower():
        return None
    return m.group(1).upper(), title


def split_sections(blocks: list[Block]) -> list[Section]:
    segments: list[Section] = [Section(item="0", title="Cover page", anchor=None)]
    skip_next = False
    for i, b in enumerate(blocks):
        if skip_next:
            skip_next = False
            continue
        nxt = blocks[i + 1] if i + 1 < len(blocks) else None
        h = _heading(b, nxt)
        if h:
            item, title = h
            title_from_next = not ITEM_RE.match(b.text).group(2).strip() if b.kind == "text" else False
            segments.append(Section(item=item, title=title or f"Item {item}", anchor=b.anchor))
            skip_next = bool(title_from_next and title)
            continue
        segments[-1].blocks.append(b)

    cover = segments[0]
    cover.blocks = [b for b in cover.blocks if b.kind == "text"]  # drop TOC tables

    best: dict[str, Section] = {}
    order: list[str] = []
    for s in segments:
        if s.item not in best:
            order.append(s.item)
            best[s.item] = s
        elif s.word_count > best[s.item].word_count:
            best[s.item] = s
    return [best[k] for k in order if best[k].blocks]


def parse_filing(html: str) -> list[Section]:
    return split_sections(extract_blocks(html))
