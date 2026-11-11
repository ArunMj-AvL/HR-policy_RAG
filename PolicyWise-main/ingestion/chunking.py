"""Step 2 - Chunking with overlap.

Design:
  * Chunks NEVER cross a page boundary, so every chunk has exactly one page
    number and citations are exact.
  * Text is split into sentences and packed greedily up to `chunk_size`
    characters, so a rule such as "Tier 1 cities: INR 6000 per night" is not
    cut in half.
  * Each new chunk starts by repeating the last sentence(s) of the previous
    chunk, up to `overlap` characters. This is the overlap.
  * The section heading (e.g. "Hotel limits") is kept in the chunk text and
    in the `section` field, because the heading carries most of the meaning.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass

from .extract import PolicyDocument

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")
_HEADING_MAX_LEN = 40


@dataclass
class Chunk:
    chunk_id: str
    doc_name: str
    doc_label: str  # "Travel Policy v2.0"
    version: str
    status: str  # current | superseded | unknown
    effective_date: str
    superseded_date: str
    page: int  # 1-based
    chunk_index: int  # position within the page
    section: str  # heading of the first sentence in the chunk
    text: str
    char_count: int

    def to_dict(self) -> dict:
        return asdict(self)


def _is_heading(line: str, has_next: bool) -> bool:
    line = line.strip()
    return (
        has_next
        and 0 < len(line) <= _HEADING_MAX_LEN
        and line[-1] not in ".;:,!?"
        and "|" not in line
    )


def _page_units(page_text: str) -> list[tuple[str, str]]:
    """Return [(sentence, section_heading), ...] for one page.

    The first line (the 'Title vX | Effective ...' banner) is skipped here; it
    is stored as document metadata and re-added to each chunk's embedding text.
    """
    lines = page_text.split("\n")
    if lines and "|" in lines[0]:
        lines = lines[1:]

    units: list[tuple[str, str]] = []
    section = ""
    buf: list[str] = []

    def flush():
        if buf:
            paragraph = " ".join(buf).strip()
            for sent in _SENT_SPLIT.split(paragraph):
                if sent.strip():
                    units.append((sent.strip(), section))
            buf.clear()

    for i, raw in enumerate(lines):
        line = raw.strip()
        if not line:
            flush()
            continue
        if _is_heading(line, has_next=i + 1 < len(lines)):
            flush()
            section = line
        else:
            buf.append(line)
    flush()
    return units


def _split_long(sentence: str, limit: int) -> list[str]:
    """Hard-split a sentence longer than `limit` on word boundaries."""
    words, parts, cur = sentence.split(), [], ""
    for w in words:
        if cur and len(cur) + 1 + len(w) > limit:
            parts.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        parts.append(cur)
    return parts


def _render(units: list[tuple[str, str]]) -> str:
    """Join sentences; put the section heading on its own line when it changes."""
    out, last_section = [], None
    for sent, section in units:
        if section and section != last_section:
            out.append(f"{section}:")
            last_section = section
        out.append(sent)
    return "\n".join(out)


def chunk_page(page_text: str, chunk_size: int = 400, overlap: int = 100) -> list[list[tuple[str, str]]]:
    """Group a page's sentences into overlapping chunks (lists of units)."""
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    units: list[tuple[str, str]] = []
    for sent, section in _page_units(page_text):
        if len(sent) > chunk_size:
            units.extend((p, section) for p in _split_long(sent, chunk_size))
        else:
            units.append((sent, section))

    chunks: list[list[tuple[str, str]]] = []
    current: list[tuple[str, str]] = []
    n_new = 0  # sentences in `current` that were not carried over as overlap

    for unit in units:
        if current and len(_render(current + [unit])) > chunk_size:
            chunks.append(current)
            # carry the tail of the finished chunk as overlap
            tail: list[tuple[str, str]] = []
            for u in reversed(current):
                if len(_render([u] + tail)) > overlap:
                    break
                tail.insert(0, u)
            current = tail + [unit]
            n_new = 1
        else:
            current.append(unit)
            n_new += 1
    if current and n_new > 0:
        chunks.append(current)
    return chunks


def chunk_document(doc: PolicyDocument, chunk_size: int = 400, overlap: int = 100) -> list[Chunk]:
    result: list[Chunk] = []
    for page in doc.pages:
        for idx, units in enumerate(chunk_page(page.text, chunk_size, overlap)):
            text = _render(units)
            digest = hashlib.sha1(f"{doc.doc_name}|{page.page}|{text}".encode()).hexdigest()[:10]
            stem = doc.doc_name.rsplit(".", 1)[0]
            result.append(
                Chunk(
                    chunk_id=f"{stem}_p{page.page}_c{idx}_{digest}",
                    doc_name=doc.doc_name,
                    doc_label=doc.label,
                    version=doc.version,
                    status=doc.status,
                    effective_date=doc.effective_date,
                    superseded_date=doc.superseded_date,
                    page=page.page,
                    chunk_index=idx,
                    section=units[0][1],
                    text=text,
                    char_count=len(text),
                )
            )
    return result


def embedding_text(chunk: Chunk) -> str:
    """Text that is actually embedded: document label + status + chunk text.

    Putting 'Travel Policy v1.0 (superseded)' in front of every chunk lets the
    vector search tell the old and new Travel Policy apart.
    """
    return f"{chunk.doc_label} ({chunk.status})\n{chunk.text}"
