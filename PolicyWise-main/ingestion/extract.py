"""Step 1 - PDF text extraction with page numbers and policy metadata.

Every page is extracted separately so the page number (1-based, as a reader
sees it) travels with the text all the way to the final citation.
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)

# Matches headers such as:
#   "Travel Policy v2.0 | Effective 2026-10-01"
#   "Travel Policy v1.0 | SUPERSEDED 2025-01-01"
_HEADER_RE = re.compile(
    r"^(?P<title>.+?)\s+v(?P<version>\d+(?:\.\d+)*)\s*\|\s*"
    r"(?P<status>Effective|SUPERSEDED)\s+(?P<date>\d{4}-\d{2}-\d{2})\s*$",
    re.IGNORECASE,
)


@dataclass
class PageText:
    page: int  # 1-based
    text: str


@dataclass
class PolicyDocument:
    doc_name: str  # file name, e.g. "G1_policy_1.pdf"  (used in citations)
    sha256: str  # hash of the file bytes (used to detect duplicate uploads)
    title: str = ""  # e.g. "Travel Policy"
    version: str = ""  # e.g. "2.0"
    status: str = "unknown"  # "current" | "superseded" | "unknown"
    effective_date: str = ""  # ISO date, set when status == "current"
    superseded_date: str = ""  # ISO date, set when status == "superseded"
    pages: list[PageText] = field(default_factory=list)

    @property
    def label(self) -> str:
        """Human friendly name used in citations, e.g. 'Travel Policy v2.0'."""
        if self.title and self.version:
            return f"{self.title} v{self.version}"
        return self.title or self.doc_name


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _clean(text: str) -> str:
    text = text.replace(" ", " ").replace("\r\n", "\n").replace("\r", "\n")
    # drop trailing spaces and collapse 3+ blank lines
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _extract_pages_pypdf(path: Path) -> list[str]:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    return [(page.extract_text() or "") for page in reader.pages]


def _extract_pages_pdfplumber(path: Path) -> list[str]:
    import pdfplumber

    with pdfplumber.open(str(path)) as pdf:
        return [(page.extract_text() or "") for page in pdf.pages]


def parse_header(first_line: str) -> dict:
    """Parse 'Title vX.Y | Effective/SUPERSEDED date' into metadata."""
    m = _HEADER_RE.match(first_line.strip())
    if not m:
        return {}
    superseded = m.group("status").lower() == "superseded"
    return {
        "title": m.group("title").strip(),
        "version": m.group("version"),
        "status": "superseded" if superseded else "current",
        "effective_date": "" if superseded else m.group("date"),
        "superseded_date": m.group("date") if superseded else "",
    }


def extract_pdf(path: str | Path) -> PolicyDocument:
    """Extract all pages of one PDF. Empty pages are kept out of the result."""
    path = Path(path)
    try:
        raw_pages = _extract_pages_pypdf(path)
    except Exception as exc:  # corrupt file, unsupported encryption, ...
        log.warning("pypdf failed on %s (%s); trying pdfplumber", path.name, exc)
        raw_pages = []

    if not any(p.strip() for p in raw_pages):
        try:
            raw_pages = _extract_pages_pdfplumber(path)
        except Exception as exc:
            raise RuntimeError(f"Could not extract text from {path.name}: {exc}") from exc

    pages = [
        PageText(page=i, text=_clean(t))
        for i, t in enumerate(raw_pages, start=1)
        if t.strip()
    ]
    if not pages:
        raise RuntimeError(
            f"{path.name} has no extractable text (scanned PDF? OCR is not supported)."
        )

    doc = PolicyDocument(doc_name=path.name, sha256=file_sha256(path), pages=pages)
    # The title/version/status header sits on the first line of page 1.
    meta = parse_header(pages[0].text.split("\n", 1)[0])
    if meta:
        for k, v in meta.items():
            setattr(doc, k, v)
    else:
        log.warning("%s: no 'Title vX | Effective/SUPERSEDED date' header found", path.name)
        doc.title = path.stem
    return doc
