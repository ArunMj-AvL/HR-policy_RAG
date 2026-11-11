"""End-to-end ingestion: PDFs -> pages -> chunks -> embeddings -> FAISS index.

Re-running is safe (idempotent):
  * a chunk id is derived from file name + page + text, so unchanged chunks
    are re-used (not re-embedded and never stored twice);
  * chunks of files that were edited or deleted are dropped;
  * a second copy of the same PDF under another name (same sha256) is skipped.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .chunking import Chunk, chunk_document, embedding_text
from .embeddings import get_embedder
from .extract import PolicyDocument, extract_pdf
from .index_store import META_FILE, VectorStore
from config import DATA_DIR, INDEX_DIR

log = logging.getLogger(__name__)


def run_ingestion(
    pdf_dir: str | Path =  DATA_DIR,
    index_dir: str | Path = INDEX_DIR,
    chunk_size: int = 400,
    overlap: int = 100,
    embedder_choice: str = "auto",
    rebuild: bool = False,
) -> dict:
    pdf_dir, index_dir = Path(pdf_dir), Path(index_dir)
    pdf_paths = sorted(pdf_dir.glob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(f"No PDF files found in {pdf_dir}")

    embedder = get_embedder(embedder_choice)
    log.info("Embedder: %s (dim=%d)", embedder.name, embedder.dim)

    # 1) extract ---------------------------------------------------------------
    docs: list[PolicyDocument] = []
    seen_hashes: dict[str, str] = {}
    skipped_duplicates: list[str] = []
    for p in pdf_paths:
        doc = extract_pdf(p)
        if doc.sha256 in seen_hashes:
            log.warning("Skipping %s: identical content to %s", p.name, seen_hashes[doc.sha256])
            skipped_duplicates.append(p.name)
            continue
        seen_hashes[doc.sha256] = p.name
        docs.append(doc)
        log.info("Extracted %-18s pages=%d  %s (%s)", p.name, len(doc.pages), doc.label, doc.status)

    # 2) chunk -----------------------------------------------------------------
    chunks: list[Chunk] = [c for d in docs for c in chunk_document(d, chunk_size, overlap)]
    log.info("Created %d chunks (size=%d, overlap=%d)", len(chunks), chunk_size, overlap)

    # 3) re-use vectors we already have, embed only what is new ------------------
    existing: dict[str, np.ndarray] = {}
    if not rebuild and (index_dir / META_FILE).exists():
        try:
            old = VectorStore.load(index_dir)
            if old.meta.get("embedder") == embedder.name:
                existing = {c["chunk_id"]: old.vectors[i] for i, c in enumerate(old.chunks)}
            else:
                log.info("Embedder changed (%s -> %s): re-embedding everything.",
                         old.meta.get("embedder"), embedder.name)
        except Exception as exc:
            log.warning("Could not read existing index (%s); rebuilding.", exc)

    # de-duplicate by chunk id (same text on same page of the same file)
    unique: dict[str, Chunk] = {}
    for c in chunks:
        unique.setdefault(c.chunk_id, c)
    ordered = list(unique.values())

    to_embed = [c for c in ordered if c.chunk_id not in existing]
    if to_embed:
        new_vecs = embedder.encode([embedding_text(c) for c in to_embed])
        for c, v in zip(to_embed, new_vecs):
            existing[c.chunk_id] = v
    reused = len(ordered) - len(to_embed)
    vectors = (
        np.vstack([existing[c.chunk_id] for c in ordered]).astype("float32")
        if ordered else np.zeros((0, embedder.dim), dtype="float32")
    )

    # 4) store -----------------------------------------------------------------
    meta = {
        "embedder": embedder.name,
        "dim": embedder.dim,
        "chunk_size": chunk_size,
        "overlap": overlap,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "documents": [
            {"doc_name": d.doc_name, "label": d.label, "status": d.status,
             "pages": len(d.pages), "sha256": d.sha256}
            for d in docs
        ],
    }
    store = VectorStore(vectors, [c.to_dict() for c in ordered], meta)
    store.save(index_dir)

    summary = {
        "documents": len(docs),
        "pages": sum(len(d.pages) for d in docs),
        "chunks_total": len(ordered),
        "chunks_embedded_now": len(to_embed),
        "chunks_reused": reused,
        "duplicate_files_skipped": skipped_duplicates,
        "embedder": embedder.name,
        "faiss_used": store._index is not None,
        "index_dir": str(index_dir),
    }
    log.info("Done: %s", summary)
    return summary
