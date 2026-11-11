"""Step 4 - Vector store (FAISS) + chunk metadata on disk.

Files written to the index directory:
  faiss.index      FAISS IndexFlatIP (inner product on normalised vectors = cosine)
  vectors.npy      same vectors as a numpy array (source of truth for re-use)
  chunks.jsonl     one JSON object per chunk; line i <-> vector row i <-> FAISS id i
  index_meta.json  embedder name, dim, chunk settings, file hashes, counts

`VectorStore.load()` + `.search()` is the interface the retrieval teammate uses.
If `faiss` is not installed, search falls back to a numpy dot product so that
nothing breaks, but `faiss.index` is only written when FAISS is available.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

try:  # optional at import time so the package can still be imported without it
    import faiss  # type: ignore

    HAVE_FAISS = True
except Exception:  # pragma: no cover - depends on environment
    faiss = None
    HAVE_FAISS = False

INDEX_FILE = "faiss.index"
VECTORS_FILE = "vectors.npy"
CHUNKS_FILE = "chunks.jsonl"
META_FILE = "index_meta.json"


class VectorStore:
    def __init__(self, vectors: np.ndarray, chunks: list[dict], meta: dict | None = None):
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        self.vectors = np.ascontiguousarray(vectors, dtype="float32")
        self.chunks = chunks
        self.meta = meta or {}
        self._index = None
        if HAVE_FAISS and len(chunks):
            self._index = faiss.IndexFlatIP(self.vectors.shape[1])
            self._index.add(self.vectors)

    # ------------------------------------------------------------------ search
    def search(self, query_vec: np.ndarray, k: int = 5) -> list[tuple[float, dict]]:
        """Return up to k (score, chunk_dict) pairs, best first. Score = cosine."""
        if not self.chunks:
            return []
        q = np.ascontiguousarray(np.asarray(query_vec, dtype="float32").reshape(1, -1))
        k = min(k, len(self.chunks))
        if self._index is not None:
            scores, ids = self._index.search(q, k)
            return [(float(s), self.chunks[i]) for s, i in zip(scores[0], ids[0]) if i >= 0]
        scores = (self.vectors @ q[0]).astype("float32")
        top = np.argsort(-scores)[:k]
        return [(float(scores[i]), self.chunks[i]) for i in top]

    # ------------------------------------------------------------- persistence
    def save(self, index_dir: str | Path) -> None:
        d = Path(index_dir)
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / VECTORS_FILE, self.vectors)
        with open(d / CHUNKS_FILE, "w", encoding="utf-8") as f:
            for c in self.chunks:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
        if self._index is not None:
            faiss.write_index(self._index, str(d / INDEX_FILE))
        else:
            (d / INDEX_FILE).unlink(missing_ok=True)
            log.warning("faiss not installed: %s was not written (numpy search will be used).", INDEX_FILE)
        with open(d / META_FILE, "w", encoding="utf-8") as f:
            json.dump(
                {**self.meta, "faiss_used": self._index is not None, "num_chunks": len(self.chunks)},
                f, indent=2, ensure_ascii=False,
            )

    @classmethod
    def load(cls, index_dir: str | Path) -> "VectorStore":
        d = Path(index_dir)
        if not (d / CHUNKS_FILE).exists():
            raise FileNotFoundError(f"No index found in {d}. Run: python ingest.py")
        vectors = np.load(d / VECTORS_FILE)
        with open(d / CHUNKS_FILE, encoding="utf-8") as f:
            chunks = [json.loads(line) for line in f if line.strip()]
        meta = json.loads((d / META_FILE).read_text(encoding="utf-8")) if (d / META_FILE).exists() else {}
        return cls(vectors, chunks, meta)

    @classmethod
    def empty(cls, dim: int, meta: dict | None = None) -> "VectorStore":
        return cls(np.zeros((0, dim), dtype="float32"), [], meta)
