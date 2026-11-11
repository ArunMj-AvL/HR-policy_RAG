"""Step 3 - Embedding models.

Default : sentence-transformers all-MiniLM-L6-v2 (384-dim, free, runs on CPU).
Fallback: a deterministic hashing embedder (pure numpy, no downloads) so the
          pipeline still runs offline / in CI. It is weaker than MiniLM and is
          clearly reported in the logs and in index/index_meta.json.

All vectors are L2-normalised, so inner product == cosine similarity.
"""

from __future__ import annotations

import hashlib
import logging
import re

import numpy as np

log = logging.getLogger(__name__)

MINILM_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class Embedder:
    name: str
    dim: int

    def encode(self, texts: list[str]) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError


class SentenceTransformerEmbedder(Embedder):
    def __init__(self, model_name: str = MINILM_NAME):
        from sentence_transformers import SentenceTransformer

        self.name = model_name
        self._model = SentenceTransformer(model_name)
        self.dim = int(self._model.get_sentence_embedding_dimension())

    def encode(self, texts: list[str]) -> np.ndarray:
        vecs = self._model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True
        )
        return np.asarray(vecs, dtype="float32")


_STOPWORDS = frozenset(
    "a an the of for to in on at by is are was were be been and or if it this that "
    "with as from can i my me we our you your what which who how do does".split()
)
_TOKEN_RE = re.compile(r"[a-z0-9]+")


class HashingEmbedder(Embedder):
    """Bag of words + bigrams hashed into a fixed-size vector (offline fallback)."""

    def __init__(self, dim: int = 1024):
        self.dim = dim
        self.name = f"hashing-bow-{dim}"

    @staticmethod
    def _stem(tok: str) -> str:
        for suf in ("ing", "ed", "es", "s"):
            if tok.endswith(suf) and len(tok) - len(suf) >= 3:
                return tok[: -len(suf)]
        return tok

    def _features(self, text: str) -> list[str]:
        toks = [self._stem(t) for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS]
        return toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype="float32")
        for row, text in enumerate(texts):
            for feat in self._features(text):
                h = int.from_bytes(hashlib.md5(feat.encode()).digest()[:8], "little")
                sign = 1.0 if (h >> 63) & 1 else -1.0
                out[row, h % self.dim] += sign
            out[row] = np.sign(out[row]) * np.log1p(np.abs(out[row]))  # sublinear tf
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (out / norms).astype("float32")


def get_embedder(choice: str = "auto") -> Embedder:
    """choice: 'auto' (MiniLM if installed, else hashing), 'minilm', or 'hash'."""
    if choice == "hash":
        return HashingEmbedder()
    try:
        return SentenceTransformerEmbedder()
    except Exception as exc:
        if choice == "minilm":
            raise
        log.warning(
            "sentence-transformers unavailable (%s). Falling back to the offline "
            "hashing embedder - install requirements.txt for MiniLM embeddings.",
            exc,
        )
        return HashingEmbedder()
