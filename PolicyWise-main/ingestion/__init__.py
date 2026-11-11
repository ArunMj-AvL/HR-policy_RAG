"""PolicyWise ingestion package: PDF -> chunks -> embeddings -> FAISS index."""

from .extract import PageText, PolicyDocument, extract_pdf
from .chunking import Chunk, chunk_document
from .embeddings import get_embedder
from .index_store import VectorStore
from .pipeline import run_ingestion

__all__ = [
    "PageText",
    "PolicyDocument",
    "extract_pdf",
    "Chunk",
    "chunk_document",
    "get_embedder",
    "VectorStore",
    "run_ingestion",
]
