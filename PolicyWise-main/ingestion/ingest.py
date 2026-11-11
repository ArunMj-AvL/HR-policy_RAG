#!/usr/bin/env python
"""Command line entry point:  python ingest.py [options]"""

import argparse
import json
import logging

from ingestion import run_ingestion
from config import DATA_DIR


def main() -> None:
    ap = argparse.ArgumentParser(description="Ingest HR policy PDFs into a FAISS index.")
    ap.add_argument("--pdf-dir",default=str(DATA_DIR),help="Folder containing policy PDFs")
    ap.add_argument("--index-dir", default="index", help="where the index is written")
    ap.add_argument("--chunk-size", type=int, default=400, help="max characters per chunk")
    ap.add_argument("--overlap", type=int, default=100, help="characters repeated between chunks")
    ap.add_argument("--embedder", choices=["auto", "minilm", "hash"], default="auto",
                    help="auto = MiniLM if installed, else offline hashing fallback")
    ap.add_argument("--rebuild", action="store_true", help="ignore the existing index and re-embed all")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    summary = run_ingestion(args.pdf_dir, args.index_dir, args.chunk_size, args.overlap,
                            args.embedder, args.rebuild)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
