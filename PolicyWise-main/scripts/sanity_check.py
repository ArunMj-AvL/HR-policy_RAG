#!/usr/bin/env python
"""Quick retrieval sanity check of the finished index (NOT the final retrieval layer).

For each question in data/G1_questions.jsonl it shows the top-3 chunks and checks
whether the expected source PDF appears in them (a simple Recall@3).
Usage:  python scripts/sanity_check.py [--index-dir index] [--query "free text"]
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ingestion import VectorStore, get_embedder  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-dir", default="index")
    ap.add_argument("--questions", default="evaluation/test_questions.json")
    ap.add_argument("--query", help="run one free-text query instead of the question file")
    ap.add_argument("-k", type=int, default=3)
    args = ap.parse_args()

    store = VectorStore.load(args.index_dir)
    embedder = get_embedder("hash" if store.meta.get("embedder", "").startswith("hashing") else "minilm")
    if embedder.name != store.meta.get("embedder"):
        sys.exit(f"Index was built with {store.meta.get('embedder')} but loaded {embedder.name}")

    def show(q: str):
        hits = store.search(embedder.encode([q])[0], args.k)
        for rank, (score, c) in enumerate(hits, 1):
            snippet = " ".join(c["text"].split())[:70]
            print(f"   {rank}. {score:.3f}  {c['doc_name']} p{c['page']} [{c['status']}] {snippet}...")
        return hits

    if args.query:
        show(args.query)
        return

    questions_path = Path(args.questions)

    if not questions_path.exists():
        sys.exit(f"Questions file not found: {questions_path}")

    raw = questions_path.read_text(encoding="utf-8")

    if questions_path.suffix == ".jsonl":
        items = [
            json.loads(line)
            for line in raw.splitlines()
            if line.strip()
        ]
    else:
        items = json.loads(raw)

    ok = total = 0

    for item in items:
        print(f"\nQ: {item['question']}")

        hits = show(item["question"])

        expected_source = (
            item.get("source")
            or item.get("expected_source")
            or "none"
        )

        if expected_source == "none":
            print("   -> Requires answer-layer abstention.")
            continue

        total += 1

        found = any(
            c["doc_name"] == expected_source
            for _, c in hits
        )

        ok += found

        print(
            f"   -> Source in top-{args.k}: "
            f"{'YES' if found else 'NO'}"
        )

    print(f"\nRecall@{args.k}: {ok}/{total}")


if __name__ == "__main__":
    main()
