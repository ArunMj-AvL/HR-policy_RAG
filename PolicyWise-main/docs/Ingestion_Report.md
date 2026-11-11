# Ingestion & Indexing – Status Report (Group 1, PolicyWise)

**Owner:** <your name>  **Role:** PDF ingestion / indexing  **Date:** 2026-11-05

## 1. Task checklist

| Assigned task | Status | Where |
|---|---|---|
| Extract text from all 4 HR policy PDFs | Done | `ingestion/extract.py` |
| Chunking with overlap | Done (400 chars, 100 overlap, sentence-aware) | `ingestion/chunking.py` |
| Preserve document names and page numbers | Done – on every chunk | `chunks.jsonl` fields `doc_name`, `page` |
| Generate embeddings and store in FAISS | Code done (MiniLM + `IndexFlatIP`) | `embeddings.py`, `index_store.py` |
| Push code to GitHub | Repo is ready; push needs my GitHub login | see README → *Push to GitHub* |

## 2. Result on the supplied data

* 4 PDFs, 4 pages → **8 chunks**, each with document, version, page, section, and `current`/`superseded` status.
* Travel Policy v1.0 (G1_policy_4) is tagged `superseded`; the other three are `current`.
* Facts needed by the 8 test questions (6000/night, 15 days, 25000 approval, 4500 old cap, “do not carry forward”,
  HR-approved cancellation, full-time-only) each sit whole inside a single chunk – checked by an automated test.
* Re-running ingestion: 8 chunks reused, 0 re-embedded, 0 duplicates. Same PDF re-uploaded under a new name is skipped.
* `pytest`: 10 / 10 pass.
* Sanity retrieval (`scripts/sanity_check.py`): the expected source PDF is in the top 3 for 7 / 7 questions that have a
  source. Caveat: the corpus is only 8 chunks, so top-3 is an easy bar; the real Recall@3 evaluation belongs to the
  retrieval/evaluation work. For the “parking fee” question nothing relevant exists, so the answer layer must abstain.

## 3. What was and was not verified (please read)

* Verified here: extraction, chunking, overlap, metadata, de-duplication, save/load, search, tests.
* **Not executed in my build environment:** the real `faiss-cpu` and `sentence-transformers` (MiniLM) packages could not be
  installed there (no network). The pipeline ran with the built-in offline hashing embedder, and the FAISS code path
  was exercised only against a minimal stand-in module. On a normal machine run `pip install -r requirements.txt`, then
  `python ingest.py` and `python scripts/sanity_check.py` to confirm the MiniLM + FAISS path (expect
  `"faiss_used": true` and `"embedder": "sentence-transformers/all-MiniLM-L6-v2"` in the output).
* The index must be rebuilt with MiniLM before it is used by teammates; `index/` is git-ignored on purpose.

## 4. Known limitations

Text PDFs only (no OCR); header format assumed for status detection; regex sentence splitting; exact (flat) search.

## 5. Hand-off

Retrieval/grounding teammate: use `VectorStore.load("index")` and the `status` field to exclude superseded policies.
UI/testing teammate: show `doc_name`, `page`, and the chunk `text` as the source excerpt.
