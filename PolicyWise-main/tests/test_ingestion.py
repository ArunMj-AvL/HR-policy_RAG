"""Run with:  pytest -q      (uses the offline hashing embedder, no downloads)"""

import json
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ingestion import VectorStore, chunk_document, extract_pdf, get_embedder, run_ingestion  # noqa: E402
from ingestion.chunking import chunk_page  # noqa: E402

PDF_DIR = ROOT / "data"


@pytest.fixture()
def pdf_dir(tmp_path):
    d = tmp_path / "pdfs"
    shutil.copytree(PDF_DIR, d)
    return d


def test_extraction_keeps_name_page_and_status():
    doc = extract_pdf(PDF_DIR / "G1_policy_1.pdf")
    assert doc.doc_name == "G1_policy_1.pdf"
    assert [p.page for p in doc.pages] == [1]
    assert (doc.label, doc.status, doc.effective_date) == ("Travel Policy v2.0", "current", "2026-10-01")

    old = extract_pdf(PDF_DIR / "G1_policy_4.pdf")
    assert (old.label, old.status, old.superseded_date) == ("Travel Policy v1.0", "superseded", "2025-01-01")


def test_every_chunk_has_doc_name_and_page():
    for pdf in PDF_DIR.glob("*.pdf"):
        chunks = chunk_document(extract_pdf(pdf))
        assert chunks
        for c in chunks:
            assert c.doc_name == pdf.name and c.page >= 1 and c.text.strip()
            assert c.char_count <= 400


def test_overlap_between_consecutive_chunks():
    page = extract_pdf(PDF_DIR / "G1_policy_1.pdf").pages[0].text
    chunks = chunk_page(page, chunk_size=400, overlap=100)
    assert len(chunks) >= 2
    for prev, nxt in zip(chunks, chunks[1:]):
        shared = {s for s, _ in prev} & {s for s, _ in nxt}
        assert shared, "consecutive chunks must share at least one sentence"


def test_no_overlap_when_zero():
    page = extract_pdf(PDF_DIR / "G1_policy_1.pdf").pages[0].text
    chunks = chunk_page(page, chunk_size=400, overlap=0)
    for prev, nxt in zip(chunks, chunks[1:]):
        assert not ({s for s, _ in prev} & {s for s, _ in nxt})


def test_bad_overlap_rejected():
    with pytest.raises(ValueError):
        chunk_page("Hello world. Foo bar.", chunk_size=50, overlap=50)


def test_expected_facts_survive_chunking_intact():
    """Facts needed by the 8 test questions must sit whole inside one chunk."""
    facts = {
        "G1_policy_1.pdf": ["INR 6000 per night", "15 calendar days after travel ends",
                            "not reimbursable unless approved by HR"],
        "G1_policy_2.pdf": ["Unused days do not carry forward", "Full-time employees"],
        "G1_policy_3.pdf": ["INR 25000 additionally require finance director approval"],
        "G1_policy_4.pdf": ["INR 4500 per night", "superseded"],
    }
    for name, needles in facts.items():
        texts = [c.text for c in chunk_document(extract_pdf(PDF_DIR / name))]
        for n in needles:
            assert any(n in t for t in texts), f"{n!r} was split across chunks in {name}"


def test_index_roundtrip_and_search(tmp_path, pdf_dir):
    idx = tmp_path / "index"
    summary = run_ingestion(pdf_dir, idx, embedder_choice="hash")
    assert summary["documents"] == 4 and summary["chunks_total"] == 8

    store = VectorStore.load(idx)
    emb = get_embedder("hash")
    hits = store.search(emb.encode(["What is the current Tier 1 hotel cap?"])[0], k=3)
    assert "G1_policy_1.pdf" in [c["doc_name"] for _, c in hits]
    assert all({"doc_name", "page", "status", "text"} <= set(c) for _, c in hits)


def test_reingest_does_not_duplicate_chunks(tmp_path, pdf_dir):
    idx = tmp_path / "index"
    first = run_ingestion(pdf_dir, idx, embedder_choice="hash")
    second = run_ingestion(pdf_dir, idx, embedder_choice="hash")
    assert second["chunks_total"] == first["chunks_total"]
    assert second["chunks_embedded_now"] == 0 and second["chunks_reused"] == first["chunks_total"]
    ids = [json.loads(l)["chunk_id"] for l in (idx / "chunks.jsonl").read_text().splitlines()]
    assert len(ids) == len(set(ids))


def test_same_pdf_uploaded_under_new_name_is_skipped(tmp_path, pdf_dir):
    idx = tmp_path / "index"
    base = run_ingestion(pdf_dir, idx, embedder_choice="hash")
    shutil.copy(pdf_dir / "G1_policy_1.pdf", pdf_dir / "Travel_copy.pdf")
    again = run_ingestion(pdf_dir, idx, embedder_choice="hash")
    assert again["chunks_total"] == base["chunks_total"]
    assert again["duplicate_files_skipped"] == ["Travel_copy.pdf"] or again["duplicate_files_skipped"] == ["G1_policy_1.pdf"]


def test_deleted_pdf_is_removed_from_index(tmp_path, pdf_dir):
    idx = tmp_path / "index"
    run_ingestion(pdf_dir, idx, embedder_choice="hash")
    (pdf_dir / "G1_policy_3.pdf").unlink()
    after = run_ingestion(pdf_dir, idx, embedder_choice="hash")
    names = {json.loads(l)["doc_name"] for l in (idx / "chunks.jsonl").read_text().splitlines()}
    assert "G1_policy_3.pdf" not in names and after["documents"] == 3
