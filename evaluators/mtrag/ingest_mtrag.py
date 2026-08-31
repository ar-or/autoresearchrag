#!/usr/bin/env python3
"""Ingest MT-RAG passage-level corpora into the Elasticsearch mtrag index.

Reads passage-level JSONL files (zipped) for all four domains
(clapnq, cloud, fiqa, govt) and ingests via the shared ingest pipeline.

Usage:
  uv run python evaluators/mtrag/ingest_mtrag.py
  uv run python evaluators/mtrag/ingest_mtrag.py clapnq fiqa   # specific domains only
"""

import hashlib
import json
import os
import sys
import zipfile
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(PROJECT_ROOT / ".env")
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ingest import (
    chunk_text,
    ensure_ingest_ready,
    ingest_prepared_items,
    prepare_document_chunks,
)

CORPORA_DIR = Path(__file__).resolve().parent / "data" / "mt-rag-benchmark" / "corpora" / "passage_level"
ALL_DOMAINS = ["clapnq", "cloud", "fiqa", "govt"]
MTRAG_INDEX_COLLECTION_NAME = os.environ.get("MTRAG_INDEX_COLLECTION_NAME", "mycollection3")
MTRAG_MAX_PASSAGE_CHARS = int(os.environ.get("MTRAG_MAX_PASSAGE_CHARS", "8000"))
MTRAG_START_LINE = int(os.environ.get("MTRAG_START_LINE", "1"))
MTRAG_END_LINE = int(os.environ.get("MTRAG_END_LINE", "0"))


def _prepare_passage(args: tuple[str, dict]):
    domain, passage = args
    text = passage.get("text", "").strip()
    if not text:
        return None

    title = passage.get("title", "")
    passage_id = passage.get("_id", passage.get("id", ""))
    doc_id = hashlib.sha256(passage_id.encode()).hexdigest()[:16]
    chunks = [text]
    if len(text) > MTRAG_MAX_PASSAGE_CHARS:
        # A small number of "passage" records are still too large for one embedding call.
        # Split only those outliers while keeping the same document_id so qrel matching still works.
        chunks = chunk_text(text, chunk_size=MTRAG_MAX_PASSAGE_CHARS, overlap=0)
    # passage_level corpora are already benchmark passage units. Keep them one-to-one by default,
    # and only split rare oversized outliers while preserving the same document_id.
    return prepare_document_chunks(
        chunks=chunks,
        hash_key=doc_id,
        fields={
            "title": title,
            "source": f"mtrag:{domain}:{passage_id}",
            "document_id": doc_id,
            "doc_name": title or passage_id,
            "collection_name": MTRAG_INDEX_COLLECTION_NAME,
        },
    )


def ingest_domain(domain: str):
    """Ingest a single MT-RAG domain corpus."""
    zip_path = CORPORA_DIR / f"{domain}.jsonl.zip"
    if not zip_path.exists():
        print(f"  WARNING: {zip_path} not found, skipping")
        return 0, 0

    total_chunks = 0
    total_errors = 0

    with zipfile.ZipFile(zip_path) as zf:
        jsonl_name = zf.namelist()[0]
        with zf.open(jsonl_name) as f:
            items = (
                (domain, json.loads(line))
                for line_number, line in enumerate(f, start=1)
                if line_number >= MTRAG_START_LINE
                and (MTRAG_END_LINE <= 0 or line_number <= MTRAG_END_LINE)
            )
            total_chunks, total_errors = ingest_prepared_items(
                items,
                _prepare_passage,
                progress_every=10000,
                progress_prefix="    ",
            )

    return total_chunks, total_errors


def main():
    domains = sys.argv[1:] if len(sys.argv) > 1 else ALL_DOMAINS
    for d in domains:
        if d not in ALL_DOMAINS:
            print(f"ERROR: Unknown domain '{d}'. Choose from: {ALL_DOMAINS}")
            sys.exit(1)

    ensure_ingest_ready()

    grand_total = 0
    grand_errors = 0

    for domain in domains:
        print(f"\nIngesting domain: {domain}")
        if MTRAG_START_LINE > 1 or MTRAG_END_LINE > 0:
            end_label = "end" if MTRAG_END_LINE <= 0 else str(MTRAG_END_LINE)
            print(f"  Line range: {MTRAG_START_LINE}..{end_label}")
        ok, errs = ingest_domain(domain)
        grand_total += ok
        grand_errors += errs
        print(f"  {domain}: {ok} chunks ({errs} errors)")

    print(f"\nDone. Total chunks indexed: {grand_total} ({grand_errors} errors)")


if __name__ == "__main__":
    main()
