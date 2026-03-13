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
import sys
import zipfile
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(PROJECT_ROOT / ".env")
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ingest import chunk_text, ensure_index, index_chunks

CORPORA_DIR = Path(__file__).resolve().parent / "data" / "mt-rag-benchmark" / "corpora" / "passage_level"
ALL_DOMAINS = ["clapnq", "cloud", "fiqa", "govt"]


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
            batch_texts = []
            batch_meta = []

            for line_num, line in enumerate(f):
                passage = json.loads(line)
                text = passage.get("text", "").strip()
                if not text:
                    continue

                title = passage.get("title", "")
                passage_id = passage.get("_id", passage.get("id", ""))
                doc_id = hashlib.sha256(passage_id.encode()).hexdigest()[:16]

                chunks = chunk_text(text)
                if not chunks:
                    continue

                ok, errs = index_chunks(chunks, title, f"mtrag:{domain}:{passage_id}", doc_id,
                                         doc_name=title or passage_id, collection_name="mycollection3")
                total_chunks += ok
                total_errors += errs

                if (line_num + 1) % 10000 == 0:
                    print(f"    [{line_num+1}] {total_chunks} chunks indexed so far ({total_errors} errors)")

    return total_chunks, total_errors


def main():
    domains = sys.argv[1:] if len(sys.argv) > 1 else ALL_DOMAINS
    for d in domains:
        if d not in ALL_DOMAINS:
            print(f"ERROR: Unknown domain '{d}'. Choose from: {ALL_DOMAINS}")
            sys.exit(1)

    ensure_index()

    grand_total = 0
    grand_errors = 0

    for domain in domains:
        print(f"\nIngesting domain: {domain}")
        ok, errs = ingest_domain(domain)
        grand_total += ok
        grand_errors += errs
        print(f"  {domain}: {ok} chunks ({errs} errors)")

    print(f"\nDone. Total chunks indexed: {grand_total} ({grand_errors} errors)")


if __name__ == "__main__":
    main()
