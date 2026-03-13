#!/usr/bin/env python3
"""Ingest HotpotQA context paragraphs into the Elasticsearch mtrag index.

Reads hotpot_dev_distractor_v1.json, deduplicates paragraphs by title,
and ingests via the shared ingest pipeline.

Usage:
  uv run python evaluators/hotpotqa/ingest_hotpotqa.py
"""

import json
import hashlib
import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(PROJECT_ROOT / ".env")
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ingest import chunk_text, ensure_index, index_chunks

DATA_PATH = Path(__file__).resolve().parent / "data" / "hotpot_dev_distractor_v1.json"


def main():
    if not DATA_PATH.exists():
        print(f"ERROR: Data not found at {DATA_PATH}")
        print("Run download_data.sh first.")
        sys.exit(1)

    print(f"Loading HotpotQA data from {DATA_PATH}")
    with open(DATA_PATH) as f:
        data = json.load(f)
    print(f"Loaded {len(data)} examples")

    # Deduplicate paragraphs by title
    paragraphs: dict[str, str] = {}
    for ex in data:
        for title, sentences in ex["context"]:
            if title not in paragraphs:
                paragraphs[title] = " ".join(sentences)

    print(f"Unique paragraphs: {len(paragraphs)}")

    ensure_index()

    total_chunks = 0
    total_errors = 0

    items = list(paragraphs.items())
    for i, (title, text) in enumerate(items):
        if not text.strip():
            continue

        doc_id = hashlib.sha256(title.encode()).hexdigest()[:16]
        chunks = chunk_text(text)
        if not chunks:
            continue

        ok, errs = index_chunks(chunks, title, f"hotpotqa:{title}", doc_id,
                                 doc_name=title, collection_name="mycollection2")
        total_chunks += ok
        total_errors += errs

        if (i + 1) % 5000 == 0 or i == 0:
            print(f"  [{i+1}/{len(items)}] {title}: {len(chunks)} chunks ({ok} ok, {errs} errors)")

    print(f"\nDone. Total chunks indexed: {total_chunks} ({total_errors} errors)")


if __name__ == "__main__":
    main()
