#!/usr/bin/env python3
"""HotpotQA ingest for H56 E5-style prefixed dense retrieval proxy."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(PROJECT_ROOT / ".env")
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ingest import ensure_ingest_ready, ingest_prepared_items, prepare_document

DATA_PATH = PROJECT_ROOT / "evaluators" / "hotpotqa" / "data" / "hotpot_dev_distractor_v1.json"
EVAL_N = int(os.environ.get("N", "30"))
def _prepare_paragraph(item: tuple[str, str]):
    title, text = item
    doc_id = hashlib.sha256(title.encode()).hexdigest()[:16]
    prefixed = f"passage: title: {title}\npassage: {text}"
    return prepare_document(
        prefixed,
        hash_key=doc_id,
        fields={
            "title": title,
            "source": f"hotpotqa:{title}",
            "document_id": doc_id,
            "doc_name": title,
            "collection_name": "mycollection2",
        },
    )


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"HotpotQA data not found at {DATA_PATH}")

    print(f"Loading HotpotQA data from {DATA_PATH}")
    with open(DATA_PATH) as f:
        data = json.load(f)
    data = data[:EVAL_N]

    paragraphs: dict[str, str] = {}
    for ex in data:
        for title, sentences in ex["context"]:
            if title not in paragraphs:
                paragraphs[title] = " ".join(sentences)

    items = list(paragraphs.items())
    ensure_ingest_ready()
    total_chunks, total_errors = ingest_prepared_items(
        items,
        _prepare_paragraph,
        total_items=len(items),
        progress_every=5000,
    )
    print(f"\nDone. Total chunks indexed: {total_chunks} ({total_errors} errors)")


if __name__ == "__main__":
    main()
