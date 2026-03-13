#!/usr/bin/env python3
"""Ingest FinQA documents into the Elasticsearch mtrag index.

Reads FinQA dev.json, builds a text representation of each financial report
(pre_text + table + post_text), and ingests via the shared ingest pipeline.

Usage:
  uv run python evaluators/finqa/ingest_finqa.py
"""

import json
import sys
from pathlib import Path

from dotenv import load_dotenv

# Add project root so we can import from scripts/
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Load .env from project root before importing ingest (which creates OpenAI client at module level)
load_dotenv(PROJECT_ROOT / ".env")

sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ingest import chunk_text, embed_batch, ensure_index, index_chunks

DATA_PATH = Path(__file__).resolve().parent / "data" / "dev.json"


def format_table_text(table: list[list[str]]) -> str:
    """Render a table as readable text."""
    if not table:
        return ""
    lines = []
    header = table[0]
    for row in table[1:]:
        parts = []
        for col, val in zip(header, row):
            parts.append(f"{col}: {val}")
        lines.append(" | ".join(parts))
    return "\n".join(lines)


def build_document_text(example: dict) -> str:
    """Build a single text document from a FinQA example."""
    parts = []

    pre = example.get("pre_text", [])
    if pre:
        # Filter out single-char noise entries
        text = " ".join(s for s in pre if len(s.strip()) > 2)
        if text.strip():
            parts.append(text.strip())

    table = example.get("table", [])
    if table:
        parts.append(format_table_text(table))

    post = example.get("post_text", [])
    if post:
        text = " ".join(s for s in post if len(s.strip()) > 2)
        if text.strip():
            parts.append(text.strip())

    return "\n\n".join(parts)


def main():
    if not DATA_PATH.exists():
        print(f"ERROR: Data not found at {DATA_PATH}")
        print("Run download_data.sh first.")
        sys.exit(1)

    print(f"Loading FinQA data from {DATA_PATH}")
    with open(DATA_PATH) as f:
        data = json.load(f)
    print(f"Loaded {len(data)} examples")

    ensure_index()

    total_chunks = 0
    total_errors = 0

    for i, ex in enumerate(data):
        doc_id = ex["id"]
        filename = ex.get("filename", doc_id)
        text = build_document_text(ex)

        if not text.strip():
            continue

        title = filename
        chunks = chunk_text(text)

        if not chunks:
            continue

        ok, errs = index_chunks(chunks, title, f"finqa:{filename}", doc_id.replace("/", "_"),
                                 doc_name=filename, collection_name="mycollection1")
        total_chunks += ok
        total_errors += errs

        if (i + 1) % 50 == 0 or i == 0:
            print(f"  [{i+1}/{len(data)}] {filename}: {len(chunks)} chunks ({ok} ok, {errs} errors)")

    print(f"\nDone. Total chunks indexed: {total_chunks} ({total_errors} errors)")


if __name__ == "__main__":
    main()
