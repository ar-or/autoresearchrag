#!/usr/bin/env python3
"""Export FinQA documents as plain text files."""

import json
import os
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parent / "data" / "dev.json"


def format_table_text(table: list[list[str]]) -> str:
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
    parts = []

    pre = example.get("pre_text", [])
    if pre:
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


def as_files(output_dir: Path):
    if not DATA_PATH.exists():
        print(f"  Data not found at {DATA_PATH}, skipping")
        return

    with open(DATA_PATH) as f:
        data = json.load(f)

    written = 0
    skipped = 0
    for example in data:
        doc_id = example["id"].replace("/", "_")
        filepath = output_dir / f"{doc_id}.txt"
        if filepath.exists():
            skipped += 1
            continue
        text = build_document_text(example)
        filepath.write_text(text)
        written += 1

    print(f"  Written: {written}, Skipped: {skipped}")
