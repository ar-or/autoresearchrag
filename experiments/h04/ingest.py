#!/usr/bin/env python3
"""HotpotQA ingest for H04 structure-aware chunk boundaries."""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(PROJECT_ROOT / ".env")
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ingest import ensure_ingest_ready, ingest_prepared_items, prepare_document_chunks

DATA_PATH = PROJECT_ROOT / "evaluators" / "hotpotqa" / "data" / "hotpot_dev_distractor_v1.json"
CHUNK_SIZE = 512


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
    return [part.strip() for part in parts if part.strip()]


def _split_structural_units(text: str) -> list[str]:
    normalized = text.replace("\r\n", "\n")
    paragraphs = re.split(r"\n{2,}", normalized)
    units: list[str] = []
    for paragraph in paragraphs:
        stripped = paragraph.strip()
        if not stripped:
            continue
        lines = [line.strip() for line in stripped.splitlines() if line.strip()]
        units.extend(lines or [stripped])
    return units


def _window_chunk(text: str, size: int = CHUNK_SIZE, overlap: int = 64) -> list[str]:
    chunks: list[str] = []
    start = 0
    while start < len(text):
        chunk = text[start : start + size].strip()
        if chunk:
            chunks.append(chunk)
        start += max(size - overlap, 1)
    return chunks


def structure_aware_chunks(text: str) -> list[str]:
    units = _split_structural_units(text)
    if not units:
        return []

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    def flush() -> None:
        nonlocal current, current_len
        if current:
            chunks.append(" ".join(current).strip())
        current = []
        current_len = 0

    for unit in units:
        sentences = _split_sentences(unit) or [unit.strip()]
        for sentence in sentences:
            if len(sentence) > CHUNK_SIZE:
                flush()
                chunks.extend(_window_chunk(sentence))
                continue
            projected = current_len + len(sentence) + (1 if current else 0)
            if current and projected > CHUNK_SIZE:
                flush()
            current.append(sentence)
            current_len += len(sentence) + (1 if len(current) > 1 else 0)

    flush()
    return [chunk for chunk in chunks if chunk]


def _prepare_paragraph(item: tuple[str, str]):
    title, text = item
    doc_id = hashlib.sha256(title.encode()).hexdigest()[:16]
    return prepare_document_chunks(
        structure_aware_chunks(text),
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
