#!/usr/bin/env python3
"""HotpotQA ingest for H03 structure-preserving paragraph serialization."""

from __future__ import annotations

import hashlib
import json
import os
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
EVAL_N = int(os.environ.get("N", "30"))


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
    return [part.strip() for part in parts if part.strip()]


def _serialize_sentence_group(title: str, sentences: list[str], offset: int) -> str:
    lines = [f"TITLE: {title}", "PARAGRAPH_STRUCTURE:"]
    for index, sentence in enumerate(sentences, start=offset):
        lines.append(f"SENTENCE_{index}: {sentence}")
    return "\n".join(lines)


def structure_preserving_chunks(title: str, text: str) -> list[str]:
    sentences = _split_sentences(text)
    if not sentences:
        return []

    chunks: list[str] = []
    offset = 0
    while offset < len(sentences):
        window = sentences[offset : offset + 4]
        serialized = _serialize_sentence_group(title, window, offset)
        if len(serialized) <= CHUNK_SIZE:
            chunks.append(serialized)
            offset += 4
            continue

        trimmed: list[str] = []
        for sentence in window:
            candidate = _serialize_sentence_group(title, trimmed + [sentence], offset)
            if trimmed and len(candidate) > CHUNK_SIZE:
                break
            trimmed.append(sentence[: CHUNK_SIZE // 2])
        if trimmed:
            chunks.append(_serialize_sentence_group(title, trimmed, offset))
        offset += max(len(trimmed), 1)
    return chunks


def _prepare_paragraph(item: tuple[str, str]):
    title, text = item
    doc_id = hashlib.sha256(title.encode()).hexdigest()[:16]
    return prepare_document_chunks(
        structure_preserving_chunks(title, text),
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
