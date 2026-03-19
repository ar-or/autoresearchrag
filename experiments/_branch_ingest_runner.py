#!/usr/bin/env python3
"""Run HotpotQA ingestion using a vendored experiment-local ingest implementation."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

EXPERIMENT_DIR = Path(__file__).absolute().parent
PROJECT_ROOT = EXPERIMENT_DIR.parent.parent
load_dotenv(PROJECT_ROOT / ".env")
sys.path.insert(0, str(PROJECT_ROOT))

DATA_PATH = PROJECT_ROOT / "evaluators" / "hotpotqa" / "data" / "hotpot_dev_distractor_v1.json"


def _load_ingest_impl():
    impl_path = EXPERIMENT_DIR / "ingest_impl.py"
    if not impl_path.exists():
        raise FileNotFoundError(f"Missing vendored ingest implementation at {impl_path}")

    spec = importlib.util.spec_from_file_location(
        f"experiments.{EXPERIMENT_DIR.name}.ingest_impl",
        impl_path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {impl_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _prepare_paragraph(prepare_document, item: tuple[str, str]):
    title, text = item
    doc_id = hashlib.sha256(title.encode()).hexdigest()[:16]
    return prepare_document(
        text=text,
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

    ingest_impl = _load_ingest_impl()

    print(f"Loading HotpotQA data from {DATA_PATH}")
    with open(DATA_PATH) as f:
        data = json.load(f)
    print(f"Loaded {len(data)} examples")

    paragraphs: dict[str, str] = {}
    for ex in data:
        for title, sentences in ex["context"]:
            if title not in paragraphs:
                paragraphs[title] = " ".join(sentences)

    print(f"Unique paragraphs: {len(paragraphs)}")

    items = list(paragraphs.items())
    ingest_impl.ensure_ingest_ready()
    total_chunks, total_errors = ingest_impl.ingest_prepared_items(
        items,
        lambda item: _prepare_paragraph(ingest_impl.prepare_document, item),
        total_items=len(items),
        progress_every=5000,
    )
    print(f"\nDone. Total chunks indexed: {total_chunks} ({total_errors} errors)")


if __name__ == "__main__":
    main()
