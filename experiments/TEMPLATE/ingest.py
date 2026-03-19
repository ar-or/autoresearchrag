#!/usr/bin/env python3
"""Experiment ingest template — copy and modify for custom ingestion.

If your hypothesis changes chunking, embedding serialization, or document
structure, create a custom ingest.py in your experiment folder.  Otherwise
you don't need this file — just use the default:

    uv run python evaluators/hotpotqa/ingest_hotpotqa.py

Usage:
    ES_INDEX=h<id> uv run python experiments/h<id>/ingest.py
"""

import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(PROJECT_ROOT / ".env")
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ingest import (  # noqa: E402
    chunk_text,
    ensure_ingest_ready,
    ingest_prepared_items,
    prepare_document,
    prepare_document_chunks,
)


def main():
    # ---- Replace everything below with your experiment's ingest logic ----
    #
    # Typical pattern: load data, chunk differently, then call the shared
    # pipeline.  See evaluators/hotpotqa/ingest_hotpotqa.py for a full
    # working example.
    #
    # Example:
    #   ensure_ingest_ready()
    #   items = load_your_data()
    #   total, errors = ingest_prepared_items(items, prepare_fn, total_items=len(items))
    #   print(f"Done. {total} chunks indexed ({errors} errors)")
    raise NotImplementedError(
        "Copy this template to experiments/h<id>/ingest.py and implement your ingest logic."
    )


if __name__ == "__main__":
    main()
