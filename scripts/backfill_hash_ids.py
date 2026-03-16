#!/usr/bin/env python3
"""Backfill hash_id on existing Elasticsearch documents and remove duplicates.

Usage:
  uv run python scripts/backfill_hash_ids.py
"""

import json
import os
import sys
from pathlib import Path

import requests
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ingest import (
    ELASTIC_INDEX,
    ELASTIC_URL,
    build_chunk_hash_id,
    build_document_hash_key,
    ensure_hash_id_mapping,
)

SCROLL_KEEPALIVE = os.environ.get("HASH_BACKFILL_SCROLL", "2m")
BATCH_SIZE = int(os.environ.get("HASH_BACKFILL_BATCH_SIZE", "500"))


def _bulk(actions: list[str]) -> int:
    if not actions:
        return 0

    payload = "\n".join(actions) + "\n"
    r = requests.post(
        f"{ELASTIC_URL}/_bulk",
        data=payload,
        headers={"Content-Type": "application/x-ndjson"},
    )
    r.raise_for_status()
    result = r.json()
    errors = [item for item in result.get("items", []) if any(v.get("error") for v in item.values())]
    if errors:
        raise RuntimeError(f"Bulk request returned {len(errors)} errors")
    return len(result.get("items", []))


def _flush(update_actions: list[str], delete_actions: list[str]) -> tuple[int, int]:
    updated = 0
    deleted = 0
    if update_actions:
        updated = _bulk(update_actions)
        update_actions.clear()
    if delete_actions:
        deleted = _bulk(delete_actions)
        delete_actions.clear()
    return updated, deleted


def _initial_search() -> dict:
    r = requests.post(
        f"{ELASTIC_URL}/{ELASTIC_INDEX}/_search",
        params={"scroll": SCROLL_KEEPALIVE},
        json={
            "size": BATCH_SIZE,
            "sort": ["_doc"],
            "_source": [
                "document_id",
                "collection_name",
                "source",
                "title",
                "doc_name",
                "text",
                "hash_id",
            ],
            "query": {"match_all": {}},
        },
    )
    r.raise_for_status()
    return r.json()


def _next_page(scroll_id: str) -> dict:
    r = requests.post(
        f"{ELASTIC_URL}/_search/scroll",
        json={"scroll": SCROLL_KEEPALIVE, "scroll_id": scroll_id},
    )
    r.raise_for_status()
    return r.json()


def _clear_scroll(scroll_id: str):
    requests.delete(f"{ELASTIC_URL}/_search/scroll", json={"scroll_id": [scroll_id]})


def main():
    ensure_hash_id_mapping()

    page = _initial_search()
    scroll_id = page.get("_scroll_id")
    seen_hash_ids: set[str] = set()
    update_actions: list[str] = []
    delete_actions: list[str] = []
    scanned = 0
    updated = 0
    deleted = 0

    try:
        while True:
            hits = page.get("hits", {}).get("hits", [])
            if not hits:
                break

            for hit in hits:
                scanned += 1
                source = hit.get("_source", {})
                text = source.get("text", "")
                if not text.strip():
                    continue

                document_key = build_document_hash_key(
                    document_id=source.get("document_id", ""),
                    collection_name=source.get("collection_name", ""),
                    source=source.get("source", ""),
                    title=source.get("title", ""),
                    fallback_id=hit.get("_id", ""),
                )
                hash_id = build_chunk_hash_id(document_key, text)

                if hash_id in seen_hash_ids:
                    delete_actions.append(
                        json.dumps({"delete": {"_index": ELASTIC_INDEX, "_id": hit["_id"]}})
                    )
                    continue

                seen_hash_ids.add(hash_id)
                if source.get("hash_id") == hash_id:
                    continue

                update_actions.append(
                    json.dumps({"update": {"_index": ELASTIC_INDEX, "_id": hit["_id"]}})
                )
                update_actions.append(json.dumps({"doc": {"hash_id": hash_id}}))

                if len(update_actions) >= BATCH_SIZE * 2 or len(delete_actions) >= BATCH_SIZE:
                    flushed_updates, flushed_deletes = _flush(update_actions, delete_actions)
                    updated += flushed_updates
                    deleted += flushed_deletes

            page = _next_page(scroll_id)
            scroll_id = page.get("_scroll_id", scroll_id)

        flushed_updates, flushed_deletes = _flush(update_actions, delete_actions)
        updated += flushed_updates
        deleted += flushed_deletes
    finally:
        if scroll_id:
            _clear_scroll(scroll_id)

    requests.post(f"{ELASTIC_URL}/{ELASTIC_INDEX}/_refresh").raise_for_status()

    print(f"Scanned: {scanned}")
    print(f"Updated hash_id: {updated}")
    print(f"Deleted duplicates: {deleted}")


if __name__ == "__main__":
    main()
