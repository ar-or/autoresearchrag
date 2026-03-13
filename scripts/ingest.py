#!/usr/bin/env python3
"""Ingest documents into the Elasticsearch mtrag index.

Accepts URLs or local files, chunks the text, embeds with text-embedding-3-small,
and indexes as dense vectors into Elasticsearch.

Usage:
  python scripts/ingest.py https://example.com/article
  python scripts/ingest.py /path/to/file.txt
  python scripts/ingest.py https://url1 /path/file2.md https://url3

Environment variables:
  ELASTIC_URL   - Elasticsearch URL (default: http://localhost:9200)
  ES_INDEX      - Index name (default: mtrag)
  CHUNK_SIZE    - Characters per chunk (default: 512)
  CHUNK_OVERLAP - Overlap between chunks (default: 64)
  EMBED_CACHE   - Path to embedding cache db (default: .embed_cache.db)
"""

import hashlib
import json
import os
import re
import sys
from pathlib import Path

import requests

from scripts.embedder import embed_batch, embed_batch_api, EMBED_DIMS, EMBED_MODEL
from scripts.embed_cache import get as cache_get, put as cache_put, put_many as cache_put_many, flush as cache_flush

ELASTIC_URL = os.environ.get("ELASTIC_URL", "http://localhost:9200")
ELASTIC_INDEX = os.environ.get("ES_INDEX", "mtrag")
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "512"))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "64"))


# ---------------------------------------------------------------------------
# Elasticsearch index
# ---------------------------------------------------------------------------


def ensure_index():
    """Create the index with dense_vector mapping if it doesn't exist or has no mapping."""
    mapping = {
        "mappings": {
            "properties": {
                "embedding": {"type": "dense_vector", "dims": EMBED_DIMS, "index": True, "similarity": "cosine"},
                "text": {"type": "text"},
                "title": {"type": "text"},
                "source": {"type": "keyword"},
                "document_id": {"type": "keyword"},
                "doc_name": {"type": "keyword"},
                "collection_name": {"type": "keyword"},
                "chunk_index": {"type": "integer"},
            }
        }
    }
    r = requests.head(f"{ELASTIC_URL}/{ELASTIC_INDEX}")
    if r.status_code == 404:
        r = requests.put(f"{ELASTIC_URL}/{ELASTIC_INDEX}", json=mapping)
        r.raise_for_status()
        print(f"Created index '{ELASTIC_INDEX}'")
        return

    r = requests.get(f"{ELASTIC_URL}/{ELASTIC_INDEX}/_mapping")
    r.raise_for_status()
    props = r.json().get(ELASTIC_INDEX, {}).get("mappings", {}).get("properties", {})
    if "embedding" not in props:
        requests.delete(f"{ELASTIC_URL}/{ELASTIC_INDEX}")
        r = requests.put(f"{ELASTIC_URL}/{ELASTIC_INDEX}", json=mapping)
        r.raise_for_status()
        print(f"Recreated index '{ELASTIC_INDEX}' with dense_vector mapping")


# ---------------------------------------------------------------------------
# Fetching / reading
# ---------------------------------------------------------------------------


def fetch_url(url: str) -> tuple[str, str]:
    """Fetch a URL and return (title, text)."""
    r = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    html = r.text
    title_match = re.search(r"<title[^>]*>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
    title = title_match.group(1).strip() if title_match else url
    text = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return title, text


def read_file(path: str) -> tuple[str, str]:
    """Read a local file and return (title, text)."""
    p = Path(path)
    text = p.read_text()
    title = p.stem.replace("_", " ").replace("-", " ").title()
    return title, text


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk.strip())
        start = end - overlap
    return chunks


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------


def index_chunks(
    chunks: list[str], title: str, source: str, doc_id: str,
    doc_name: str = "", collection_name: str = "",
):
    """Index chunks into Elasticsearch (embeds via realtime API)."""
    all_embeddings = []
    for i in range(0, len(chunks), 100):
        batch = chunks[i : i + 100]
        all_embeddings.extend(embed_batch(batch))

    return _index_with_embeddings(chunks, all_embeddings, title, source, doc_id, doc_name, collection_name)


def index_chunks_preembedded(
    chunks: list[str], embeddings: list[list[float]], title: str, source: str, doc_id: str,
    doc_name: str = "", collection_name: str = "",
):
    """Index chunks into Elasticsearch with pre-computed embeddings."""
    return _index_with_embeddings(chunks, embeddings, title, source, doc_id, doc_name, collection_name)


def _index_with_embeddings(
    chunks: list[str], embeddings: list[list[float]], title: str, source: str, doc_id: str,
    doc_name: str = "", collection_name: str = "",
):
    """Bulk-index chunks with their embeddings into Elasticsearch."""
    bulk_body = []
    for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
        chunk_id = f"{doc_id}_{i}"
        bulk_body.append(json.dumps({"index": {"_index": ELASTIC_INDEX, "_id": chunk_id}}))
        bulk_body.append(
            json.dumps(
                {
                    "text": chunk,
                    "title": title,
                    "source": source,
                    "document_id": doc_id,
                    "doc_name": doc_name,
                    "collection_name": collection_name,
                    "chunk_index": i,
                    "embedding": emb,
                }
            )
        )
    bulk_payload = "\n".join(bulk_body) + "\n"
    r = requests.post(
        f"{ELASTIC_URL}/_bulk",
        data=bulk_payload,
        headers={"Content-Type": "application/x-ndjson"},
    )
    r.raise_for_status()
    result = r.json()
    errors = sum(1 for item in result.get("items", []) if item.get("index", {}).get("error"))
    return len(chunks) - errors, errors


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def ingest(source: str, doc_name: str = "", collection_name: str = ""):
    """Ingest a single URL or file."""
    is_url = source.startswith("http://") or source.startswith("https://")
    print(f"\n{'URL' if is_url else 'File'}: {source}")

    if is_url:
        title, text = fetch_url(source)
    else:
        title, text = read_file(source)

    doc_id = hashlib.sha256(source.encode()).hexdigest()[:16]
    if not doc_name:
        doc_name = title
    chunks = chunk_text(text)
    print(f"  Title: {title}")
    print(f"  Text length: {len(text)} chars")
    print(f"  Chunks: {len(chunks)}")

    ok, errs = index_chunks(chunks, title, source, doc_id, doc_name, collection_name)
    print(f"  Indexed: {ok} chunks ({errs} errors)")
    return ok


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/ingest.py <url_or_file> [url_or_file ...]")
        sys.exit(1)

    ensure_index()
    total = 0
    for source in sys.argv[1:]:
        try:
            total += ingest(source)
        except Exception as e:
            print(f"  ERROR: {e}")
    print(f"\nDone. Total chunks indexed: {total}")


if __name__ == "__main__":
    main()
