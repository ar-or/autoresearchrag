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
  ELASTIC_INDEX - Index name (default: mtrag)
  CHUNK_SIZE    - Characters per chunk (default: 512)
  CHUNK_OVERLAP - Overlap between chunks (default: 64)
"""

import hashlib
import json
import os
import re
import sys
import uuid
from pathlib import Path
from textwrap import dedent

import requests
from openai import OpenAI

ELASTIC_URL = os.environ.get("ELASTIC_URL", "http://localhost:9200")
ELASTIC_INDEX = os.environ.get("ELASTIC_INDEX", "mtrag")
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "512"))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "64"))
EMBED_MODEL = "text-embedding-3-small"
EMBED_DIMS = 1536

openai = OpenAI()


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
                "chunk_index": {"type": "integer"},
            }
        }
    }
    # Check if index exists
    r = requests.head(f"{ELASTIC_URL}/{ELASTIC_INDEX}")
    if r.status_code == 404:
        r = requests.put(f"{ELASTIC_URL}/{ELASTIC_INDEX}", json=mapping)
        r.raise_for_status()
        print(f"Created index '{ELASTIC_INDEX}'")
        return

    # Index exists — check if it has the embedding field
    r = requests.get(f"{ELASTIC_URL}/{ELASTIC_INDEX}/_mapping")
    r.raise_for_status()
    props = r.json().get(ELASTIC_INDEX, {}).get("mappings", {}).get("properties", {})
    if "embedding" not in props:
        # Delete and recreate with proper mapping
        requests.delete(f"{ELASTIC_URL}/{ELASTIC_INDEX}")
        r = requests.put(f"{ELASTIC_URL}/{ELASTIC_INDEX}", json=mapping)
        r.raise_for_status()
        print(f"Recreated index '{ELASTIC_INDEX}' with dense_vector mapping")


def fetch_url(url: str) -> tuple[str, str]:
    """Fetch a URL and return (title, text)."""
    r = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    html = r.text
    # Extract title
    title_match = re.search(r"<title[^>]*>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
    title = title_match.group(1).strip() if title_match else url
    # Strip HTML tags for plain text
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


def embed_batch(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts."""
    resp = openai.embeddings.create(input=texts, model=EMBED_MODEL)
    return [d.embedding for d in resp.data]


def index_chunks(chunks: list[str], title: str, source: str, doc_id: str):
    """Index chunks into Elasticsearch."""
    # Embed in batches of 100
    all_embeddings = []
    for i in range(0, len(chunks), 100):
        batch = chunks[i : i + 100]
        all_embeddings.extend(embed_batch(batch))

    # Bulk index
    bulk_body = []
    for i, (chunk, emb) in enumerate(zip(chunks, all_embeddings)):
        chunk_id = f"{doc_id}_{i}"
        bulk_body.append(json.dumps({"index": {"_index": ELASTIC_INDEX, "_id": chunk_id}}))
        bulk_body.append(
            json.dumps(
                {
                    "text": chunk,
                    "title": title,
                    "source": source,
                    "document_id": doc_id,
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


def ingest(source: str):
    """Ingest a single URL or file."""
    is_url = source.startswith("http://") or source.startswith("https://")
    print(f"\n{'URL' if is_url else 'File'}: {source}")

    if is_url:
        title, text = fetch_url(source)
    else:
        title, text = read_file(source)

    doc_id = hashlib.sha256(source.encode()).hexdigest()[:16]
    chunks = chunk_text(text)
    print(f"  Title: {title}")
    print(f"  Text length: {len(text)} chars")
    print(f"  Chunks: {len(chunks)}")

    ok, errs = index_chunks(chunks, title, source, doc_id)
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
