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
from collections.abc import Callable, Iterable, Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor
from itertools import islice
from pathlib import Path
from typing import Any

import requests

from scripts.embedder import EMBED_DIMS, embed_batch

ELASTIC_URL = os.environ.get("ELASTIC_URL", "http://localhost:9200")
ELASTIC_INDEX = os.environ.get("ES_INDEX", "mtrag")
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "512"))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "64"))
HASH_LOOKUP_BATCH_SIZE = int(os.environ.get("HASH_LOOKUP_BATCH_SIZE", "1000"))
EMBED_BATCH_SIZE = int(os.environ.get("EMBED_BATCH_SIZE", "100"))
INDEX_BULK_BATCH_SIZE = int(os.environ.get("INDEX_BULK_BATCH_SIZE", "1000"))
INGEST_DOC_BATCH_SIZE = int(os.environ.get("INGEST_DOC_BATCH_SIZE", "128"))
INGEST_PREPARE_WORKERS = int(
    os.environ.get("INGEST_PREPARE_WORKERS", str(min(32, os.cpu_count() or 4)))
)
RESERVED_INDEX_FIELDS = {"text", "chunk_index", "hash_id", "embedding"}


def ensure_hash_id_mapping():
    """Add the hash_id field to the existing index mapping if needed."""
    r = requests.get(f"{ELASTIC_URL}/{ELASTIC_INDEX}/_mapping")
    r.raise_for_status()
    props = r.json().get(ELASTIC_INDEX, {}).get("mappings", {}).get("properties", {})
    if "hash_id" in props:
        return

    r = requests.put(
        f"{ELASTIC_URL}/{ELASTIC_INDEX}/_mapping",
        json={"properties": {"hash_id": {"type": "keyword"}}},
    )
    r.raise_for_status()


def ensure_sentence_chunk_mapping():
    """Add sentence-aware chunk metadata fields if needed."""
    r = requests.get(f"{ELASTIC_URL}/{ELASTIC_INDEX}/_mapping")
    r.raise_for_status()
    props = r.json().get(ELASTIC_INDEX, {}).get("mappings", {}).get("properties", {})
    missing: dict[str, dict[str, str]] = {}
    if "sentence_texts" not in props:
        missing["sentence_texts"] = {"type": "text"}
    if "sentence_count" not in props:
        missing["sentence_count"] = {"type": "integer"}
    if not missing:
        return

    r = requests.put(f"{ELASTIC_URL}/{ELASTIC_INDEX}/_mapping", json={"properties": missing})
    r.raise_for_status()


def ensure_hash_ids_ready():
    """Fail fast if the index still contains legacy documents without hash_id."""
    r = requests.post(
        f"{ELASTIC_URL}/{ELASTIC_INDEX}/_count",
        json={"query": {"bool": {"must_not": [{"exists": {"field": "hash_id"}}]}}},
    )
    r.raise_for_status()
    missing = int(r.json().get("count", 0))
    if missing == 0:
        return

    raise RuntimeError(
        f"Index '{ELASTIC_INDEX}' still has {missing} documents without hash_id. "
        "Run `uv run python scripts/backfill_hash_ids.py` before ingesting more data."
    )


def ensure_ingest_ready():
    """Ensure the target index exists and is ready for deduplicated ingestion."""
    ensure_index()
    ensure_hash_ids_ready()


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
                "hash_id": {"type": "keyword"},
                "sentence_texts": {"type": "text"},
                "sentence_count": {"type": "integer"},
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
    else:
        ensure_hash_id_mapping()
        ensure_sentence_chunk_mapping()


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


def chunk_text(
    text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP
) -> list[dict[str, Any]]:
    """Pack chunks on sentence boundaries before falling back to raw slicing."""
    structural_units = _split_structural_units(text)
    if not structural_units:
        return []

    chunks: list[dict[str, Any]] = []
    current_sentences: list[str] = []
    current_length = 0

    def flush_current() -> None:
        nonlocal current_sentences, current_length
        if not current_sentences:
            return
        chunk_text_value = " ".join(current_sentences).strip()
        if chunk_text_value:
            chunks.append(
                {
                    "text": chunk_text_value,
                    "sentence_texts": list(current_sentences),
                }
            )
        current_sentences = []
        current_length = 0

    for unit in structural_units:
        sentences = _split_sentences(unit)
        if not sentences:
            sentences = [unit.strip()]

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            if len(sentence) > chunk_size:
                flush_current()
                for fragment in _window_chunk(sentence, chunk_size, overlap):
                    chunks.append(
                        {
                            "text": fragment,
                            "sentence_texts": [fragment],
                        }
                    )
                continue

            projected = current_length + len(sentence) + (1 if current_sentences else 0)
            if current_sentences and projected > chunk_size:
                flush_current()
                projected = len(sentence)

            current_sentences.append(sentence)
            current_length = projected

    flush_current()
    return chunks


def _split_structural_units(text: str) -> list[str]:
    normalized = text.replace("\r\n", "\n")
    paragraphs = re.split(r"\n{2,}", normalized)
    units: list[str] = []
    for paragraph in paragraphs:
        stripped = paragraph.strip()
        if not stripped:
            continue
        lines = [line.strip() for line in stripped.splitlines() if line.strip()]
        if len(lines) > 1:
            units.extend(lines)
        else:
            units.append(stripped)
    return units


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
    return [part.strip() for part in parts if part.strip()]


def _window_chunk(text: str, chunk_size: int, overlap: int) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk.strip())
        start = end - overlap
    return chunks


def normalize_chunk_text(text: str) -> str:
    """Normalize chunk text for stable hashing."""
    return re.sub(r"\s+", " ", text).strip()


def build_document_hash_key(
    document_id: str,
    collection_name: str = "",
    source: str = "",
    title: str = "",
    fallback_id: str = "",
) -> str:
    """Build a stable per-document key for chunk dedupe."""
    if document_id:
        return document_id
    if source:
        return f"{collection_name}:{source}" if collection_name else source
    if title:
        return f"{collection_name}:{title}" if collection_name else title
    return fallback_id


def build_chunk_hash_id(document_key: str, chunk_text: str) -> str:
    """Hash a normalized chunk within its document scope."""
    normalized_text = normalize_chunk_text(chunk_text)
    return hashlib.sha256(f"{document_key}\n{normalized_text}".encode()).hexdigest()


def find_existing_hash_ids(hash_ids: list[str]) -> set[str]:
    """Return the subset of hash IDs that already exist in Elasticsearch."""
    existing: set[str] = set()
    unique_hash_ids = list(dict.fromkeys(hash_ids))
    for start in range(0, len(unique_hash_ids), HASH_LOOKUP_BATCH_SIZE):
        batch = unique_hash_ids[start : start + HASH_LOOKUP_BATCH_SIZE]
        if not batch:
            continue

        r = requests.post(
            f"{ELASTIC_URL}/{ELASTIC_INDEX}/_search",
            json={
                "size": len(batch),
                "track_total_hits": False,
                "_source": ["hash_id"],
                "collapse": {"field": "hash_id"},
                "query": {"terms": {"hash_id": batch}},
            },
        )
        r.raise_for_status()
        hits = r.json().get("hits", {}).get("hits", [])
        for hit in hits:
            hash_id = hit.get("_source", {}).get("hash_id")
            if hash_id:
                existing.add(hash_id)
    return existing


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------


def prepare_document(
    text: str,
    hash_key: str,
    fields: Mapping[str, object] | None = None,
):
    """Chunk a document and package it for batched ingestion."""
    if not text.strip():
        return None

    chunks = chunk_text(text)
    if not chunks:
        return None

    document_fields = dict(fields or {})
    _validate_document_fields(document_fields)
    return {
        "hash_key": hash_key,
        "fields": document_fields,
        "chunks": chunks,
    }


def prepare_document_chunks(
    chunks: list[str],
    hash_key: str,
    fields: Mapping[str, object] | None = None,
):
    """Package pre-chunked content for batched ingestion."""
    if not chunks:
        return None

    document_fields = dict(fields or {})
    _validate_document_fields(document_fields)
    return {
        "hash_key": hash_key,
        "fields": document_fields,
        "chunks": [
            {"text": chunk, "sentence_texts": _split_sentences(chunk) or [chunk]}
            for chunk in chunks
        ],
    }


def _validate_document_fields(fields: Mapping[str, object]):
    overlapping = RESERVED_INDEX_FIELDS.intersection(fields)
    if overlapping:
        raise ValueError(
            f"Document fields may not override reserved ingestion fields: {sorted(overlapping)}"
        )


def batched(items: Iterable, batch_size: int = INGEST_DOC_BATCH_SIZE) -> Iterator[list]:
    """Yield an iterable in fixed-size lists."""
    iterator = iter(items)
    while True:
        batch = list(islice(iterator, batch_size))
        if not batch:
            return
        yield batch


def ingest_prepared_items(
    items: Iterable,
    prepare_fn: Callable[[object], dict | None],
    *,
    total_items: int | None = None,
    progress_every: int | None = None,
    progress_prefix: str = "  ",
) -> tuple[int, int]:
    """Prepare and ingest items in parallel using the shared batching pipeline."""
    total_chunks = 0
    total_errors = 0
    processed = 0
    next_progress_at = progress_every or 0

    with ThreadPoolExecutor(max_workers=INGEST_PREPARE_WORKERS) as executor:
        for batch_items in batched(items, INGEST_DOC_BATCH_SIZE):
            documents = [doc for doc in executor.map(prepare_fn, batch_items) if doc is not None]
            ok = 0
            errs = 0
            if documents:
                ok, errs = index_documents(documents)
                total_chunks += ok
                total_errors += errs

            processed += len(batch_items)

            should_log = False
            if total_items is not None and processed == total_items:
                should_log = True
            elif progress_every is None:
                should_log = True
            elif processed >= next_progress_at:
                should_log = True
                next_progress_at += progress_every

            if should_log:
                if total_items is None:
                    progress = f"[{processed}]"
                else:
                    progress = f"[{processed}/{total_items}]"
                print(
                    f"{progress_prefix}{progress} flushed {len(documents)} docs "
                    f"({ok} chunks indexed, {errs} errors)"
                )

    return total_chunks, total_errors


def index_documents(documents: list[dict]) -> tuple[int, int]:
    """Index a batch of prepared documents into Elasticsearch."""
    records = _prepare_batch_records(documents)
    if not records:
        return 0, 0

    texts = [record["text"] for record in records]
    embeddings = []
    for start in range(0, len(texts), EMBED_BATCH_SIZE):
        embeddings.extend(embed_batch(texts[start : start + EMBED_BATCH_SIZE]))

    return _index_with_embeddings(records, embeddings)


def index_chunks(
    chunks: list[str], title: str, source: str, doc_id: str,
    doc_name: str = "", collection_name: str = "",
):
    """Index chunks into Elasticsearch (embeds via realtime API)."""
    document = prepare_document_chunks(
        chunks,
        hash_key=doc_id,
        fields={
            "title": title,
            "source": source,
            "document_id": doc_id,
            "doc_name": doc_name or title,
            "collection_name": collection_name,
        },
    )
    if document is None:
        return 0, 0
    return index_documents([document])


def index_chunks_preembedded(
    chunks: list[str], embeddings: list[list[float]], title: str, source: str, doc_id: str,
    doc_name: str = "", collection_name: str = "",
):
    """Index chunks into Elasticsearch with pre-computed embeddings."""
    document = prepare_document_chunks(
        chunks,
        hash_key=doc_id,
        fields={
            "title": title,
            "source": source,
            "document_id": doc_id,
            "doc_name": doc_name or title,
            "collection_name": collection_name,
        },
    )
    if document is None:
        return 0, 0

    records = _prepare_batch_records([document])
    if not records:
        return 0, 0

    filtered_embeddings = [embeddings[record["chunk_index"]] for record in records]
    return _index_with_embeddings(records, filtered_embeddings)


def _prepare_batch_records(documents: list[dict]) -> list[dict]:
    """Build chunk records for a batch of documents and drop existing hashes."""
    records = []
    seen_in_batch: set[str] = set()
    for document in documents:
        fields = dict(document.get("fields", {}))
        document_key = str(document["hash_key"])

        for chunk_index, chunk in enumerate(document.get("chunks", [])):
            if isinstance(chunk, str):
                chunk_text_value = chunk.strip()
                sentence_texts = _split_sentences(chunk_text_value) or [chunk_text_value]
            else:
                chunk_text_value = str(chunk.get("text", "")).strip()
                sentence_texts = [
                    sentence.strip()
                    for sentence in chunk.get("sentence_texts", [])
                    if isinstance(sentence, str) and sentence.strip()
                ]
                if not sentence_texts and chunk_text_value:
                    sentence_texts = _split_sentences(chunk_text_value) or [chunk_text_value]

            if not chunk_text_value:
                continue

            hash_id = build_chunk_hash_id(document_key, chunk_text_value)
            if hash_id in seen_in_batch:
                continue
            seen_in_batch.add(hash_id)
            records.append(
                {
                    "chunk_index": chunk_index,
                    "text": chunk_text_value,
                    "hash_id": hash_id,
                    "fields": fields,
                    "sentence_texts": sentence_texts,
                }
            )

    existing_hash_ids = find_existing_hash_ids([record["hash_id"] for record in records])
    return [record for record in records if record["hash_id"] not in existing_hash_ids]


def _index_with_embeddings(
    records: list[dict], embeddings: list[list[float]],
):
    """Bulk-index chunks with their embeddings into Elasticsearch."""
    created = 0
    errors = 0
    for start in range(0, len(records), INDEX_BULK_BATCH_SIZE):
        batch_records = records[start : start + INDEX_BULK_BATCH_SIZE]
        batch_embeddings = embeddings[start : start + INDEX_BULK_BATCH_SIZE]
        batch_created, batch_errors = _index_with_embeddings_batch(batch_records, batch_embeddings)
        created += batch_created
        errors += batch_errors
    return created, errors


def _index_with_embeddings_batch(records: list[dict], embeddings: list[list[float]]) -> tuple[int, int]:
    bulk_body = []
    for record, emb in zip(records, embeddings):
        body = dict(record["fields"])
        body["text"] = record["text"]
        body["chunk_index"] = record["chunk_index"]
        body["hash_id"] = record["hash_id"]
        body["sentence_texts"] = record.get("sentence_texts", [])
        body["sentence_count"] = len(record.get("sentence_texts", []))
        bulk_body.append(json.dumps({"create": {"_index": ELASTIC_INDEX, "_id": record["hash_id"]}}))
        body["embedding"] = emb
        bulk_body.append(json.dumps(body))

    bulk_payload = "\n".join(bulk_body) + "\n"
    r = requests.post(
        f"{ELASTIC_URL}/_bulk",
        data=bulk_payload,
        headers={"Content-Type": "application/x-ndjson"},
    )
    r.raise_for_status()
    result = r.json()

    created = 0
    errors = 0
    for item in result.get("items", []):
        create_result = item.get("create", {})
        error = create_result.get("error")
        if error is None:
            created += 1
            continue
        if error.get("type") == "version_conflict_engine_exception":
            continue
        errors += 1
    return created, errors


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
    document = prepare_document(
        text,
        hash_key=doc_id,
        fields={
            "title": title,
            "source": source,
            "document_id": doc_id,
            "doc_name": doc_name,
            "collection_name": collection_name,
        },
    )
    chunks = [] if document is None else document["chunks"]
    print(f"  Title: {title}")
    print(f"  Text length: {len(text)} chars")
    print(f"  Chunks: {len(chunks)}")

    if document is None:
        print("  Indexed: 0 chunks (0 errors)")
        return 0

    ok, errs = index_documents([document])
    print(f"  Indexed: {ok} chunks ({errs} errors)")
    return ok


def _prepare_source(source: str):
    try:
        is_url = source.startswith("http://") or source.startswith("https://")
        if is_url:
            title, text = fetch_url(source)
        else:
            title, text = read_file(source)

        doc_id = hashlib.sha256(source.encode()).hexdigest()[:16]
        document = prepare_document(
            text,
            hash_key=doc_id,
            fields={
                "title": title,
                "source": source,
                "document_id": doc_id,
                "doc_name": title,
                "collection_name": "",
            },
        )
        return {"source": source, "title": title, "text_len": len(text), "document": document, "error": None}
    except Exception as exc:
        return {"source": source, "title": source, "text_len": 0, "document": None, "error": exc}


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/ingest.py <url_or_file> [url_or_file ...]")
        sys.exit(1)

    ensure_index()
    ensure_hash_ids_ready()
    total = 0
    batch_documents = []

    with ThreadPoolExecutor(max_workers=INGEST_PREPARE_WORKERS) as executor:
        for i, result in enumerate(executor.map(_prepare_source, sys.argv[1:]), start=1):
            source = result["source"]
            title = result["title"]
            text_len = result["text_len"]
            document = result["document"]
            print(f"\n{'URL' if source.startswith('http://') or source.startswith('https://') else 'File'}: {source}")
            if result["error"] is not None:
                print(f"  ERROR: {result['error']}")
                continue

            print(f"  Title: {title}")
            print(f"  Text length: {text_len} chars")
            print(f"  Chunks: {0 if document is None else len(document['chunks'])}")

            if document is not None:
                batch_documents.append(document)

            if len(batch_documents) < INGEST_DOC_BATCH_SIZE and i != len(sys.argv) - 1:
                continue

            if not batch_documents:
                continue

            try:
                ok, errs = index_documents(batch_documents)
                total += ok
                print(f"  Batch indexed: {ok} chunks ({errs} errors)")
            except Exception as e:
                print(f"  ERROR: {e}")
            finally:
                batch_documents.clear()
    print(f"\nDone. Total chunks indexed: {total}")


if __name__ == "__main__":
    main()
