"""OpenAI embedding client with cache support.

Provides two modes:
  - embed_batch():     Realtime API (immediate, full price)
  - embed_batch_api(): Batch API (async, 50% cheaper)

Both check the sqlite cache first and store new embeddings back.

Environment variables:
  OPENAI_API_KEY - Required by the OpenAI client
"""

import json
import os
import tempfile
import time

from openai import OpenAI

from scripts import embed_cache

EMBED_MODEL = "text-embedding-3-small"
EMBED_DIMS = 1536
EMBED_MAX_REQUEST_ITEMS = int(os.environ.get("EMBED_MAX_REQUEST_ITEMS", "64"))
EMBED_MAX_REQUEST_CHARS = int(os.environ.get("EMBED_MAX_REQUEST_CHARS", "24000"))

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI()
    return _client


def _chunk_embedding_inputs(texts: list[str]) -> list[list[str]]:
    batches: list[list[str]] = []
    current_batch: list[str] = []
    current_chars = 0

    for text in texts:
        text_chars = max(len(text), 1)
        should_flush = (
            current_batch
            and (
                len(current_batch) >= EMBED_MAX_REQUEST_ITEMS
                or current_chars + text_chars > EMBED_MAX_REQUEST_CHARS
            )
        )
        if should_flush:
            batches.append(current_batch)
            current_batch = []
            current_chars = 0

        current_batch.append(text)
        current_chars += text_chars

    if current_batch:
        batches.append(current_batch)
    return batches


def embed_batch(texts: list[str], max_retries: int = 3) -> list[list[float]]:
    """Embed texts using the realtime API, with cache and retry."""
    results = [None] * len(texts)
    cached_embeddings = embed_cache.get_many(texts)
    uncached_indices = []

    for i, text in enumerate(texts):
        cached = cached_embeddings.get(text)
        if cached is None:
            uncached_indices.append(i)
            continue
        results[i] = cached

    if uncached_indices:
        uncached_texts = [texts[i] for i in uncached_indices]
        new_embeddings = []
        for request_batch in _chunk_embedding_inputs(uncached_texts):
            for attempt in range(1, max_retries + 1):
                try:
                    resp = _get_client().embeddings.create(input=request_batch, model=EMBED_MODEL)
                    break
                except Exception as e:
                    if attempt == max_retries:
                        raise RuntimeError(
                            f"Embedding API failed after {max_retries} attempts: {e}"
                        ) from e
                    wait = 2 ** attempt
                    print(f"  Embedding API error (attempt {attempt}/{max_retries}): {e}. Retrying in {wait}s...")
                    time.sleep(wait)
            new_embeddings.extend(d.embedding for d in resp.data)
        pairs = []
        for idx, emb in zip(uncached_indices, new_embeddings):
            results[idx] = emb
            pairs.append((texts[idx], emb))
        embed_cache.put_many(pairs)

    return results


def embed_batch_api(texts: list[str], batch_size: int = 50000, desc: str = "") -> list[list[float]]:
    """Embed texts using the OpenAI Batch API (50% cheaper).

    Checks the cache first; only sends uncached texts to the API.
    Results are stored back in the cache.
    """
    results = [None] * len(texts)
    cached_embeddings = embed_cache.get_many(texts)
    uncached_indices = []

    for i, text in enumerate(texts):
        cached = cached_embeddings.get(text)
        if cached is None:
            uncached_indices.append(i)
            continue
        results[i] = cached

    cached_count = len(texts) - len(uncached_indices)
    if cached_count > 0:
        print(f"  Cache hit: {cached_count}/{len(texts)} embeddings")

    if not uncached_indices:
        return results

    print(f"  Sending {len(uncached_indices)} texts to Batch API...")

    for batch_start in range(0, len(uncached_indices), batch_size):
        batch_indices = uncached_indices[batch_start : batch_start + batch_size]
        _embed_batch_api_chunk(texts, results, batch_indices, desc)

    return results


def _embed_batch_api_chunk(
    texts: list[str],
    results: list[list[float] | None],
    indices: list[int],
    desc: str,
):
    """Send one chunk of texts to the Batch API and fill in results."""
    client = _get_client()

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        jsonl_path = f.name
        for i in indices:
            line = {
                "custom_id": str(i),
                "method": "POST",
                "url": "/v1/embeddings",
                "body": {"model": EMBED_MODEL, "input": texts[i]},
            }
            f.write(json.dumps(line) + "\n")

    try:
        with open(jsonl_path, "rb") as f:
            file_obj = client.files.create(file=f, purpose="batch")
        print(f"  Uploaded batch file: {file_obj.id} ({len(indices)} requests)")

        batch = client.batches.create(
            input_file_id=file_obj.id,
            endpoint="/v1/embeddings",
            completion_window="24h",
        )
        print(f"  Batch created: {batch.id}")

        while True:
            batch = client.batches.retrieve(batch.id)
            status = batch.status
            completed = batch.request_counts.completed if batch.request_counts else 0
            total = batch.request_counts.total if batch.request_counts else len(indices)
            print(f"  Status: {status} ({completed}/{total})", end="\r")

            if status in ("completed", "failed", "expired", "cancelled"):
                print()
                break
            time.sleep(5)

        if status != "completed":
            failed = batch.request_counts.failed if batch.request_counts else 0
            raise RuntimeError(f"Batch {batch.id} ended with status: {status} ({failed} failed)")

        output_file = client.files.content(batch.output_file_id)
        pairs = []
        for line in output_file.text.strip().split("\n"):
            item = json.loads(line)
            idx = int(item["custom_id"])
            embedding = item["response"]["body"]["data"][0]["embedding"]
            results[idx] = embedding
            pairs.append((texts[idx], embedding))

        embed_cache.put_many(pairs)
        print(f"  Batch complete: {len(pairs)} embeddings cached")

    finally:
        os.unlink(jsonl_path)
