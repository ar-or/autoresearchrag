#!/usr/bin/env python3
"""Test the embedding cache and batch API with a few sample texts."""

import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.embedder import embed_batch, embed_batch_api, EMBED_DIMS
from scripts.embed_cache import flush as cache_flush, EMBED_CACHE_PATH

SAMPLE_TEXTS = [
    "The quick brown fox jumps over the lazy dog.",
    "Elasticsearch is a distributed search and analytics engine.",
    "Python is a high-level programming language.",
    "Machine learning models can generate text embeddings.",
    "The Eiffel Tower is located in Paris, France.",
]


def main():
    print(f"Cache path: {EMBED_CACHE_PATH}")
    print(f"Expected dims: {EMBED_DIMS}")
    print()

    # --- Test 1: Realtime API with cache ---
    print("=== Test 1: Realtime embed_batch (first call, should miss cache) ===")
    embeddings = embed_batch(SAMPLE_TEXTS[:3])
    for i, emb in enumerate(embeddings):
        print(f"  [{i}] dims={len(emb)}, first3={emb[:3]}")
    assert all(len(e) == EMBED_DIMS for e in embeddings), "Dimension mismatch!"

    print("\n=== Test 2: Realtime embed_batch (second call, should hit cache) ===")
    embeddings2 = embed_batch(SAMPLE_TEXTS[:3])
    for i, (a, b) in enumerate(zip(embeddings, embeddings2)):
        match = a == b
        print(f"  [{i}] cache match: {match}")
    assert embeddings == embeddings2, "Cache returned different embeddings!"

    # --- Test 3: Batch API ---
    print("\n=== Test 3: Batch API (2 cached + 2 uncached) ===")
    mixed_texts = SAMPLE_TEXTS[:2] + SAMPLE_TEXTS[3:]  # first 2 cached, last 2 not
    embeddings3 = embed_batch_api(mixed_texts, desc="test")
    for i, emb in enumerate(embeddings3):
        print(f"  [{i}] dims={len(emb)}, first3={emb[:3]}")
    assert all(len(e) == EMBED_DIMS for e in embeddings3), "Dimension mismatch!"

    # Verify the cached ones match
    assert embeddings3[0] == embeddings[0], "Batch API cache mismatch for text 0!"
    assert embeddings3[1] == embeddings[1], "Batch API cache mismatch for text 1!"

    print("\n=== Test 4: Batch API (all cached now) ===")
    embeddings4 = embed_batch_api(mixed_texts, desc="test-cached")
    assert embeddings3 == embeddings4, "Full cache mismatch!"
    print("  All from cache - OK")

    cache_flush()
    print("\n All tests passed!")


if __name__ == "__main__":
    main()
