"""SQLite-backed embedding cache.

Stores embeddings keyed by SHA-256 hash of the input text.
Re-runs of ingestion scripts hit the cache instead of calling the API.

Environment variables:
  EMBED_CACHE - Path to the sqlite database (default: .embed_cache.db)
"""

import hashlib
import os
import sqlite3
import struct

EMBED_CACHE_PATH = os.environ.get("EMBED_CACHE", ".embed_cache.db")

_conn: sqlite3.Connection | None = None
_SQLITE_VAR_LIMIT = 900


def _get_conn() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        _conn = sqlite3.connect(EMBED_CACHE_PATH, timeout=30)
        _conn.execute(
            "CREATE TABLE IF NOT EXISTS embeddings "
            "(text_hash TEXT PRIMARY KEY, embedding BLOB)"
        )
        _conn.execute("PRAGMA journal_mode=WAL")
        _conn.execute("PRAGMA synchronous=NORMAL")
    return _conn


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _pack(emb: list[float]) -> bytes:
    return struct.pack(f"{len(emb)}d", *emb)


def _unpack(blob: bytes) -> list[float]:
    n = len(blob) // 8
    return list(struct.unpack(f"{n}d", blob))


def get(text: str) -> list[float] | None:
    row = _get_conn().execute(
        "SELECT embedding FROM embeddings WHERE text_hash = ?", (_text_hash(text),)
    ).fetchone()
    if row:
        return _unpack(row[0])
    return None


def get_many(texts: list[str]) -> dict[str, list[float]]:
    if not texts:
        return {}

    conn = _get_conn()
    by_hash: dict[str, list[str]] = {}
    for text in texts:
        by_hash.setdefault(_text_hash(text), []).append(text)

    found: dict[str, list[float]] = {}
    text_hashes = list(by_hash)
    for start in range(0, len(text_hashes), _SQLITE_VAR_LIMIT):
        batch = text_hashes[start : start + _SQLITE_VAR_LIMIT]
        placeholders = ",".join("?" for _ in batch)
        rows = conn.execute(
            f"SELECT text_hash, embedding FROM embeddings WHERE text_hash IN ({placeholders})",
            batch,
        ).fetchall()
        for text_hash, blob in rows:
            embedding = _unpack(blob)
            for text in by_hash.get(text_hash, []):
                found[text] = embedding
    return found


def put(text: str, embedding: list[float]):
    _get_conn().execute(
        "INSERT OR REPLACE INTO embeddings (text_hash, embedding) VALUES (?, ?)",
        (_text_hash(text), _pack(embedding)),
    )


def put_many(pairs: list[tuple[str, list[float]]]):
    conn = _get_conn()
    conn.executemany(
        "INSERT OR REPLACE INTO embeddings (text_hash, embedding) VALUES (?, ?)",
        [(_text_hash(text), _pack(emb)) for text, emb in pairs],
    )
    conn.commit()


def flush():
    _get_conn().commit()
