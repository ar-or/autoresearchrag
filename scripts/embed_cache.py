"""SQLite-backed embedding cache.

Stores embeddings keyed by SHA-256 hash of the input text.
Re-runs of ingestion scripts hit the cache instead of calling the API.

Uses a file lock so multiple processes can safely share the same database.

Environment variables:
  EMBED_CACHE - Path to the sqlite database (default: .embed_cache.db)
"""

import fcntl
import hashlib
import os
import sqlite3
import struct

EMBED_CACHE_PATH = os.environ.get("EMBED_CACHE", ".embed_cache.db")
_LOCK_PATH = EMBED_CACHE_PATH + ".lock"

_conn: sqlite3.Connection | None = None
_lock_fd = None


def _acquire_lock():
    global _lock_fd
    if _lock_fd is None:
        _lock_fd = open(_LOCK_PATH, "w")
    fcntl.flock(_lock_fd, fcntl.LOCK_EX)


def _release_lock():
    if _lock_fd is not None:
        fcntl.flock(_lock_fd, fcntl.LOCK_UN)


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
    _acquire_lock()
    try:
        row = _get_conn().execute(
            "SELECT embedding FROM embeddings WHERE text_hash = ?", (_text_hash(text),)
        ).fetchone()
        if row:
            return _unpack(row[0])
        return None
    finally:
        _release_lock()


def put(text: str, embedding: list[float]):
    _acquire_lock()
    try:
        _get_conn().execute(
            "INSERT OR REPLACE INTO embeddings (text_hash, embedding) VALUES (?, ?)",
            (_text_hash(text), _pack(embedding)),
        )
    finally:
        _release_lock()


def put_many(pairs: list[tuple[str, list[float]]]):
    _acquire_lock()
    try:
        conn = _get_conn()
        conn.executemany(
            "INSERT OR REPLACE INTO embeddings (text_hash, embedding) VALUES (?, ?)",
            [(_text_hash(text), _pack(emb)) for text, emb in pairs],
        )
        conn.commit()
    finally:
        _release_lock()


def flush():
    _acquire_lock()
    try:
        _get_conn().commit()
    finally:
        _release_lock()
