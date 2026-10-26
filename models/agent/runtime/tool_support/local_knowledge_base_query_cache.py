from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Any


class QueryEmbeddingCache:
    """Persistent cache for query embeddings to avoid repeated API calls."""

    TABLE_NAME = "query_embedding_cache"
    REQUIRED_COLUMNS = (
        "cache_key",
        "model_name",
        "dimensions",
        "query_text",
        "vector_json",
    )

    def __init__(self, cache_path: Path, *, read_only: bool = False) -> None:
        self.cache_path = Path(cache_path).resolve()
        self.read_only = bool(read_only)
        if self.read_only:
            if not self.cache_path.is_file():
                raise FileNotFoundError(
                    f"Read-only query-embedding cache does not exist: {self.cache_path}"
                )
            self._validate_schema()
        else:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        if self.read_only:
            conn = sqlite3.connect(
                f"file:{self.cache_path.as_posix()}?mode=ro&immutable=1",
                uri=True,
            )
        else:
            conn = sqlite3.connect(str(self.cache_path))
        conn.row_factory = sqlite3.Row
        if self.read_only:
            # query_only is connection-local and does not mutate the database.
            # Together with mode=ro it protects the frozen cache even when a
            # caller accidentally executes a write statement.
            conn.execute("PRAGMA query_only=ON;")
        else:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def _validate_schema(self) -> None:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT type FROM sqlite_master WHERE name = ?",
                (self.TABLE_NAME,),
            ).fetchone()
            if row is None or str(row["type"]) != "table":
                raise ValueError(
                    "Read-only query-embedding cache is missing table "
                    f"{self.TABLE_NAME}: {self.cache_path}"
                )
            columns = tuple(
                str(item["name"])
                for item in conn.execute(
                    f"PRAGMA table_info({self.TABLE_NAME})"
                ).fetchall()
            )
            if columns != self.REQUIRED_COLUMNS:
                raise ValueError(
                    "Read-only query-embedding cache has an unexpected schema: "
                    f"expected={self.REQUIRED_COLUMNS}, actual={columns}, "
                    f"path={self.cache_path}"
                )
        finally:
            conn.close()

    def _ensure_schema(self) -> None:
        conn = self._connect()
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS query_embedding_cache (
                    cache_key TEXT PRIMARY KEY,
                    model_name TEXT NOT NULL,
                    dimensions INTEGER NOT NULL,
                    query_text TEXT NOT NULL,
                    vector_json TEXT NOT NULL
                )
                """
            )
            conn.commit()
        finally:
            conn.close()

    def _cache_key(self, *, model_name: str, dimensions: int, query_text: str) -> str:
        digest = hashlib.sha256()
        digest.update(model_name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(dimensions).encode("utf-8"))
        digest.update(b"\0")
        digest.update(query_text.encode("utf-8"))
        return digest.hexdigest()

    def get(self, *, model_name: str, dimensions: int, query_text: str) -> list[float] | None:
        cache_key = self._cache_key(model_name=model_name, dimensions=dimensions, query_text=query_text)
        conn = self._connect()
        try:
            row = conn.execute(
                """
                SELECT vector_json
                FROM query_embedding_cache
                WHERE cache_key = ?
                """,
                (cache_key,),
            ).fetchone()
        finally:
            conn.close()

        if row is None:
            return None
        try:
            vector = json.loads(str(row["vector_json"]))
        except Exception:
            return None
        if not isinstance(vector, list):
            return None
        return [float(item) for item in vector]

    def set(self, *, model_name: str, dimensions: int, query_text: str, vector: list[float]) -> None:
        if self.read_only:
            # A cache miss during evaluation may still be embedded live, but
            # formal runtime state is immutable and therefore must not be
            # persisted.  Deliberately make this a no-op rather than failing
            # the answer path after a successful embedding request.
            return
        cache_key = self._cache_key(model_name=model_name, dimensions=dimensions, query_text=query_text)
        conn = self._connect()
        try:
            conn.execute(
                """
                INSERT INTO query_embedding_cache (
                    cache_key, model_name, dimensions, query_text, vector_json
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(cache_key) DO UPDATE SET
                    vector_json = excluded.vector_json,
                    query_text = excluded.query_text,
                    model_name = excluded.model_name,
                    dimensions = excluded.dimensions
                """,
                (
                    cache_key,
                    model_name,
                    dimensions,
                    query_text,
                    json.dumps(vector, separators=(",", ":")),
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def stats(self) -> dict[str, Any]:
        conn = self._connect()
        try:
            row = conn.execute("SELECT COUNT(*) AS count FROM query_embedding_cache").fetchone()
            count = int(row["count"]) if row is not None else 0
        finally:
            conn.close()
        return {
            "cache_path": str(self.cache_path),
            "cached_query_count": count,
            "read_only": self.read_only,
        }
