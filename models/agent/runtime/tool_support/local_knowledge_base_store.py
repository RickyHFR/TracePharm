from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import sys
from threading import Lock
from typing import Any

import faiss
import numpy as np
import yaml

from models.text_embedding_3_large import TextEmbedding3LargeClient

from .local_knowledge_base_query_cache import QueryEmbeddingCache
from .support_tools import normalize_medical_query_text


PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_CONFIG_ENV_VAR = "SEARCH_LOCAL_CONFIG_PATH"
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_MULTISPACE_RE = re.compile(r"\s+")
_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "what",
    "which",
    "with",
}


def _resolve_project_path(raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path.resolve()
    return (PROJECT_ROOT / path).resolve()


def _default_config_path() -> Path:
    raw = str(os.environ.get(DEFAULT_CONFIG_ENV_VAR, "") or "").strip()
    if raw:
        return _resolve_project_path(raw)
    return (PROJECT_ROOT / "models" / "agent" / "runtime" / "config" / "search_local.yaml").resolve()


def _load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping config at {path}")
    return data


def _normalize_text(text: Any) -> str:
    normalized = normalize_medical_query_text(str(text or ""))
    normalized = normalized.replace("®", " ").replace("™", " ")
    normalized = normalized.replace("/", " / ").replace("-", " ")
    return _MULTISPACE_RE.sub(" ", normalized).strip()


def _tokenize(text: Any) -> list[str]:
    tokens = _TOKEN_RE.findall(_normalize_text(text))
    return [token for token in tokens if len(token) >= 2 and token not in _STOPWORDS]


def compose_query_text(question: str, choices: Any = None, entities: list[str] | None = None) -> str:
    parts = [str(question or "").strip()]
    if isinstance(choices, dict):
        parts.append("\n".join(f"{key}. {value}" for key, value in choices.items()))
    elif isinstance(choices, list):
        parts.append("\n".join(str(item) for item in choices))
    elif choices:
        parts.append(str(choices))
    if entities:
        parts.append(" ".join(str(item).strip() for item in entities if str(item).strip()))
    return "\n".join(part for part in parts if str(part).strip())


def _safe_json_loads(raw: Any, default: Any) -> Any:
    text = str(raw or "").strip()
    if not text:
        return default
    try:
        return json.loads(text)
    except Exception:
        return default


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _portable_source_path(path: Path) -> str:
    """Return a clone-stable source identifier without embedding a home path."""
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        # External corpora cannot have a repo-relative path.  Their filename is
        # only an identifier; the SHA-256 below remains the integrity check.
        return resolved.name


def _legacy_source_fingerprints(items: Any) -> list[tuple[str, str]] | None:
    """Normalize schema-v1 absolute paths to (filename, sha256) fingerprints."""
    if not isinstance(items, list):
        return None
    fingerprints: list[tuple[str, str]] = []
    for item in items:
        if not isinstance(item, dict):
            return None
        source_path = str(item.get("path", "") or "").strip()
        digest = str(item.get("sha256", "") or "").strip().lower()
        if not source_path or not re.fullmatch(r"[0-9a-f]{64}", digest):
            return None
        fingerprints.append((Path(source_path).name, digest))
    return sorted(fingerprints)


class LocalKnowledgeBaseStore:
    SCHEMA_VERSION = 2

    def __init__(self, config_path: Path | None = None) -> None:
        self.config_path = Path(config_path or _default_config_path()).resolve()
        self.config = _load_yaml(self.config_path)
        self.read_only_runtime = bool(self.config.get("read_only_runtime", False))
        self.exclude_contradictions = bool(self.config.get("exclude_contradictions", True))
        self.allow_legacy_artifacts = bool(self.config.get("allow_legacy_artifacts", False))
        self.entity_info_column = (
            str(self.config.get("entity_info_column", "all_entity_info_json") or "all_entity_info_json").strip()
            or "all_entity_info_json"
        )
        self.source_csv_paths = [_resolve_project_path(str(item)) for item in self.config.get("source_csv_paths", [])]
        if not self.source_csv_paths:
            raise ValueError("search_local source_csv_paths must not be empty.")

        self.db_path = _resolve_project_path(str(self.config.get("database_path")))
        if not self.read_only_runtime:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.db_path.with_suffix(".manifest.json")
        self.faiss_index_path = _resolve_project_path(
            str(self.config.get("faiss_index_path") or self.db_path.with_suffix(".faiss"))
        )
        self.faiss_doc_ids_path = self.faiss_index_path.with_suffix(".doc_ids.json")
        self.embedding_config_path = _resolve_project_path(str(self.config.get("embedding_config_path")))
        embedding_config = _load_yaml(self.embedding_config_path)
        self.embedding_model_name = str(embedding_config.get("model", "text-embedding-3-large"))
        self.embedding_dimensions = int(embedding_config.get("dimensions", 3072) or 3072)
        self.query_cache_path = _resolve_project_path(
            str(self.config.get("query_cache_path") or self.db_path.with_name(f"{self.db_path.stem}_query_cache.sqlite"))
        )

        self._faiss_index: faiss.Index | None = None
        self._faiss_doc_ids: list[int] | None = None
        self._embedder: TextEmbedding3LargeClient | None = None
        self._query_cache: QueryEmbeddingCache | None = None

        self.ensure_ready()

    def _build_manifest(self) -> dict[str, Any]:
        embedding_config = _load_yaml(self.embedding_config_path)
        return {
            "schema_version": self.SCHEMA_VERSION,
            "entity_info_column": self.entity_info_column,
            "exclude_contradictions": self.exclude_contradictions,
            "embedding_model": str(embedding_config.get("model", "text-embedding-3-large")),
            "embedding_dimensions": int(embedding_config.get("dimensions", 3072) or 3072),
            "source_files": [
                {
                    "path": _portable_source_path(path),
                    "filename": path.name,
                    "sha256": _file_sha256(path),
                }
                for path in self.source_csv_paths
            ],
        }

    def _manifest_matches(self) -> bool:
        if not self.db_path.exists() or not self.manifest_path.exists():
            return False
        try:
            current = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except Exception:
            return False
        expected = self._build_manifest()
        if current == expected:
            return True
        if not self.allow_legacy_artifacts:
            return False
        # Reproduction-only compatibility: accept the existing schema-v1 index
        # when all source and embedding fingerprints still match. This never
        # upgrades or silently rebuilds the frozen artifact.
        legacy_keys = ("entity_info_column", "embedding_model", "embedding_dimensions")
        if not all(current.get(key) == expected.get(key) for key in legacy_keys):
            return False

        # Schema-v1 stored machine-specific absolute source paths.  Accept a
        # relocated clone only when every source filename and SHA-256 still
        # matches.  This removes the home-directory dependency without
        # weakening corpus-content verification or rebuilding the artifact.
        try:
            current_schema = int(current.get("schema_version", 0) or 0)
        except Exception:
            return False
        if current_schema > 1:
            return current.get("source_files") == expected.get("source_files")
        return _legacy_source_fingerprints(current.get("source_files")) == _legacy_source_fingerprints(
            expected.get("source_files")
        )

    def _connect(self) -> sqlite3.Connection:
        if self.read_only_runtime:
            conn = sqlite3.connect(
                f"file:{self.db_path.as_posix()}?mode=ro&immutable=1",
                uri=True,
            )
        else:
            conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        if self.read_only_runtime:
            conn.execute("PRAGMA query_only=ON;")
        else:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA temp_store=MEMORY;")
        conn.execute("PRAGMA mmap_size=268435456;")
        return conn

    def _get_embedder(self) -> TextEmbedding3LargeClient:
        if self._embedder is None:
            self._embedder = TextEmbedding3LargeClient(config_path=self.embedding_config_path)
        return self._embedder

    def _get_query_cache(self) -> QueryEmbeddingCache:
        if self._query_cache is None:
            self._query_cache = QueryEmbeddingCache(
                cache_path=self.query_cache_path,
                read_only=self.read_only_runtime,
            )
        return self._query_cache

    def ensure_ready(self) -> None:
        if self.read_only_runtime:
            if not self._manifest_matches():
                raise RuntimeError(
                    "Local KB is not a frozen artifact matching its manifest. "
                    "Run the explicit offline build command before evaluation; runtime rebuilding is disabled."
                )
            if not self.faiss_index_path.exists() or not self.faiss_doc_ids_path.exists():
                raise RuntimeError("Frozen local KB is missing its FAISS artifacts.")
            if not self.query_cache_path.is_file():
                raise RuntimeError("Frozen local KB is missing its query-embedding cache.")
            # Open and validate the cache now so a malformed frozen artifact
            # fails before the first benchmark example is evaluated.
            self._get_query_cache()
            return
        self._ensure_db()
        self._ensure_faiss_index()

    def _ensure_db(self) -> None:
        if self._manifest_matches():
            return
        self.rebuild()

    def rebuild(self) -> None:
        if self.read_only_runtime:
            raise RuntimeError("Cannot rebuild a local KB while read_only_runtime=true.")
        if self.db_path.exists():
            self.db_path.unlink()
        if self.manifest_path.exists():
            self.manifest_path.unlink()
        if self.faiss_index_path.exists():
            self.faiss_index_path.unlink()
        if self.faiss_doc_ids_path.exists():
            self.faiss_doc_ids_path.unlink()

        documents = self._load_documents()
        if not documents:
            raise ValueError("No knowledge-base documents could be extracted from the configured CSV files.")

        vectors = self._get_embedder().embed_texts([document["chunk_text"] for document in documents])
        if len(vectors) != len(documents):
            raise ValueError("Embedding count did not match extracted knowledge-base chunks.")

        conn = self._connect()
        try:
            conn.executescript(
                """
                CREATE TABLE kb_document (
                    doc_id INTEGER PRIMARY KEY,
                    benchmark TEXT NOT NULL,
                    question_index INTEGER NOT NULL,
                    source_csv TEXT NOT NULL,
                    entity_name TEXT NOT NULL,
                    source_url TEXT NOT NULL,
                    chunk_text TEXT NOT NULL,
                    normalized_chunk_text TEXT NOT NULL,
                    metadata_json TEXT NOT NULL
                );

                CREATE TABLE kb_embedding (
                    doc_id INTEGER PRIMARY KEY,
                    model_name TEXT NOT NULL,
                    dimensions INTEGER NOT NULL,
                    vector_json TEXT NOT NULL
                );

                CREATE INDEX idx_kb_document_benchmark_question ON kb_document(benchmark, question_index);
                CREATE VIRTUAL TABLE kb_fts USING fts5(search_text);
                """
            )

            embedding_config = _load_yaml(self.embedding_config_path)
            model_name = str(embedding_config.get("model", "text-embedding-3-large"))
            dimensions = len(vectors[0]) if vectors else int(embedding_config.get("dimensions", 3072) or 3072)

            for doc_id, (document, vector) in enumerate(zip(documents, vectors, strict=True), start=1):
                metadata_json = json.dumps(document["metadata"], ensure_ascii=False, separators=(",", ":"))
                conn.execute(
                    """
                    INSERT INTO kb_document (
                        doc_id, benchmark, question_index, source_csv, entity_name, source_url,
                        chunk_text, normalized_chunk_text, metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        doc_id,
                        document["benchmark"],
                        document["question_index"],
                        document["source_csv"],
                        document["entity_name"],
                        document["source_url"],
                        document["chunk_text"],
                        document["normalized_chunk_text"],
                        metadata_json,
                    ),
                )
                conn.execute(
                    """
                    INSERT INTO kb_embedding (doc_id, model_name, dimensions, vector_json)
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        doc_id,
                        model_name,
                        dimensions,
                        json.dumps(vector, separators=(",", ":")),
                    ),
                )
                search_text = " ".join(part for part in [document["entity_name"], document["chunk_text"]] if part).strip()
                conn.execute(
                    "INSERT INTO kb_fts(rowid, search_text) VALUES (?, ?)",
                    (doc_id, search_text),
                )
            conn.commit()
        finally:
            conn.close()

        doc_ids = list(range(1, len(vectors) + 1))
        self._write_faiss_index(doc_ids=doc_ids, vectors=vectors)
        self.manifest_path.write_text(
            json.dumps(self._build_manifest(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _load_documents(self) -> list[dict[str, Any]]:
        documents: list[dict[str, Any]] = []
        seen_keys: set[tuple[str, int, str, str, str]] = set()

        csv.field_size_limit(sys.maxsize)
        for csv_path in self.source_csv_paths:
            with csv_path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    benchmark = str(row.get("benchmark", "") or "").strip()
                    try:
                        question_index = int(row.get("index", "") or 0)
                    except Exception:
                        continue
                    raw_triples = row.get(self.entity_info_column)
                    if raw_triples is None and self.entity_info_column != "all_entity_info_json":
                        raw_triples = row.get("all_entity_info_json")
                    triples = _safe_json_loads(raw_triples, [])
                    if not isinstance(triples, list):
                        continue
                    for triple_index, triple in enumerate(triples):
                        if not isinstance(triple, list) or len(triple) < 3:
                            continue
                        entity_name = str(triple[0] or "").strip()
                        chunk_text = str(triple[1] or "").strip()
                        source_url = str(triple[2] or "").strip()
                        status_raw = str(triple[3] or "").strip() if len(triple) >= 4 else ""
                        contradiction = bool(re.search(r"(?:^|\b)contradict\s*=\s*1(?:\b|$)", status_raw, re.IGNORECASE))
                        if contradiction and self.exclude_contradictions:
                            continue
                        if not chunk_text:
                            continue
                        dedupe_key = (benchmark, question_index, entity_name, chunk_text, source_url)
                        if dedupe_key in seen_keys:
                            continue
                        seen_keys.add(dedupe_key)
                        metadata = {
                            "entity_name": entity_name,
                            "source_url": source_url,
                            "benchmark": benchmark,
                            "question_index": question_index,
                            "source_csv": csv_path.name,
                            "triple_index": triple_index,
                            "evidence_status": "contradiction" if contradiction else "usable",
                            "contradiction": contradiction,
                            "contradiction_note": status_raw if contradiction else "",
                            "document_hash": hashlib.sha256(chunk_text.encode("utf-8")).hexdigest(),
                        }
                        documents.append(
                            {
                                "benchmark": benchmark,
                                "question_index": question_index,
                                "source_csv": csv_path.name,
                                "entity_name": entity_name,
                                "source_url": source_url,
                                "chunk_text": chunk_text,
                                "normalized_chunk_text": _normalize_text(chunk_text),
                                "metadata": metadata,
                            }
                        )
        return documents

    def _fetch_all_vectors(self) -> tuple[list[int], list[list[float]]]:
        conn = self._connect()
        try:
            rows = conn.execute(
                """
                SELECT kb_document.doc_id AS doc_id, kb_embedding.vector_json AS vector_json
                FROM kb_document
                JOIN kb_embedding ON kb_embedding.doc_id = kb_document.doc_id
                ORDER BY kb_document.doc_id
                """
            ).fetchall()
        finally:
            conn.close()

        doc_ids = [int(row["doc_id"]) for row in rows]
        vectors = [list(_safe_json_loads(row["vector_json"], [])) for row in rows]
        return doc_ids, vectors

    def _write_faiss_index(self, doc_ids: list[int], vectors: list[list[float]]) -> None:
        if self.read_only_runtime:
            raise RuntimeError("Cannot write FAISS artifacts while read_only_runtime=true.")
        matrix = np.asarray(vectors, dtype="float32")
        if matrix.ndim != 2 or matrix.shape[0] == 0:
            raise ValueError("Expected a non-empty 2D embedding matrix for FAISS index creation.")
        faiss.normalize_L2(matrix)
        index = faiss.IndexFlatIP(int(matrix.shape[1]))
        index.add(matrix)
        faiss.write_index(index, str(self.faiss_index_path))
        self.faiss_doc_ids_path.write_text(json.dumps(doc_ids), encoding="utf-8")
        self._faiss_index = index
        self._faiss_doc_ids = list(doc_ids)

    def _ensure_faiss_index(self) -> None:
        if self.faiss_index_path.exists() and self.faiss_doc_ids_path.exists():
            return
        if self.read_only_runtime:
            raise RuntimeError("Frozen local KB is missing its FAISS artifacts.")
        doc_ids, vectors = self._fetch_all_vectors()
        if not doc_ids or not vectors:
            raise ValueError("Cannot build FAISS index because no stored vectors were found.")
        self._write_faiss_index(doc_ids=doc_ids, vectors=vectors)

    def _load_faiss_artifacts(self) -> tuple[faiss.Index, list[int]]:
        self._ensure_faiss_index()
        if self._faiss_index is None:
            self._faiss_index = faiss.read_index(str(self.faiss_index_path))
        if self._faiss_doc_ids is None:
            self._faiss_doc_ids = [int(item) for item in json.loads(self.faiss_doc_ids_path.read_text(encoding="utf-8"))]
        return self._faiss_index, self._faiss_doc_ids

    def bm25_search(self, query_text: str, k: int) -> dict[int, float]:
        terms = list(dict.fromkeys(_tokenize(query_text)))
        if not terms:
            return {}
        match_query = " OR ".join(terms)
        conn = self._connect()
        try:
            rows = conn.execute(
                """
                SELECT rowid AS doc_id, bm25(kb_fts) AS bm25_score
                FROM kb_fts
                WHERE kb_fts MATCH ?
                ORDER BY bm25_score
                LIMIT ?
                """,
                (match_query, max(1, k)),
            ).fetchall()
        finally:
            conn.close()

        scores: dict[int, float] = {}
        for row in rows:
            raw = float(row["bm25_score"] or 0.0)
            scores[int(row["doc_id"])] = -raw if raw < 0 else 1.0 / (1.0 + raw)
        return scores

    def dense_search(self, query_text: str, k: int) -> dict[int, float]:
        if not str(query_text or "").strip():
            return {}
        index, doc_ids = self._load_faiss_artifacts()
        cache = self._get_query_cache()
        cached = cache.get(
            model_name=self.embedding_model_name,
            dimensions=self.embedding_dimensions,
            query_text=query_text,
        )
        if cached is None:
            embedder = self._get_embedder()
            cached = embedder.embed_query(query_text)
            if cached:
                cache.set(
                    model_name=self.embedding_model_name,
                    dimensions=self.embedding_dimensions,
                    query_text=query_text,
                    vector=cached,
                )
        vector = np.asarray([cached], dtype="float32")
        if vector.size == 0:
            return {}
        faiss.normalize_L2(vector)
        limit = min(max(1, k), len(doc_ids))
        distances, indices = index.search(vector, limit)
        scores: dict[int, float] = {}
        for score, idx in zip(distances[0].tolist(), indices[0].tolist(), strict=True):
            if idx < 0 or idx >= len(doc_ids):
                continue
            scores[int(doc_ids[idx])] = float(score)
        return scores

    def fetch_documents(self, doc_ids: list[int]) -> dict[int, dict[str, Any]]:
        if not doc_ids:
            return {}
        conn = self._connect()
        try:
            placeholders = ",".join("?" for _ in doc_ids)
            rows = conn.execute(
                f"""
                SELECT doc_id, benchmark, question_index, source_csv, entity_name, source_url,
                       chunk_text, normalized_chunk_text, metadata_json
                FROM kb_document
                WHERE doc_id IN ({placeholders})
                """,
                tuple(doc_ids),
            ).fetchall()
        finally:
            conn.close()

        documents: dict[int, dict[str, Any]] = {}
        for row in rows:
            metadata = _safe_json_loads(row["metadata_json"], {})
            documents[int(row["doc_id"])] = {
                "doc_id": int(row["doc_id"]),
                "benchmark": str(row["benchmark"]),
                "question_index": int(row["question_index"]),
                "source_csv": str(row["source_csv"]),
                "entity_name": str(row["entity_name"]),
                "source_url": str(row["source_url"]),
                "chunk_text": str(row["chunk_text"]),
                "normalized_chunk_text": str(row["normalized_chunk_text"]),
                "metadata": metadata,
                "contradiction": bool(metadata.get("contradiction", False)) if isinstance(metadata, dict) else False,
                "evidence_status": str(metadata.get("evidence_status", "usable")) if isinstance(metadata, dict) else "usable",
            }
        return documents

    def stats(self) -> dict[str, Any]:
        self.ensure_ready()
        conn = self._connect()
        try:
            document_count = int(conn.execute("SELECT COUNT(*) FROM kb_document").fetchone()[0])
            benchmark_counts = {
                row["benchmark"]: int(row["count"])
                for row in conn.execute(
                    "SELECT benchmark, COUNT(*) AS count FROM kb_document GROUP BY benchmark ORDER BY benchmark"
                ).fetchall()
            }
            source_files = [str(path) for path in self.source_csv_paths]
            cache_stats = self._get_query_cache().stats()
            return {
                "database_path": str(self.db_path),
                "faiss_index_path": str(self.faiss_index_path),
                "query_cache_path": str(cache_stats["cache_path"]),
                "entity_info_column": self.entity_info_column,
                "cached_query_count": int(cache_stats["cached_query_count"]),
                "document_count": document_count,
                "benchmarks": benchmark_counts,
                "source_files": source_files,
            }
        finally:
            conn.close()

    def close(self) -> None:
        embedder = self._embedder
        self._embedder = None
        if embedder is not None:
            embedder.close()


_STORE_CACHE: dict[str, LocalKnowledgeBaseStore] = {}
_STORE_CACHE_LOCK = Lock()


def get_store(config_path: Path | None = None) -> LocalKnowledgeBaseStore:
    resolved = Path(config_path or _default_config_path()).resolve()
    cache_key = str(resolved)
    with _STORE_CACHE_LOCK:
        store = _STORE_CACHE.get(cache_key)
        if store is None:
            store = LocalKnowledgeBaseStore(config_path=resolved)
            _STORE_CACHE[cache_key] = store
        return store
