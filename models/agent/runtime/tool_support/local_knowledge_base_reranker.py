from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import tempfile
import time
from threading import BoundedSemaphore, Lock
from typing import Any

from llm.deepseek_client import DeepSeekClient
from agent.telemetry import bind_timing_recorder, get_active_timing_recorder


_GLOBAL_RERANK_CONCURRENCY = 50
_RERANK_SEMAPHORE = BoundedSemaphore(_GLOBAL_RERANK_CONCURRENCY)
_RERANK_CLIENTS: dict[str, DeepSeekClient] = {}
_RERANK_CLIENTS_LOCK = Lock()
_RERANK_SLOT_DIR = Path(tempfile.gettempdir()) / "nature_search_local_rerank_slots"
_RERANK_CACHE_SCHEMA_LOCK = Lock()
_RERANK_PROMPT_VERSION = "local_kb_rerank_v2_exact_visible_query"


def _strip_code_fences(text: str) -> str:
    cleaned = str(text or "").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


class LocalKnowledgeBaseReranker:
    def __init__(self, config: dict[str, Any], project_root: Path) -> None:
        self.enabled = bool(config.get("rerank_enabled", True))
        self.candidate_k = max(1, int(config.get("rerank_candidate_k", 5) or 5))
        self.batch_size = max(1, int(config.get("rerank_batch_size", 5) or 5))
        self.max_retries = max(1, int(config.get("rerank_max_retries", 3) or 3))
        self.max_workers = min(_GLOBAL_RERANK_CONCURRENCY, max(1, int(config.get("rerank_max_workers", 32) or 32)))
        self.keep_scores = {int(item) for item in (config.get("rerank_keep_scores") or [1, 2])}
        self.always_keep_hybrid_top_k = max(0, int(config.get("rerank_always_keep_hybrid_top_k", 0) or 0))
        self.cache_fallback_scores = bool(config.get("rerank_cache_fallback_scores", True))
        support_atom_cap = config.get("support_atom_max_rerank_score", 1)
        self.support_atom_max_rerank_score = None if support_atom_cap is None else int(support_atom_cap)
        self.cache_enabled = bool(config.get("rerank_cache_enabled", True))
        self.fallback_on_error = bool(config.get("rerank_fallback_on_error", True))
        cache_path_raw = str(config.get("rerank_cache_path") or "").strip()
        if cache_path_raw:
            cache_path = Path(cache_path_raw)
            if not cache_path.is_absolute():
                cache_path = (project_root / cache_path).resolve()
        else:
            cache_path = (project_root / "resources/search_local_key_entity/drug_knowledge_base_rerank_cache.sqlite").resolve()
        self.cache_path = cache_path
        model_config_path = Path(str(config.get("rerank_model_config_path") or "models/gpt_5_4/config.yaml"))
        if not model_config_path.is_absolute():
            model_config_path = (project_root / model_config_path).resolve()
        self.model_config_path = model_config_path
        # Client construction validates credentials.  Keep it lazy so loading
        # a frozen local index remains possible in offline diagnostics; a
        # missing/unavailable reranker falls back to deterministic hybrid rank.
        self.client: DeepSeekClient | None = None
        self.model_name = ""

    def _ensure_cache_schema(self) -> None:
        if not self.cache_enabled:
            return
        with _RERANK_CACHE_SCHEMA_LOCK:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with sqlite3.connect(str(self.cache_path), timeout=30) as conn:
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS rerank_cache (
                        cache_key TEXT PRIMARY KEY,
                        prompt_version TEXT NOT NULL,
                        model_config_path TEXT NOT NULL,
                        candidate_count INTEGER NOT NULL,
                        scores_json TEXT NOT NULL,
                        status TEXT NOT NULL,
                        created_at REAL NOT NULL
                    )
                    """
                )
                conn.commit()

    def _cache_key(self, query: str, candidates: list[dict[str, Any]]) -> str:
        payload = {
            "version": _RERANK_PROMPT_VERSION,
            "model_config_path": str(self.model_config_path),
            "query": str(query or ""),
            "candidates": [
                {
                    "paragraph_id": idx,
                    "entity_name": str(item.get("entity_name", "") or ""),
                    "source_url": str(item.get("source_url", "") or ""),
                    "source_csv": str(item.get("source_csv", "") or ""),
                    "match_branches": [str(value) for value in (item.get("match_branches") or [])],
                    "chunk_text": str(item.get("chunk_text", "") or ""),
                }
                for idx, item in enumerate(candidates, start=1)
            ],
        }
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _read_cache(self, cache_key: str, expected_count: int) -> list[int] | None:
        if not self.cache_enabled:
            return None
        try:
            self._ensure_cache_schema()
            with sqlite3.connect(str(self.cache_path), timeout=30) as conn:
                row = conn.execute(
                    "SELECT scores_json, candidate_count FROM rerank_cache WHERE cache_key = ?",
                    (cache_key,),
                ).fetchone()
            if not row:
                return None
            if int(row[1] or 0) != int(expected_count):
                return None
            scores = json.loads(str(row[0] or "[]"))
            if not isinstance(scores, list) or len(scores) != expected_count:
                return None
            parsed = [int(item) for item in scores]
            if all(score in {-1, 0, 1, 2} for score in parsed):
                return parsed
        except Exception:
            return None
        return None

    def _write_cache(self, cache_key: str, scores: list[int], *, status: str) -> None:
        if not self.cache_enabled:
            return
        try:
            self._ensure_cache_schema()
            with sqlite3.connect(str(self.cache_path), timeout=30) as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO rerank_cache (
                        cache_key, prompt_version, model_config_path, candidate_count,
                        scores_json, status, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        cache_key,
                        _RERANK_PROMPT_VERSION,
                        str(self.model_config_path),
                        len(scores),
                        json.dumps([int(score) for score in scores], separators=(",", ":")),
                        str(status or "model_scored"),
                        time.time(),
                    ),
                )
                conn.commit()
        except Exception:
            return

    def _ensure_client(self) -> DeepSeekClient | None:
        if not self.enabled:
            return None
        if self.client is not None:
            return self.client
        try:
            self.client = _get_shared_deepseek_client(self.model_config_path)
            self.model_name = str(self.client.model or "deepseek-v4-flash").strip()
        except Exception:
            self.client = None
        return self.client

    def _format_content(self, candidates: list[dict[str, Any]]) -> str:
        blocks: list[str] = []
        for idx, item in enumerate(candidates, start=1):
            metadata = " | ".join(
                part
                for part in (
                    f"entity={str(item.get('entity_name', '') or '').strip()}",
                    f"source={str(item.get('source_url', '') or '').strip()}",
                    f"source_csv={str(item.get('source_csv', '') or '').strip()}",
                    f"branches={','.join(str(value) for value in (item.get('match_branches') or []))}",
                )
                if part and not part.endswith("=")
            )
            blocks.append(
                "\n".join(
                    [
                        f"Paragraph {idx}:",
                        f"Metadata: {metadata}" if metadata else "Metadata: (none)",
                        str(item.get("chunk_text", "") or "").strip(),
                    ]
                )
            )
        return "\n\n".join(blocks)

    def _build_prompt(self, query: str, candidates: list[dict[str, Any]]) -> str:
        score_items = ",".join(
            f'{{"paragraph_id":{idx},"score":0}}'
            for idx in range(1, len(candidates) + 1)
        )
        prompt_temp = """## Instructions
You are reranking local medical evidence for a fixed-answer benchmark question.
Judge whether each paragraph can help answer the exact visible query, not whether
it is generally medically interesting.

Use this scoring rubric:
- -1: likely misleading for this query: wrong drug/entity, wrong population,
  wrong route/timing/dose/setting, unsupported answer-like hint, or contradicts
  the visible decision boundary.
- 0: irrelevant or only shares broad words with the query.
- 1: relevant background/entity evidence, but it does not directly distinguish
  the leading answer choices or the patient-specific decision boundary.
- 2: directly useful evidence: it mentions the exact candidate action/drug,
  parameter, mechanism, diagnosis, route, timing, or guideline boundary needed
  for this stem, preferably with a patient anchor or an authoritative source.

Prefer guideline, official label/regulator, trial registry, or peer-reviewed
source text when it directly matches the stem. Do not give score 2 to a broad
disease page, uninspected snippet-like text, or partial local support atom unless
it independently contains the exact discriminating fact and source context.

## Query
{query}

## Paragraphs requiring judgment
{content}

Your output must be valid JSON in the following format:
{{"scores":[{score_items}]}}

Return exactly one score for each paragraph in ascending paragraph_id order. The score must be one of -1, 0, 1 or 2.
"""
        return prompt_temp.format(
            query=query,
            content=self._format_content(candidates),
            score_items=score_items,
        )

    def _parse_scores(self, text: str, expected_count: int) -> list[int] | None:
        cleaned = _strip_code_fences(text)
        if not cleaned:
            return None

        def _loads_jsonish(value: str) -> Any:
            try:
                return json.loads(value)
            except Exception:
                pass
            start = value.find("{")
            end = value.rfind("}")
            if 0 <= start < end:
                try:
                    return json.loads(value[start : end + 1])
                except Exception:
                    pass
            start = value.find("[")
            end = value.rfind("]")
            if 0 <= start < end:
                try:
                    return json.loads(value[start : end + 1])
                except Exception:
                    pass
            return None

        try:
            payload = json.loads(cleaned)
        except Exception:
            payload = _loads_jsonish(cleaned)

        if isinstance(payload, dict):
            raw_scores = payload.get("scores")
            if isinstance(raw_scores, list) and len(raw_scores) == expected_count:
                parsed: list[int] = []
                for idx, item in enumerate(raw_scores, start=1):
                    if not isinstance(item, dict):
                        return None
                    paragraph_id = item.get("paragraph_id")
                    score = item.get("score")
                    if int(paragraph_id) != idx:
                        return None
                    if int(score) not in {-1, 0, 1, 2}:
                        return None
                    parsed.append(int(score))
                return parsed

        if isinstance(payload, list) and len(payload) == expected_count:
            try:
                parsed = [int(item) for item in payload]
            except Exception:
                return None
            if all(score in {-1, 0, 1, 2} for score in parsed):
                return parsed

        pairs = re.findall(
            r'"?paragraph_id"?\s*:\s*(\d+)\D+?"?score"?\s*:\s*(-?\d+)',
            cleaned,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if len(pairs) == expected_count:
            parsed_by_id: dict[int, int] = {}
            for paragraph_id_raw, score_raw in pairs:
                try:
                    paragraph_id = int(paragraph_id_raw)
                    score = int(score_raw)
                except Exception:
                    return None
                if paragraph_id < 1 or paragraph_id > expected_count or score not in {-1, 0, 1, 2}:
                    return None
                parsed_by_id[paragraph_id] = score
            if len(parsed_by_id) == expected_count:
                return [parsed_by_id[idx] for idx in range(1, expected_count + 1)]
        return None

    def _fallback_scores(self, query: str, candidates: list[dict[str, Any]]) -> list[int]:
        del query
        scores: list[int] = []
        for idx, item in enumerate(candidates):
            branches = set(item.get("match_branches") or [])
            if idx == 0 and {"dense", "bm25"}.issubset(branches):
                scores.append(2)
            elif {"dense", "bm25"}.issubset(branches):
                scores.append(1)
            elif idx == 0:
                scores.append(1)
            else:
                scores.append(0)
        return scores

    @staticmethod
    def _is_local_support_atom(candidate: dict[str, Any]) -> bool:
        source_url = str(candidate.get("source_url", "") or "").casefold()
        source_csv = str(candidate.get("source_csv", "") or "").casefold()
        benchmark = str(candidate.get("benchmark", "") or "").casefold()
        return (
            source_url.startswith("local://support-atom/")
            or "support_atoms" in source_csv
            or benchmark.startswith("support_atoms")
        )

    def _effective_score(self, candidate: dict[str, Any], score: int) -> int:
        if self.support_atom_max_rerank_score is None:
            return int(score)
        if self._is_local_support_atom(candidate):
            return min(int(score), int(self.support_atom_max_rerank_score))
        return int(score)

    def _score_batch(self, query: str, candidates: list[dict[str, Any]]) -> dict[str, Any]:
        cache_key = self._cache_key(query=query, candidates=candidates)
        cached_scores = self._read_cache(cache_key, expected_count=len(candidates))
        if cached_scores is not None:
            return {
                "scores": cached_scores,
                "status": "model_scored_cache_hit",
                "attempts": 0,
                "raw_responses": [],
                "cache_hit": True,
            }

        client = self._ensure_client()
        if client is None:
            return {
                "scores": self._fallback_scores(query, candidates),
                "status": "fallback_scored_client_unavailable",
                "attempts": 0,
                "raw_responses": [],
                "cache_hit": False,
            }
        prompt = self._build_prompt(query=query, candidates=candidates)
        messages: list[dict[str, str]] = [
            {
                "role": "system",
                "content": (
                    "You are a medical retrieval reranker. "
                    "Return strict JSON only with one score per paragraph."
                ),
            },
            {"role": "user", "content": prompt},
        ]
        raw_responses: list[str] = []

        for attempt in range(self.max_retries):
            try:
                with _RERANK_SEMAPHORE:
                    with _acquire_cross_process_rerank_slot():
                        raw_text = client.chat_json(
                            messages,
                            max_tokens=400,
                            model_name=self.model_name,
                            thinking_enabled=False,
                        )
            except Exception as exc:
                error_text = f"{type(exc).__name__}: {exc}"
                raw_responses.append(error_text[:1000])
                if self.fallback_on_error:
                    return {
                        "scores": self._fallback_scores(query, candidates),
                        "status": "fallback_scored_provider_error",
                        "attempts": attempt + 1,
                        "raw_responses": raw_responses,
                        "cache_hit": False,
                        "error": error_text[:1000],
                    }
                raise
            raw_responses.append(raw_text)
            parsed = self._parse_scores(raw_text, expected_count=len(candidates))
            if parsed is not None:
                self._write_cache(cache_key, parsed, status="model_scored")
                return {
                    "scores": parsed,
                    "status": "model_scored",
                    "attempts": attempt + 1,
                    "raw_responses": raw_responses,
                    "cache_hit": False,
                }

            messages.extend(
                [
                    {"role": "assistant", "content": raw_text},
                    {
                        "role": "user",
                        "content": (
                            "Your previous output was invalid. Return only valid JSON with one score for each paragraph "
                            "in ascending paragraph_id order and with score values limited to -1, 0, 1, or 2."
                        ),
                    },
                ]
            )

        fallback_scores = self._fallback_scores(query, candidates)
        if self.cache_fallback_scores:
            self._write_cache(cache_key, fallback_scores, status="fallback_scored_parse_failure")
        return {
            "scores": fallback_scores,
            "status": "fallback_scored",
            "attempts": self.max_retries,
            "raw_responses": raw_responses,
            "cache_hit": False,
        }

    def rerank(self, query: str, candidates: list[dict[str, Any]]) -> dict[str, Any]:
        if not self.enabled or not candidates:
            return {
                "status": "disabled",
                "candidate_count": len(candidates),
                "kept_count": len(candidates),
                "results": [
                    {
                        "paragraph_id": idx,
                        "score": int(candidate.get("hybrid_score", 0) > 0),
                        "status": "disabled",
                        "candidate": candidate,
                    }
                    for idx, candidate in enumerate(candidates, start=1)
                ],
                "filtered_candidates": list(candidates),
            }

        truncated = list(candidates[: self.candidate_k])
        batches = [truncated[i : i + self.batch_size] for i in range(0, len(truncated), self.batch_size)]

        if len(batches) == 1:
            batch_results = [self._score_batch(query=query, candidates=batches[0])]
        else:
            recorder = get_active_timing_recorder()
            def _score_with_recorder(batch: list[dict[str, Any]]) -> dict[str, Any]:
                if recorder is None:
                    return self._score_batch(query=query, candidates=batch)
                with bind_timing_recorder(recorder):
                    return self._score_batch(query=query, candidates=batch)
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                batch_results = list(executor.map(_score_with_recorder, batches))

        results: list[dict[str, Any]] = []
        filtered_candidates: list[dict[str, Any]] = []
        paragraph_id = 1
        overall_status = "model_scored"
        cache_hits = 0
        for batch, batch_result in zip(batches, batch_results, strict=True):
            batch_status = str(batch_result.get("status", "fallback_scored"))
            if bool(batch_result.get("cache_hit", False)):
                cache_hits += 1
                if overall_status == "model_scored":
                    overall_status = "model_scored_cache_hit"
            if batch_status not in {"model_scored", "model_scored_cache_hit"}:
                overall_status = "fallback_scored"
            scores = list(batch_result.get("scores") or [])
            for candidate, score in zip(batch, scores, strict=True):
                effective_score = self._effective_score(candidate, int(score))
                scored_candidate = {
                    **candidate,
                    "rerank_score": int(effective_score),
                    "rerank_score_raw": int(score),
                    "rerank_status": batch_status,
                }
                if effective_score != int(score):
                    scored_candidate["rerank_status"] = f"{batch_status}_support_atom_capped"
                results.append(
                    {
                        "paragraph_id": paragraph_id,
                        "score": int(effective_score),
                        "raw_score": int(score),
                        "status": scored_candidate["rerank_status"],
                        "candidate": scored_candidate,
                    }
                )
                if int(effective_score) in self.keep_scores:
                    filtered_candidates.append(scored_candidate)
                paragraph_id += 1

        if self.always_keep_hybrid_top_k > 0:
            seen_doc_ids = {int(item.get("doc_id", 0) or 0) for item in filtered_candidates}
            for candidate in truncated[: self.always_keep_hybrid_top_k]:
                doc_id = int(candidate.get("doc_id", 0) or 0)
                if doc_id in seen_doc_ids:
                    continue
                filtered_candidates.append(
                    {
                        **candidate,
                        "rerank_score": 1,
                        "rerank_status": "hybrid_topk_forced_keep",
                    }
                )
                seen_doc_ids.add(doc_id)

        filtered_candidates.sort(
            key=lambda item: (
                int(item.get("rerank_score", -1) or -1),
                float(item.get("hybrid_score", 0.0) or 0.0),
            ),
            reverse=True,
        )
        empty_filter_fallback = False
        if not filtered_candidates and truncated:
            empty_filter_fallback = True
            fallback_candidate = {**truncated[0], "rerank_score": 0, "rerank_status": "empty_filter_top1_fallback"}
            filtered_candidates = [fallback_candidate]
            overall_status = "empty_filter_top1_fallback"

        return {
            "status": overall_status,
            "candidate_count": len(truncated),
            "kept_count": len(filtered_candidates),
            "results": results,
            "filtered_candidates": filtered_candidates,
            "max_workers": self.max_workers,
            "batch_size": self.batch_size,
            "cache_hits": cache_hits,
            "cache_path": str(self.cache_path) if self.cache_enabled else "",
            "always_keep_hybrid_top_k": self.always_keep_hybrid_top_k,
            "empty_filter_fallback": empty_filter_fallback,
        }


def _shared_client_cache_key(config_path: Path) -> str:
    return str(config_path.resolve())


def _get_shared_deepseek_client(config_path: Path) -> DeepSeekClient:
    cache_key = _shared_client_cache_key(config_path)
    with _RERANK_CLIENTS_LOCK:
        client = _RERANK_CLIENTS.get(cache_key)
        if client is None:
            client = DeepSeekClient(config_path=str(config_path))
            _RERANK_CLIENTS[cache_key] = client
        return client


@contextmanager
def _acquire_cross_process_rerank_slot(max_slots: int = _GLOBAL_RERANK_CONCURRENCY):
    import fcntl

    _RERANK_SLOT_DIR.mkdir(parents=True, exist_ok=True)
    acquired_file = None
    while acquired_file is None:
        for slot_index in range(max_slots):
            slot_path = _RERANK_SLOT_DIR / f"slot_{slot_index:02d}.lock"
            handle = slot_path.open("a+b")
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired_file = handle
                break
            except BlockingIOError:
                handle.close()
        if acquired_file is None:
            time.sleep(0.05)
    try:
        yield
    finally:
        try:
            fcntl.flock(acquired_file.fileno(), fcntl.LOCK_UN)
        finally:
            acquired_file.close()
