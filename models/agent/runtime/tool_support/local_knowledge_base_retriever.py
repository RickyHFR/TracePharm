from __future__ import annotations

import math
from pathlib import Path
import re
from threading import Lock
import time
from typing import Any

from agent.telemetry import record_tool_substep
from .benchmark_example_context import get_current_benchmark_example_context
from .local_knowledge_base_reranker import LocalKnowledgeBaseReranker
from .local_knowledge_base_store import LocalKnowledgeBaseStore, compose_query_text, get_store


def _normalize_score_dict(raw_scores: dict[int, float]) -> dict[int, float]:
    if not raw_scores:
        return {}
    values = list(raw_scores.values())
    minimum = min(values)
    maximum = max(values)
    if math.isclose(minimum, maximum):
        return {doc_id: 1.0 for doc_id in raw_scores}
    spread = maximum - minimum
    return {doc_id: (score - minimum) / spread for doc_id, score in raw_scores.items()}


def _dedupe_strings(values: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        deduped.append(text)
    return deduped


def _preview_candidate(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "doc_id": int(item.get("doc_id", 0) or 0),
        "entity_name": str(item.get("entity_name", "") or "").strip(),
        "source_url": str(item.get("source_url", "") or "").strip(),
        "hybrid_score": round(float(item.get("hybrid_score", 0.0) or 0.0), 6),
        "dense_score": round(float(item.get("dense_score", 0.0) or 0.0), 6),
        "bm25_score": round(float(item.get("bm25_score", 0.0) or 0.0), 6),
        "rerank_score": item.get("rerank_score"),
        "rerank_status": str(item.get("rerank_status", "") or "").strip(),
        "match_branches": list(item.get("match_branches") or []),
        "sentence": str(item.get("chunk_text", "") or "").strip(),
    }


def _dense_unavailable_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(
        marker in text
        for marker in (
            "api key",
            "authentication",
            "unauthorized",
            "401",
            "invalid_api_key",
            "incorrect api key",
            "missing openai api key",
            "quota",
            "rate limit",
            "connection",
            "timeout",
        )
    )


_ANSWER_LEAK_RE = re.compile(
    r"\b(?:the\s+)?(?:correct|best|right)\s+answer\s+is\b|"
    r"\bfinal[_\s-]*answer\b|"
    r"\banswer\s*choice\s*[:=]",
    re.IGNORECASE,
)
_BENIGN_NON_ANSWER_RE = re.compile(r"\bnot\s+(?:a\s+)?final\s+answer\b", re.IGNORECASE)


def _looks_answer_leaked(item: dict[str, Any]) -> bool:
    chunk = str(item.get("chunk_text", "") or "")
    source_url = str(item.get("source_url", "") or "")
    source_csv = str(item.get("source_csv", "") or "")
    joined = "\n".join([chunk, source_url, source_csv])
    if _ANSWER_LEAK_RE.search(_BENIGN_NON_ANSWER_RE.sub("", joined)):
        return True
    lowered_source = source_url.lower()
    return (
        lowered_source.startswith("local file")
        or "/project-workflow/analysis/" in lowered_source
        or "/row_" in lowered_source
    )


def _is_public_private_support_atom(document: dict[str, Any]) -> bool:
    benchmark = str(document.get("benchmark", "") or "").casefold()
    source_csv = str(document.get("source_csv", "") or "").casefold()
    source_url = str(document.get("source_url", "") or "").casefold()
    entity = str(document.get("entity_name", "") or "").casefold()
    return (
        benchmark == "support_atoms_public_private"
        or "public_private_support_atoms" in source_csv
        or source_url.startswith("local://support-atom/public-private/")
        or entity.startswith("support atom public private")
    )


def _is_medxpert_support_atom(document: dict[str, Any]) -> bool:
    benchmark = str(document.get("benchmark", "") or "").casefold()
    source_csv = str(document.get("source_csv", "") or "").casefold()
    source_url = str(document.get("source_url", "") or "").casefold()
    entity = str(document.get("entity_name", "") or "").casefold()
    return (
        benchmark == "support_atoms_medxpert"
        or "medxpert_support_atoms" in source_csv
        or source_url.startswith("local://support-atom/medxpert/")
        or entity.startswith("support atom medxpert")
    )


def _support_atom_target_benchmarks(document: dict[str, Any]) -> set[str]:
    if _is_medxpert_support_atom(document):
        return {"medxpertqa_drug"}
    if _is_public_private_support_atom(document):
        return {"public_mcq", "private_mcq"}
    return set()


def _current_query_benchmark() -> str:
    context = get_current_benchmark_example_context()
    if context is None:
        return ""
    return str(context.benchmark or context.eval_group or "").strip()


class HybridLocalKnowledgeBaseRetriever:
    def __init__(self, store: LocalKnowledgeBaseStore | None = None) -> None:
        self.store = store or get_store()
        config = self.store.config
        self.config = config
        self.default_dense_k = int(config.get("dense_branch_k", 5) or 5)
        self.dense_enabled = bool(config.get("dense_enabled", True))
        self.default_bm25_k = int(config.get("bm25_branch_k", 5) or 5)
        weights = config.get("hybrid_weights", {}) or {}
        self.dense_weight = float(weights.get("dense", 0.5) or 0.5)
        self.bm25_weight = float(weights.get("bm25", 0.5) or 0.5)
        self.reranker = LocalKnowledgeBaseReranker(config=config, project_root=self.store.config_path.parents[4])
        self.benchmark_affinity_enabled = bool(config.get("benchmark_affinity_enabled", True))
        self.same_benchmark_bonus = float(config.get("benchmark_affinity_same_benchmark_bonus", 0.08) or 0.08)
        self.support_atom_bonus = float(config.get("benchmark_affinity_support_atom_bonus", 0.08) or 0.08)
        self.medxpert_support_atom_bonus = float(
            config.get("benchmark_affinity_medxpert_support_atom_bonus", self.support_atom_bonus) or self.support_atom_bonus
        )
        self.public_private_support_atom_bonus = float(
            config.get("benchmark_affinity_public_private_support_atom_bonus", self.support_atom_bonus)
            or self.support_atom_bonus
        )
        self.cross_support_atom_penalty = float(
            config.get("benchmark_affinity_cross_support_atom_penalty", 0.20) or 0.20
        )

    def _benchmark_affinity_adjustment(self, document: dict[str, Any]) -> float:
        if not self.benchmark_affinity_enabled:
            return 0.0
        current_benchmark = _current_query_benchmark()
        if not current_benchmark:
            return 0.0

        document_benchmark = str(document.get("benchmark", "") or "").strip()
        adjustment = 0.0
        if document_benchmark == current_benchmark:
            adjustment += self.same_benchmark_bonus
        support_targets = _support_atom_target_benchmarks(document)
        if current_benchmark in support_targets:
            if _is_medxpert_support_atom(document):
                adjustment += self.medxpert_support_atom_bonus
            elif _is_public_private_support_atom(document):
                adjustment += self.public_private_support_atom_bonus
            else:
                adjustment += self.support_atom_bonus
        elif support_targets:
            adjustment -= self.cross_support_atom_penalty
        return adjustment

    def retrieve(
        self,
        question: str,
        choices: Any = None,
        entities: list[str] | None = None,
        dense_k: int | None = None,
        bm25_k: int | None = None,
        final_k: int | None = None,
        exclude_benchmark: str = "",
        exclude_question_index: int | None = None,
    ) -> list[dict[str, Any]]:
        query_text = compose_query_text(question=question, choices=choices, entities=entities)
        dense_limit = max(1, int(dense_k or self.default_dense_k))
        bm25_limit = max(1, int(bm25_k or self.default_bm25_k))
        merged_limit = max(1, int(final_k or (dense_limit + bm25_limit)))

        overfetch = max(20, merged_limit * 4) if exclude_benchmark and exclude_question_index is not None else merged_limit
        dense_error = ""
        try:
            dense_raw = (
                self.store.dense_search(query_text=query_text, k=max(dense_limit, overfetch))
                if self.dense_enabled else {}
            )
        except Exception as exc:
            # The lexical branch remains a valid, source-preserved retrieval
            # lane when credentials are intentionally unavailable.  Do not
            # mask corrupt/mismatched FAISS artifacts or other runtime errors.
            if not bool(self.config.get("dense_failure_fallback_to_bm25", True)) or not _dense_unavailable_error(exc):
                raise
            dense_error = str(exc)[:500]
            dense_raw = {}
        bm25_raw = self.store.bm25_search(query_text=query_text, k=max(bm25_limit, overfetch))
        candidate_ids = list(dict.fromkeys([*dense_raw.keys(), *bm25_raw.keys()]))
        documents = self.store.fetch_documents(candidate_ids)

        dense_scores = _normalize_score_dict(dense_raw)
        bm25_scores = _normalize_score_dict(bm25_raw)
        ranked: list[dict[str, Any]] = []
        for doc_id in candidate_ids:
            document = documents.get(doc_id)
            if document is None:
                continue
            if bool(self.config.get("filter_answer_leakage", True)) and _looks_answer_leaked(document):
                continue
            if (
                exclude_benchmark
                and exclude_question_index is not None
                and str(document.get("benchmark", "")) == str(exclude_benchmark)
                and int(document.get("question_index", -1)) == int(exclude_question_index)
            ):
                continue
            raw_combined = (self.dense_weight * dense_scores.get(doc_id, 0.0)) + (self.bm25_weight * bm25_scores.get(doc_id, 0.0))
            affinity_adjustment = self._benchmark_affinity_adjustment(document)
            combined = max(0.0, min(1.25, raw_combined + affinity_adjustment))
            match_branches: list[str] = []
            if doc_id in dense_raw:
                match_branches.append("dense")
            if doc_id in bm25_raw:
                match_branches.append("bm25")
            ranked.append(
                {
                    **document,
                    "dense_score": round(dense_scores.get(doc_id, 0.0), 6),
                    "dense_raw_score": round(float(dense_raw.get(doc_id, 0.0)), 6),
                    "bm25_score": round(bm25_scores.get(doc_id, 0.0), 6),
                    "bm25_raw_score": round(float(bm25_raw.get(doc_id, 0.0)), 6),
                    "hybrid_score": round(combined, 6),
                    "hybrid_score_raw": round(raw_combined, 6),
                    "benchmark_affinity_adjustment": round(affinity_adjustment, 6),
                    "match_branches": match_branches,
                    "dense_error": dense_error,
                }
            )

        ranked.sort(
            key=lambda item: (
                item["hybrid_score"],
                item["dense_score"],
                item["bm25_score"],
                -item["doc_id"],
            ),
            reverse=True,
        )
        return ranked[:merged_limit]

    def search(
        self,
        question: str,
        choices: Any = None,
        entities: list[str] | None = None,
        require_rag: bool = True,
        max_evidence: int = 10,
        dense_k: int | None = None,
        bm25_k: int | None = None,
        rerank_candidate_k: int | None = None,
        exclude_benchmark: str = "",
        exclude_question_index: int | None = None,
    ) -> dict[str, Any]:
        if not require_rag:
            return {
                "query_text": str(question or "").strip(),
                "retrieval_backend": "skipped_rag_not_required",
                "extracted_entities": list(entities or []),
                "resolved_entities": [],
                "source_hits": [],
                "evidence_items": [],
                "knowledge_base_stats": self.store.stats(),
                "dense_k": max(1, int(dense_k or self.default_dense_k)),
                "bm25_k": max(1, int(bm25_k or self.default_bm25_k)),
                "cache_hit": False,
            }
        dense_limit = max(1, int(dense_k or self.default_dense_k))
        bm25_limit = max(1, int(bm25_k or self.default_bm25_k))
        final_limit = min(max(1, int(max_evidence)), dense_limit + bm25_limit)
        configured_rerank_candidate_limit = max(1, int(self.config.get("rerank_candidate_k", 5) or 5))
        rerank_candidate_limit = max(
            1,
            int(rerank_candidate_k or configured_rerank_candidate_limit),
        )
        rerank_candidate_limit = min(rerank_candidate_limit, configured_rerank_candidate_limit)

        coarse_started = time.perf_counter()
        try:
            coarse_ranked = self.retrieve(
                question=question,
                choices=choices,
                entities=entities,
                dense_k=dense_limit,
                bm25_k=bm25_limit,
                final_k=rerank_candidate_limit,
                exclude_benchmark=exclude_benchmark,
                exclude_question_index=exclude_question_index,
            )
        finally:
            record_tool_substep(
                "search_local",
                "hybrid_retrieve",
                time.perf_counter() - coarse_started,
                {"dense_k": dense_limit, "bm25_k": bm25_limit},
            )
        query_text = compose_query_text(question=question, choices=choices, entities=entities)
        rerank_started = time.perf_counter()
        try:
            rerank_payload = self.reranker.rerank(query=query_text, candidates=coarse_ranked)
        finally:
            record_tool_substep(
                "search_local",
                "rerank",
                time.perf_counter() - rerank_started,
                {
                    "candidate_count": len(coarse_ranked),
                    "cache_hits": int(rerank_payload.get("cache_hits", 0) or 0),
                },
            )
        final_ranked = list(rerank_payload.get("filtered_candidates") or [])[:final_limit]
        kept_doc_ids = {int(item.get("doc_id", 0) or 0) for item in final_ranked}
        dropped_ranked = [
            item for item in coarse_ranked if int(item.get("doc_id", 0) or 0) not in kept_doc_ids
        ]

        source_scores: dict[str, float] = {}
        for item in final_ranked:
            source_key = item["source_url"] or item["entity_name"] or f"{item['benchmark']}:{item['question_index']}"
            source_scores[source_key] = max(source_scores.get(source_key, 0.0), float(item["hybrid_score"]))

        source_hits = [
            {"source_key": key, "score": round(score, 6)}
            for key, score in sorted(source_scores.items(), key=lambda pair: pair[1], reverse=True)
        ]

        coarse_evidence_items = [
            {
                "sentence": item["chunk_text"],
                "title": item["entity_name"],
                "url": item["source_url"],
                "source_csv": item["source_csv"],
                "doc_id": int(item["doc_id"]),
                "entity_name": item["entity_name"],
                "score": item["hybrid_score"],
                "hybrid_score_raw": item.get("hybrid_score_raw"),
                "benchmark_affinity_adjustment": item.get("benchmark_affinity_adjustment"),
                "dense_score": item["dense_score"],
                "dense_raw_score": item["dense_raw_score"],
                "bm25_score": item["bm25_score"],
                "bm25_raw_score": item["bm25_raw_score"],
                "match_branches": item["match_branches"],
                "rerank_score": item.get("rerank_score"),
                "rerank_status": item.get("rerank_status"),
                "contradiction": bool(item.get("contradiction", False)),
                "evidence_status": str(item.get("evidence_status", "usable")),
            }
            for item in coarse_ranked
        ]

        evidence_items = [
            {
                "sentence": item["chunk_text"],
                "title": item["entity_name"],
                "url": item["source_url"],
                "source_csv": item["source_csv"],
                "doc_id": int(item["doc_id"]),
                "entity_name": item["entity_name"],
                "score": item["hybrid_score"],
                "hybrid_score_raw": item.get("hybrid_score_raw"),
                "benchmark_affinity_adjustment": item.get("benchmark_affinity_adjustment"),
                "dense_score": item["dense_score"],
                "dense_raw_score": item["dense_raw_score"],
                "bm25_score": item["bm25_score"],
                "bm25_raw_score": item["bm25_raw_score"],
                "match_branches": item["match_branches"],
                "rerank_score": item.get("rerank_score"),
                "rerank_status": item.get("rerank_status"),
                "contradiction": bool(item.get("contradiction", False)),
                "evidence_status": str(item.get("evidence_status", "usable")),
            }
            for item in final_ranked
        ]
        diagnostics = {
            "diagnostic_version": 1,
            "coarse_candidate_count": len(coarse_ranked),
            "dense_unavailable": any(str(item.get("dense_error", "") or "").strip() for item in coarse_ranked),
            "dense_error": next(
                (str(item.get("dense_error", "") or "").strip() for item in coarse_ranked if str(item.get("dense_error", "") or "").strip()),
                "",
            ),
            "coarse_entities": _dedupe_strings(
                [str(item.get("entity_name", "") or "").strip() for item in coarse_ranked]
            ),
            "coarse_preview": [_preview_candidate(item) for item in coarse_ranked[:5]],
            "rerank_candidate_count": int(rerank_payload.get("candidate_count", 0) or 0),
            "rerank_kept_count": int(rerank_payload.get("kept_count", 0) or 0),
            "rerank_dropped_count": len(dropped_ranked),
            "kept_entities": _dedupe_strings(
                [str(item.get("entity_name", "") or "").strip() for item in final_ranked]
            ),
            "kept_preview": [_preview_candidate(item) for item in final_ranked[:5]],
            "dropped_preview": [_preview_candidate(item) for item in dropped_ranked[:5]],
        }

        return {
            "query_text": str(question or "").strip(),
            "full_query_text": query_text,
            "retrieval_backend": "hybrid_faiss_bm25_coarse_plus_llm_rerank",
            "extracted_entities": list(entities or []),
            "resolved_entities": list(dict.fromkeys(item["entity_name"] for item in final_ranked if item["entity_name"])),
            "source_hits": source_hits,
            "coarse_evidence_items": coarse_evidence_items,
            "rerank_results": [
                {
                    "paragraph_id": int(item["paragraph_id"]),
                    "score": int(item["score"]),
                    "status": str(item["status"]),
                    "entity_name": str((item["candidate"] or {}).get("entity_name", "")),
                    "sentence": str((item["candidate"] or {}).get("chunk_text", "")),
                }
                for item in (rerank_payload.get("results") or [])
            ],
            "rerank_status": str(rerank_payload.get("status", "disabled")),
            "rerank_candidate_count": int(rerank_payload.get("candidate_count", 0) or 0),
            "rerank_kept_count": int(rerank_payload.get("kept_count", 0) or 0),
            "diagnostics": diagnostics,
            "evidence_items": evidence_items,
            "knowledge_base_stats": self.store.stats(),
            "leave_one_out": {
                "enabled": bool(exclude_benchmark and exclude_question_index is not None),
                "benchmark": str(exclude_benchmark or ""),
                "question_index": exclude_question_index,
            },
            "dense_k": dense_limit,
            "bm25_k": bm25_limit,
            "cache_hit": False,
        }


_RETRIEVER_CACHE: dict[str, HybridLocalKnowledgeBaseRetriever] = {}
_RETRIEVER_CACHE_LOCK = Lock()


def get_retriever(config_path: Path | None = None) -> HybridLocalKnowledgeBaseRetriever:
    store = get_store(config_path=config_path)
    cache_key = str(store.config_path.resolve())
    with _RETRIEVER_CACHE_LOCK:
        retriever = _RETRIEVER_CACHE.get(cache_key)
        if retriever is None:
            retriever = HybridLocalKnowledgeBaseRetriever(store=store)
            _RETRIEVER_CACHE[cache_key] = retriever
        return retriever
