from __future__ import annotations

import csv
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any

from .base_tool import FunctionTool
from tool_support.benchmark_example_context import get_current_benchmark_example_context
from tool_support.local_knowledge_base_retriever import get_retriever


PROJECT_ROOT = Path(__file__).resolve().parents[4]


def _summarize_preview_item(item: dict[str, Any]) -> str:
    entity = str(item.get("entity_name", "") or "").strip()
    score = item.get("hybrid_score")
    rerank_score = item.get("rerank_score")
    match_branches = list(item.get("match_branches") or [])
    parts: list[str] = []
    if entity:
        parts.append(entity)
    if score is not None:
        parts.append(f"hybrid={score}")
    if rerank_score is not None and str(rerank_score).strip():
        parts.append(f"rerank={rerank_score}")
    if match_branches:
        parts.append("branches=" + "+".join(str(branch) for branch in match_branches))
    sentence = str(item.get("sentence", "") or "").strip()
    if sentence:
        trimmed = sentence if len(sentence) <= 140 else sentence[:137].rstrip() + "..."
        parts.append(f"\"{trimmed}\"")
    return " | ".join(part for part in parts if part)


def _render_diagnostics(payload: dict[str, Any]) -> list[str]:
    diagnostics = payload.get("diagnostics")
    if not isinstance(diagnostics, dict):
        return []

    lines = ["Retrieval diagnostics:"]
    coarse_candidate_count = int(diagnostics.get("coarse_candidate_count", 0) or 0)
    rerank_candidate_count = int(diagnostics.get("rerank_candidate_count", 0) or 0)
    rerank_kept_count = int(diagnostics.get("rerank_kept_count", 0) or 0)
    rerank_dropped_count = int(diagnostics.get("rerank_dropped_count", 0) or 0)
    lines.append(
        f"- coarse candidates: {coarse_candidate_count}; rerank examined: {rerank_candidate_count}; kept: {rerank_kept_count}; dropped: {rerank_dropped_count}"
    )

    coarse_entities = diagnostics.get("coarse_entities") or []
    if coarse_entities:
        lines.append("Coarse entities: " + ", ".join(str(item) for item in coarse_entities[:8]))

    kept_entities = diagnostics.get("kept_entities") or []
    if kept_entities:
        lines.append("Kept entities: " + ", ".join(str(item) for item in kept_entities[:8]))

    coarse_preview = diagnostics.get("coarse_preview") or []
    if coarse_preview:
        lines.append("Coarse top candidates:")
        for index, item in enumerate(coarse_preview[:3], start=1):
            if isinstance(item, dict):
                lines.append(f"  {index}. {_summarize_preview_item(item)}")

    kept_preview = diagnostics.get("kept_preview") or []
    if kept_preview:
        lines.append("Kept after rerank:")
        for index, item in enumerate(kept_preview[:3], start=1):
            if isinstance(item, dict):
                lines.append(f"  {index}. {_summarize_preview_item(item)}")

    dropped_preview = diagnostics.get("dropped_preview") or []
    if dropped_preview:
        lines.append("Dropped by rerank:")
        for index, item in enumerate(dropped_preview[:3], start=1):
            if isinstance(item, dict):
                lines.append(f"  {index}. {_summarize_preview_item(item)}")

    return lines


def _render_context(payload: dict[str, Any]) -> str:
    lines: list[str] = []
    include_diagnostics = str(os.environ.get("SEARCH_LOCAL_RENDER_DIAGNOSTICS", "") or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    queries_executed = payload.get("queries_executed") or []
    if queries_executed:
        lines.append("Queries executed: " + " || ".join(str(item) for item in queries_executed[:6]))
    extracted_entities = payload.get("extracted_entities") or []
    resolved_entities = payload.get("resolved_entities") or []
    if extracted_entities:
        lines.append("Extracted entities: " + ", ".join(str(item) for item in extracted_entities[:12]))
    if resolved_entities:
        lines.append("Resolved entities: " + ", ".join(str(item) for item in resolved_entities[:12]))
    source_hits = payload.get("source_hits") or []
    if source_hits:
        lines.append(
            "Matched source groups: "
            + ", ".join(f"{item['source_key']} ({item['score']})" for item in source_hits[:5] if isinstance(item, dict))
        )
    evidence_items = payload.get("evidence_items") or []
    if evidence_items:
        lines.append("Evidence sentences:")
        for item in evidence_items:
            if not isinstance(item, dict):
                continue
            sentence = str(item.get("sentence", "") or "").strip()
            title = str(item.get("title", "") or "").strip()
            url = str(item.get("url", "") or "").strip()
            prefix = " | ".join(part for part in [title, url] if part)
            lines.append(f"- {prefix} | {sentence}" if prefix else f"- {sentence}")
    if include_diagnostics:
        lines.extend(_render_diagnostics(payload))
    if not lines:
        return "No local evidence was found."
    return "\n".join(lines)


def _dedupe_strings(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def _resolve_project_path(raw_path: str) -> Path:
    path = Path(str(raw_path or ""))
    if path.is_absolute():
        return path.resolve()
    return (PROJECT_ROOT / path).resolve()


def _safe_json_loads(raw: Any, default: Any) -> Any:
    try:
        return json.loads(str(raw or ""))
    except Exception:
        return default


def _rerank_score_value(item: dict[str, Any]) -> int:
    raw_score = item.get("rerank_score")
    if raw_score is None or str(raw_score).strip() == "":
        return -2
    try:
        return int(raw_score)
    except Exception:
        return -2


def _evidence_rank_key(item: dict[str, Any]) -> tuple[int, int, float, float, float, int]:
    """Stable score used after multi-query union.

    Each subquery is already independently reranked.  When merging several
    subqueries, preserve that relevance judgment before falling back to hybrid
    retrieval scores so one high-quality targeted query is not drowned out by a
    broader query with larger lexical/dense scores.
    """
    rerank_score = _rerank_score_value(item)
    branch_bonus = 1 if str(item.get("rerank_status", "") or "").startswith("source_audit") else 0
    return (
        rerank_score,
        branch_bonus,
        float(item.get("score", 0.0) or 0.0),
        float(item.get("dense_score", 0.0) or 0.0),
        float(item.get("bm25_score", 0.0) or 0.0),
        -int(item.get("doc_id", 0) or 0),
    )


def _evidence_merge_key(item: dict[str, Any]) -> str:
    sentence = re.sub(
        r"\s+",
        " ",
        str(item.get("sentence") or item.get("chunk_text") or "").strip(),
    ).casefold()
    url = str(item.get("url") or item.get("source_url") or "").strip().casefold()
    if sentence:
        return "text:" + url + "|" + sentence[:800]
    return "doc:" + str(int(item.get("doc_id", 0) or 0))


def _merge_evidence_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        merge_key = _evidence_merge_key(item)
        existing = merged.get(merge_key)
        if existing is None or _evidence_rank_key(item) > _evidence_rank_key(existing):
            replacement = {**item}
            if existing is not None:
                branches = list(existing.get("match_branches") or [])
                for branch in list(item.get("match_branches") or []):
                    if branch not in branches:
                        branches.append(branch)
                replacement["match_branches"] = branches
            merged[merge_key] = replacement
            continue
        branches = list(existing.get("match_branches") or [])
        for branch in list(item.get("match_branches") or []):
            if branch not in branches:
                branches.append(branch)
        existing["match_branches"] = branches
    ranked = list(merged.values())
    ranked.sort(
        key=_evidence_rank_key,
        reverse=True,
    )
    return ranked


def _merge_multi_query_payloads(
    queries: list[str],
    payloads: list[dict[str, Any]],
    max_evidence: int,
    dense_k: int,
    bm25_k: int,
) -> dict[str, Any]:
    if not payloads:
        return {
            "query_text": "",
            "full_query_text": "",
            "queries_executed": [],
            "retrieval_backend": "hybrid_faiss_bm25_coarse_plus_llm_rerank_multiquery",
            "extracted_entities": [],
            "resolved_entities": [],
            "source_hits": [],
            "coarse_evidence_items": [],
            "rerank_results": [],
            "rerank_status": "multiquery_empty",
            "rerank_candidate_count": 0,
            "rerank_kept_count": 0,
            "diagnostics": {
                "diagnostic_version": 1,
                "query_mode": "multi",
                "query_count": 0,
                "coarse_candidate_count": 0,
                "rerank_candidate_count": 0,
                "rerank_kept_count": 0,
                "rerank_dropped_count": 0,
                "coarse_entities": [],
                "kept_entities": [],
                "coarse_preview": [],
                "kept_preview": [],
                "dropped_preview": [],
            },
            "evidence_items": [],
            "knowledge_base_stats": {},
            "dense_k": dense_k,
            "bm25_k": bm25_k,
            "cache_hit": False,
        }

    coarse_union = _merge_evidence_items(
        [item for payload in payloads for item in list(payload.get("coarse_evidence_items") or [])]
    )
    final_union = _merge_evidence_items(
        [item for payload in payloads for item in list(payload.get("evidence_items") or [])]
    )[: max(1, int(max_evidence))]
    kept_doc_ids = {int(item.get("doc_id", 0) or 0) for item in final_union}
    dropped_union = [item for item in coarse_union if int(item.get("doc_id", 0) or 0) not in kept_doc_ids]

    source_scores: dict[str, float] = {}
    for item in final_union:
        source_key = str(item.get("url", "") or "").strip() or str(item.get("title", "") or "").strip()
        if not source_key:
            source_key = f"{item.get('benchmark', '')}:{item.get('question_index', '')}"
        source_scores[source_key] = max(source_scores.get(source_key, 0.0), float(item.get("score", 0.0) or 0.0))
    source_hits = [
        {"source_key": key, "score": round(score, 6)}
        for key, score in sorted(source_scores.items(), key=lambda pair: pair[1], reverse=True)
    ]

    per_query_summaries = []
    for query, payload in zip(queries, payloads):
        diagnostics = payload.get("diagnostics") or {}
        per_query_summaries.append(
            {
                "query": query,
                "coarse_candidate_count": int(diagnostics.get("coarse_candidate_count", 0) or 0),
                "rerank_candidate_count": int(diagnostics.get("rerank_candidate_count", 0) or 0),
                "rerank_kept_count": int(diagnostics.get("rerank_kept_count", 0) or 0),
                "rerank_dropped_count": int(diagnostics.get("rerank_dropped_count", 0) or 0),
            }
        )

    return {
        "query_text": " || ".join(queries),
        "full_query_text": " || ".join(queries),
        "queries_executed": list(queries),
        "retrieval_backend": "hybrid_faiss_bm25_coarse_plus_llm_rerank_multiquery",
        "extracted_entities": [],
        "resolved_entities": _dedupe_strings([str(item.get("title", "") or "").strip() for item in final_union]),
        "source_hits": source_hits,
        "coarse_evidence_items": coarse_union,
        "rerank_results": [],
        "rerank_status": "per_query_union",
        "rerank_candidate_count": sum(int((payload.get("rerank_candidate_count", 0) or 0)) for payload in payloads),
        "rerank_kept_count": len(final_union),
        "diagnostics": {
            "diagnostic_version": 1,
            "query_mode": "multi",
            "query_count": len(queries),
            "per_query_summaries": per_query_summaries,
            "coarse_candidate_count": len(coarse_union),
            "rerank_candidate_count": sum(
                int((payload.get("rerank_candidate_count", 0) or 0)) for payload in payloads
            ),
            "rerank_kept_count": len(final_union),
            "rerank_dropped_count": len(dropped_union),
            "coarse_entities": _dedupe_strings([str(item.get("title", "") or "").strip() for item in coarse_union]),
            "kept_entities": _dedupe_strings([str(item.get("title", "") or "").strip() for item in final_union]),
            "coarse_preview": coarse_union[:5],
            "kept_preview": final_union[:5],
            "dropped_preview": dropped_union[:5],
        },
        "evidence_items": final_union,
        "knowledge_base_stats": payloads[0].get("knowledge_base_stats", {}),
        "dense_k": dense_k,
        "bm25_k": bm25_k,
        "cache_hit": False,
    }


def _choice_text(choices: Any) -> str:
    if isinstance(choices, dict):
        return "\n".join(f"{key}. {value}" for key, value in choices.items())
    if isinstance(choices, list):
        return "\n".join(str(item) for item in choices)
    return str(choices or "")


_ANCHOR_PATTERNS = [
    re.compile(
        r"\b(?:after|following|underwent|received|managed with|treated with|upon removal of|removal of)\b[^.;?\n]{0,180}",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:blood cultures? (?:grow|show|reveal)|urine cultures? (?:grow|show|reveal)|culture (?:grew|shows|reveals))\b[^.;?\n]{0,180}",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:physical examination reveals|examination reveals|laboratory (?:testing|studies) (?:show|reveal)|imaging (?:shows|reveals))\b[^.;?\n]{0,180}",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:reports?|complains? of|experiencing|develops?|symptoms? (?:include|began|started))\b[^.;?\n]{0,240}",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:current medications? include|medications? include|current medications? are|takes|taking)\b[^.;?\n]{0,220}",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bvital signs show\b[^?\n]{0,260}",
        re.IGNORECASE,
    ),
]

_ANCHOR_LEAD_TERMS = (
    "packing",
    "catheter",
    "central line",
    "peripheral line",
    "tube",
    "device",
    "foreign bod",
    "prosthe",
    "implant",
    "stent",
    "valve",
    "culture",
    "cultures",
    "grew",
    "echocardiography",
    "biopsy",
    "underwent",
    "received",
    "managed with",
    "treated with",
    "postoperative",
    "post-operative",
    "current medications",
    "medications include",
    "taking",
    "diarrhea",
    "vomiting",
    "nausea",
    "dehydration",
    "myalgia",
    "myalgias",
    "arthralgia",
    "arthralgias",
    "crepitus",
    "range of motion",
    "active and passive",
    "primidone",
    "methadone",
)


def _should_lead_with_anchor_context(anchor_query: str, planner_query: str) -> bool:
    anchor_l = str(anchor_query or "").casefold()
    planner_l = str(planner_query or "").casefold()
    if (
        "answer choices:" in anchor_l
        and "answer choices:" not in planner_l
        and len(re.findall(r"\b[A-J]\.", anchor_query)) >= 6
        and len(str(planner_query or "")) <= 240
    ):
        return True
    for term in _ANCHOR_LEAD_TERMS:
        if term in anchor_l and term not in planner_l:
            return True
    return False


def _build_anchor_choice_query(
    *,
    question: str,
    choices: Any,
    goal: str = "",
    max_chars: int = 900,
) -> str:
    """Compact, deterministic local-retrieval query built from visible context.

    The main planner can over-focus on one inferred diagnosis. This branch keeps
    unusual procedures, devices, cultures, labs, physical findings, and answer
    choices visible to retrieval without using benchmark labels or answers.
    """
    question_text = str(question or "").strip()
    anchors: list[str] = []
    for pattern in _ANCHOR_PATTERNS:
        for match in pattern.finditer(question_text):
            text = re.sub(r"\s+", " ", match.group(0)).strip(" ;,.")
            if text:
                anchors.append(text)
    anchors = _dedupe_strings(anchors)[:8]

    parts: list[str] = []
    if anchors:
        parts.append("Patient anchors: " + "; ".join(anchors))
    choice_block = _choice_text(choices).strip()
    if choice_block:
        compact_choices = re.sub(r"\s+", " ", choice_block)
        parts.append("Answer choices: " + compact_choices)
    if goal:
        parts.append("Decision goal: " + re.sub(r"\s+", " ", str(goal)).strip())
    query = "\n".join(part for part in parts if part.strip())
    if len(query) > max_chars:
        query = query[:max_chars].rsplit(" ", 1)[0].rstrip()
    return query


def _extract_visible_anchor_entities(question: str, *, limit: int = 8) -> list[str]:
    anchors: list[str] = []
    question_text = str(question or "").strip()
    for pattern in _ANCHOR_PATTERNS:
        for match in pattern.finditer(question_text):
            text = re.sub(r"\s+", " ", match.group(0)).strip(" ;,.")
            if text:
                anchors.append(text)
    return _dedupe_strings(anchors)[: max(0, int(limit))]


_SPECIFIC_ENTITY_LOW_SIGNAL_TOKENS = {
    "adult",
    "adults",
    "adolescent",
    "adolescents",
    "acute",
    "approximately",
    "boy",
    "bid",
    "child",
    "children",
    "clinical",
    "daily",
    "department",
    "dose",
    "dosing",
    "drug",
    "drugs",
    "emergency",
    "female",
    "girl",
    "guidance",
    "guideline",
    "guidelines",
    "history",
    "include",
    "infant",
    "infants",
    "kg",
    "male",
    "management",
    "matched",
    "matching",
    "medication",
    "medications",
    "mcg",
    "mg",
    "ml",
    "patient",
    "patients",
    "past",
    "pediatric",
    "paediatric",
    "this",
    "tablet",
    "tablets",
    "capsule",
    "capsules",
    "therapy",
    "treatment",
    "when",
    "which",
    "with",
    "without",
    "woman",
    "women",
    "year",
    "years",
}

_LOCAL_EVIDENCE_LOW_SIGNAL_TOKENS = _SPECIFIC_ENTITY_LOW_SIGNAL_TOKENS | {
    "anchor",
    "anchors",
    "atom",
    "branch",
    "candidate",
    "choice",
    "choices",
    "compare",
    "concept",
    "concepts",
    "context",
    "blood",
    "center",
    "appropriate",
    "decision",
    "differentiat",
    "differentiator",
    "examination",
    "evidence",
    "final",
    "focus",
    "heart",
    "hospitalized",
    "laboratory",
    "physical",
    "local",
    "matched",
    "medical",
    "mmhg",
    "most",
    "oxygen",
    "other",
    "pressure",
    "pulse",
    "question",
    "rate",
    "record",
    "regimen",
    "reveals",
    "respiratory",
    "room",
    "retrieval",
    "saturation",
    "search",
    "signs",
    "source",
    "support",
    "supports",
    "than",
    "that",
    "visible",
    "vital",
    "vitals",
    "verify",
}


def _meaningful_entity_tokens(entity: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9]+", str(entity or "").casefold())
    return [
        token
        for token in tokens
        if not token.isdigit()
        and token not in _SPECIFIC_ENTITY_LOW_SIGNAL_TOKENS
        and len(token) >= 4
    ]


def _meaningful_local_evidence_tokens(text: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9]+", str(text or "").casefold())
    return [
        token
        for token in tokens
        if not token.isdigit()
        and token not in _LOCAL_EVIDENCE_LOW_SIGNAL_TOKENS
        and len(token) >= 4
    ]


def _is_support_or_parameter_atom(item: dict[str, Any]) -> bool:
    haystack = " ".join(
        str(item.get(field) or "")
        for field in (
            "url",
            "title",
            "entity_name",
            "sentence",
            "source_csv",
            "local_branch",
            "rerank_status",
        )
    ).casefold()
    return (
        "local://support-atom/" in haystack
        or "support atom" in haystack
        or "decision-support atom" in haystack
        or "differentiator atom" in haystack
        or "parameter atom" in haystack
    )


def _support_atom_source_family(item: dict[str, Any]) -> str:
    haystack = " ".join(
        str(item.get(field) or "")
        for field in (
            "url",
            "title",
            "entity_name",
            "sentence",
            "source_csv",
            "local_branch",
        )
    ).casefold()
    if "local://support-atom/medxpert/" in haystack or "support_atoms_medxpert" in haystack or "medxpert_support_atoms" in haystack:
        return "medxpert"
    if (
        "local://support-atom/public-private/" in haystack
        or "support_atoms_public_private" in haystack
        or "public_private_support_atoms" in haystack
    ):
        return "public_private"
    return "generic"


def _support_atom_target_benchmarks(family: str) -> set[str]:
    if family == "medxpert":
        return {"medxpertqa_drug"}
    if family == "public_private":
        return {"public_mcq", "private_mcq", "public_cloze", "private_cloze"}
    return set()


def _current_local_benchmark() -> str:
    context = get_current_benchmark_example_context()
    if context is None:
        return ""
    return str(context.benchmark or context.eval_group or "").strip()


def _local_item_visible_overlap(item: dict[str, Any], query_text: str) -> set[str]:
    query_l = str(query_text or "").casefold()
    fields = [
        str(item.get("entity_name") or ""),
        str(item.get("title") or ""),
        str(item.get("sentence") or ""),
    ]
    tokens: set[str] = set()
    for field in fields:
        tokens.update(_meaningful_local_evidence_tokens(field))
    return {token for token in tokens if token and token in query_l}


def _has_specific_local_overlap(
    item: dict[str, Any],
    query_text: str,
    *,
    min_tokens: int = 2,
    min_long_token_length: int = 8,
) -> bool:
    overlap = _local_item_visible_overlap(item, query_text)
    if len(overlap) >= max(1, int(min_tokens)):
        return True
    return any(len(token) >= max(1, int(min_long_token_length)) for token in overlap)


def _filter_model_visible_local_evidence(
    payload: dict[str, Any],
    *,
    max_evidence: int,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Keep weak local hits audit-visible but out of model-visible evidence."""
    evidence_items = [item for item in list(payload.get("evidence_items") or []) if isinstance(item, dict)]
    if not evidence_items:
        return payload

    query_parts = [
        str(payload.get("full_query_text") or ""),
        str(payload.get("query_text") or ""),
        " ".join(str(item) for item in list(payload.get("queries_executed") or [])),
    ]
    query_text = "\n".join(part for part in query_parts if part.strip())
    cfg = config or {}
    support_atom_min_tokens = int(cfg.get("support_atom_min_visible_overlap_tokens", 3) or 3)
    support_atom_min_long = int(cfg.get("support_atom_min_visible_long_token_length", 14) or 14)
    weak_hit_min_tokens = int(cfg.get("weak_local_hit_min_visible_overlap_tokens", 2) or 2)
    weak_hit_min_long = int(cfg.get("weak_local_hit_min_visible_long_token_length", 8) or 8)
    current_benchmark = _current_local_benchmark()
    kept: list[dict[str, Any]] = []
    filtered: list[dict[str, Any]] = []
    filtered_cross_family_count = 0
    for item in evidence_items:
        status = str(item.get("rerank_status") or "").casefold()
        if status.startswith("source_audit"):
            kept.append(item)
            continue

        support_atom = _is_support_or_parameter_atom(item)
        weak_fallback = status in {
            "empty_filter_top1_fallback",
            "hybrid_topk_forced_keep",
        }
        if support_atom:
            family = _support_atom_source_family(item)
            target_benchmarks = _support_atom_target_benchmarks(family)
            if current_benchmark and target_benchmarks and current_benchmark not in target_benchmarks:
                filtered_cross_family_count += 1
                filtered.append(item)
                continue
            family_min_tokens = int(
                cfg.get(f"support_atom_{family}_min_visible_overlap_tokens", support_atom_min_tokens)
                or support_atom_min_tokens
            )
            family_min_long = int(
                cfg.get(f"support_atom_{family}_min_visible_long_token_length", support_atom_min_long)
                or support_atom_min_long
            )
            has_overlap = _has_specific_local_overlap(
                item,
                query_text,
                min_tokens=family_min_tokens,
                min_long_token_length=family_min_long,
            )
        else:
            has_overlap = _has_specific_local_overlap(
                item,
                query_text,
                min_tokens=weak_hit_min_tokens,
                min_long_token_length=weak_hit_min_long,
            )

        if (support_atom or weak_fallback) and not has_overlap:
            filtered.append(item)
            continue
        kept.append(item)

    if len(kept) == len(evidence_items):
        return payload

    updated = dict(payload)
    updated["evidence_items"] = kept[: max(1, int(max_evidence))]
    diagnostics = dict(updated.get("diagnostics") or {})
    diagnostics["model_visible_relevance_filter"] = {
        "enabled": True,
        "filtered_count": len(filtered),
        "filtered_cross_family_count": filtered_cross_family_count,
        "kept_count": len(updated["evidence_items"]),
        "reason": "support_atom_cross_family_or_low_confidence_local_hits_without_specific_visible_overlap",
        "current_benchmark": current_benchmark,
        "support_atom_min_visible_overlap_tokens": support_atom_min_tokens,
        "support_atom_min_visible_long_token_length": support_atom_min_long,
        "support_atom_medxpert_min_visible_overlap_tokens": cfg.get("support_atom_medxpert_min_visible_overlap_tokens"),
        "support_atom_medxpert_min_visible_long_token_length": cfg.get("support_atom_medxpert_min_visible_long_token_length"),
        "support_atom_public_private_min_visible_overlap_tokens": cfg.get(
            "support_atom_public_private_min_visible_overlap_tokens"
        ),
        "support_atom_public_private_min_visible_long_token_length": cfg.get(
            "support_atom_public_private_min_visible_long_token_length"
        ),
        "weak_local_hit_min_visible_overlap_tokens": weak_hit_min_tokens,
        "weak_local_hit_min_visible_long_token_length": weak_hit_min_long,
        "filtered_preview": [
            {
                "title": str(item.get("title") or "")[:160],
                "entity_name": str(item.get("entity_name") or "")[:160],
                "rerank_status": str(item.get("rerank_status") or "")[:80],
                "local_branch": str(item.get("local_branch") or "")[:80],
            }
            for item in filtered[:5]
        ],
    }
    diagnostics["rerank_kept_count"] = len(updated["evidence_items"])
    updated["diagnostics"] = diagnostics
    return updated


def _context_matches_overlay_benchmark(row_benchmark: str, context_benchmark: str) -> bool:
    row_l = str(row_benchmark or "").strip().casefold()
    context_l = str(context_benchmark or "").strip().casefold()
    if not row_l or not context_l:
        return True
    if row_l in {"external_authoritative", context_l}:
        return True
    if row_l.startswith("support_atoms_"):
        suffix = row_l.removeprefix("support_atoms_")
        if suffix == "public_private":
            return context_l in {"public_mcq", "private_mcq"}
        if suffix == "medxpert":
            return context_l == "medxpertqa_drug"
        if suffix == "hle":
            return context_l == "hle_drug"
    return False


@lru_cache(maxsize=8)
def _load_overlay_csv_records(
    paths_key: tuple[str, ...],
    entity_info_column: str,
) -> tuple[dict[str, Any], ...]:
    records: list[dict[str, Any]] = []
    for raw_path in paths_key:
        path = _resolve_project_path(raw_path)
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row_index, row in enumerate(reader):
                benchmark = str(row.get("benchmark", "") or "").strip()
                try:
                    question_index = int(row.get("index", "") or row_index)
                except Exception:
                    question_index = row_index
                raw_triples = row.get(entity_info_column)
                if raw_triples is None and entity_info_column != "key_entity_info_json":
                    raw_triples = row.get("key_entity_info_json")
                triples = _safe_json_loads(raw_triples, [])
                if not isinstance(triples, list):
                    continue
                for triple_index, triple in enumerate(triples):
                    if not isinstance(triple, list) or len(triple) < 3:
                        continue
                    entity_name = str(triple[0] or "").strip()
                    sentence = str(triple[1] or "").strip()
                    url = str(triple[2] or "").strip()
                    if not entity_name or not sentence:
                        continue
                    stable = hashlib.sha256(
                        f"{raw_path}|{row_index}|{triple_index}|{entity_name}|{sentence}|{url}".encode("utf-8")
                    ).hexdigest()
                    records.append(
                        {
                            "doc_id": -2_000_000 - int(stable[:10], 16),
                            "benchmark": benchmark,
                            "question_index": question_index,
                            "source_csv": path.name,
                            "entity_name": entity_name,
                            "source_url": url,
                            "chunk_text": sentence,
                            "row_index": row_index,
                            "triple_index": triple_index,
                        }
                    )
    return tuple(records)


def _overlay_record_score(record: dict[str, Any], query_text: str) -> tuple[float, set[str]]:
    query_tokens = set(_meaningful_local_evidence_tokens(query_text))
    if not query_tokens:
        return 0.0, set()
    record_text = " ".join(
        [
            str(record.get("entity_name") or ""),
            str(record.get("chunk_text") or ""),
        ]
    )
    record_tokens = set(_meaningful_local_evidence_tokens(record_text))
    overlap = query_tokens & record_tokens
    if not overlap:
        return 0.0, set()
    score = len(overlap) / max(6.0, len(record_tokens) ** 0.5)
    if any(len(token) >= 8 for token in overlap):
        score += 0.12
    score += _visible_choice_phrase_bonus(record_tokens=record_tokens, query_text=query_text)
    return score, overlap


def _visible_choice_phrase_bonus(*, record_tokens: set[str], query_text: str) -> float:
    choices_block = str(query_text or "")
    marker = re.search(r"Visible answer choices:\s*", choices_block, flags=re.IGNORECASE)
    if marker:
        choices_block = choices_block[marker.end():]
    choice_lines = re.findall(r"(?m)^\s*[A-Z][\.\)]\s+(.+)$", choices_block)
    best = 0.0
    for choice in choice_lines:
        tokens = _meaningful_local_evidence_tokens(choice)
        if not tokens:
            continue
        token_set = set(tokens)
        matched = token_set & record_tokens
        if not matched:
            continue
        if len(token_set) <= 2 and matched == token_set:
            best = max(best, 0.55)
            continue
        head = set(tokens[: min(3, len(tokens))])
        if head and head.issubset(record_tokens):
            best = max(best, 0.45)
        coverage = len(matched) / max(1, len(token_set))
        if coverage >= 0.75:
            best = max(best, 0.35)
        elif coverage >= 0.5:
            best = max(best, 0.20)
    return best


def _overlay_support_payload(
    base_payload: dict[str, Any],
    *,
    config: dict[str, Any],
    max_evidence: int,
) -> dict[str, Any] | None:
    raw_paths = tuple(str(item) for item in list(config.get("overlay_csv_paths") or []) if str(item).strip())
    if not raw_paths:
        return None
    context = get_current_benchmark_example_context()
    context_benchmark = str((context.benchmark if context is not None else "") or "").strip()
    query_text = "\n".join(
        part
        for part in [
            str(base_payload.get("full_query_text") or ""),
            str(base_payload.get("query_text") or ""),
            " ".join(str(item) for item in list(base_payload.get("queries_executed") or [])),
        ]
        if part.strip()
    )
    if bool(config.get("overlay_use_choices", False)) and context is not None and context.choices not in (None, "", []):
        choices_text = _choice_text(context.choices).strip()
        if choices_text:
            query_text = "\n\n".join([query_text, "Visible answer choices:\n" + choices_text])
    if not query_text:
        return None

    records = _load_overlay_csv_records(
        raw_paths,
        str(config.get("overlay_entity_info_column", "key_entity_info_json") or "key_entity_info_json"),
    )
    min_overlap = max(1, int(config.get("overlay_min_overlap_tokens", 2) or 2))
    min_score = float(config.get("overlay_min_score", 0.22) or 0.22)
    max_overlay = max(1, int(config.get("overlay_max_evidence", 4) or 4))

    scored: list[tuple[float, set[str], dict[str, Any]]] = []
    for record in records:
        if not _context_matches_overlay_benchmark(str(record.get("benchmark") or ""), context_benchmark):
            continue
        score, overlap = _overlay_record_score(record, query_text)
        if len(overlap) < min_overlap or score < min_score:
            continue
        scored.append((score, overlap, record))
    if not scored:
        return None
    scored.sort(key=lambda item: (item[0], len(item[1]), -abs(int(item[2].get("doc_id", 0) or 0))), reverse=True)
    best_entity = str(scored[0][2].get("entity_name") or "").strip()
    if best_entity:
        same_entity = [item for item in scored if str(item[2].get("entity_name") or "").strip() == best_entity]
        other_entity = [item for item in scored if str(item[2].get("entity_name") or "").strip() != best_entity]
        scored = [*same_entity, *other_entity]

    evidence_items: list[dict[str, Any]] = []
    source_scores: dict[str, float] = {}
    for score, overlap, record in scored[: min(max_overlay, max(1, int(max_evidence)))]:
        item = {
            "sentence": str(record.get("chunk_text") or ""),
            "title": str(record.get("entity_name") or ""),
            "url": str(record.get("source_url") or ""),
            "source_csv": str(record.get("source_csv") or ""),
            "doc_id": int(record.get("doc_id", 0) or 0),
            "entity_name": str(record.get("entity_name") or ""),
            "score": round(min(1.0, max(0.05, score)), 6),
            "dense_score": 0.0,
            "dense_raw_score": 0.0,
            "bm25_score": round(min(1.0, max(0.05, score)), 6),
            "bm25_raw_score": round(score, 6),
            "match_branches": ["overlay_lexical"],
            "rerank_score": 1,
            "rerank_status": "support_overlay_lexical",
            "contradiction": False,
            "evidence_status": "usable",
            "local_branch": "support_overlay_lexical",
            "matched_overlay_tokens": sorted(overlap)[:12],
        }
        evidence_items.append(item)
        key = item["url"] or item["title"]
        if key:
            source_scores[key] = max(source_scores.get(key, 0.0), float(item["score"]))

    return {
        "query_text": query_text,
        "full_query_text": query_text,
        "queries_executed": list(base_payload.get("queries_executed") or []),
        "retrieval_backend": "support_overlay_lexical",
        "extracted_entities": [],
        "resolved_entities": _dedupe_strings([str(item.get("entity_name", "")) for item in evidence_items]),
        "source_hits": [
            {"source_key": key, "score": round(score, 6)}
            for key, score in sorted(source_scores.items(), key=lambda pair: pair[1], reverse=True)
        ],
        "coarse_evidence_items": list(evidence_items),
        "rerank_results": [],
        "rerank_status": "support_overlay_lexical_no_llm_rerank",
        "rerank_candidate_count": len(scored),
        "rerank_kept_count": len(evidence_items),
        "diagnostics": {
            "diagnostic_version": 1,
            "query_mode": "support_overlay_lexical",
            "candidate_count": len(scored),
            "returned_evidence_count": len(evidence_items),
            "context_benchmark": context_benchmark,
            "source_files": list(raw_paths),
            "kept_preview": evidence_items[:5],
        },
        "evidence_items": evidence_items,
        "knowledge_base_stats": {
            "branch": "support_overlay_lexical",
            "source_files": list(raw_paths),
        },
        "dense_k": 0,
        "bm25_k": 0,
        "cache_hit": False,
    }


def _merge_overlay_payload(
    base_payload: dict[str, Any],
    overlay_payload: dict[str, Any] | None,
    *,
    max_evidence: int,
) -> dict[str, Any]:
    if not overlay_payload:
        return base_payload
    merged = dict(base_payload)
    merged["retrieval_backend"] = "+".join(
        part
        for part in [
            str(base_payload.get("retrieval_backend") or "").strip(),
            str(overlay_payload.get("retrieval_backend") or "").strip(),
        ]
        if part
    )
    merged["source_hits"] = [
        *list(overlay_payload.get("source_hits") or []),
        *list(base_payload.get("source_hits") or []),
    ]
    merged["resolved_entities"] = _dedupe_strings(
        [*list(base_payload.get("resolved_entities") or []), *list(overlay_payload.get("resolved_entities") or [])]
    )
    merged["coarse_evidence_items"] = _merge_evidence_items(
        [*list(overlay_payload.get("coarse_evidence_items") or []), *list(base_payload.get("coarse_evidence_items") or [])]
    )
    merged["evidence_items"] = _merge_evidence_items(
        [
            *[item for item in list(base_payload.get("evidence_items") or []) if isinstance(item, dict)],
            *[item for item in list(overlay_payload.get("evidence_items") or []) if isinstance(item, dict)],
        ]
    )[: max(1, int(max_evidence))]

    diagnostics = dict(base_payload.get("diagnostics") or {})
    diagnostics["support_overlay_lexical_branch"] = {
        "enabled": True,
        "returned_evidence_count": len(overlay_payload.get("evidence_items") or []),
        "resolved_entities": list(overlay_payload.get("resolved_entities") or []),
        "source_hits": list(overlay_payload.get("source_hits") or [])[:5],
    }
    diagnostics["coarse_candidate_count"] = len(merged.get("coarse_evidence_items") or [])
    diagnostics["rerank_kept_count"] = len(merged.get("evidence_items") or [])
    merged["diagnostics"] = diagnostics

    stats = dict(base_payload.get("knowledge_base_stats") or {})
    overlay_stats = overlay_payload.get("knowledge_base_stats") if isinstance(overlay_payload.get("knowledge_base_stats"), dict) else {}
    stats["source_files"] = _dedupe_strings(
        [*list(stats.get("source_files") or []), *list(overlay_stats.get("source_files") or [])]
    )
    stats["auxiliary_branches"] = _dedupe_strings(
        [*list(stats.get("auxiliary_branches") or []), str(overlay_stats.get("branch") or "support_overlay_lexical")]
    )
    merged["knowledge_base_stats"] = stats
    return merged


def _force_int_env(name: str) -> int | None:
    raw = str(os.environ.get(name, "") or "").strip()
    if not raw:
        return None
    try:
        value = int(raw)
    except Exception:
        return None
    return value if value > 0 else None


def _force_int_config(config: dict[str, Any], name: str) -> int | None:
    raw = config.get(name)
    if raw in (None, ""):
        return None
    try:
        value = int(raw)
    except Exception:
        return None
    return value if value > 0 else None


def _search_with_retriever(
    *,
    retriever: Any,
    question: str,
    choices: Any,
    entities: list[str] | None,
    require_rag: bool,
    max_evidence: int,
    dense_k: int,
    bm25_k: int,
    rerank_candidate_k: int | None,
    exclude_benchmark: str,
    exclude_question_index: int | None,
) -> dict[str, Any]:
    return retriever.search(
        question=question,
        choices=choices,
        entities=entities,
        require_rag=bool(require_rag),
        max_evidence=max_evidence,
        dense_k=dense_k,
        bm25_k=bm25_k,
        rerank_candidate_k=rerank_candidate_k,
        exclude_benchmark=exclude_benchmark,
        exclude_question_index=exclude_question_index,
    )


def _augment_query_with_current_example(
    query_text: str,
    *,
    goal: str = "",
    entities: list[str] | None = None,
) -> tuple[str, Any, list[str] | None, str, int | None]:
    context = get_current_benchmark_example_context()
    if context is None:
        parts = [query_text]
        if goal:
            parts.append(f"Decision goal:\n{goal}")
        return "\n\n".join(item for item in parts if item), None, entities, "", None

    parts: list[str] = []
    visible_anchors: list[str] = []
    if context.use_example_question and context.question:
        parts.append(context.question)
        visible_anchors = _extract_visible_anchor_entities(context.question)
        if visible_anchors:
            parts.append("Visible patient/retrieval anchors:\n" + "\n".join(f"- {item}" for item in visible_anchors))
    if query_text:
        parts.append(f"Search focus:\n{query_text}")
    if goal:
        parts.append(f"Decision goal:\n{goal}")

    effective_query = "\n\n".join(part for part in parts if str(part).strip()) or query_text
    effective_choices = context.choices if context.use_example_choices else None
    effective_entities = list(entities or [])
    for anchor in visible_anchors:
        if anchor not in effective_entities:
            effective_entities.append(anchor)
    if context.use_key_entities and context.key_entities:
        for entity in context.key_entities:
            if entity not in effective_entities:
                effective_entities.append(entity)
    return (
        effective_query,
        effective_choices,
        effective_entities or None,
        context.benchmark if context.exclude_current_example else "",
        context.index if context.exclude_current_example else None,
    )


def search_local(
    query: Any,
    choices: Any = None,
    require_rag: bool = True,
    max_evidence: int = 10,
    dense_k: int = 8,
    bm25_k: int = 8,
    goal: str = "",
    entities: list[str] | None = None,
    max_question_hits: int | None = None,
    context: Any = None,
) -> dict[str, Any]:
    # Compatibility-only input from the former question-grouped local index.
    # The unified store retrieves source-preserved evidence chunks rather than
    # benchmark question groups, so this value intentionally does not alter
    # ranking or expose benchmark grouping to the model.
    del max_question_hits
    query_list = [str(item).strip() for item in (query if isinstance(query, list) else [query]) if str(item).strip()]
    if not query_list:
        return {"success": False, "error": "Missing required parameter: query", "tool": "search_local"}

    retriever = get_retriever()
    multiquery_limit = max(1, min(6, int(retriever.config.get("multiquery_max_queries", 3) or 3)))
    query_list = query_list[:multiquery_limit]
    forced_dense_k = _force_int_env("SEARCH_LOCAL_FORCE_DENSE_K")
    forced_bm25_k = _force_int_env("SEARCH_LOCAL_FORCE_BM25_K")
    forced_max_evidence = _force_int_env("SEARCH_LOCAL_FORCE_MAX_EVIDENCE")
    config_forced_dense_k = _force_int_config(retriever.config, "force_dense_branch_k")
    config_forced_bm25_k = _force_int_config(retriever.config, "force_bm25_branch_k")
    config_forced_max_evidence = _force_int_config(retriever.config, "force_max_evidence")
    default_dense_k = max(1, int(getattr(retriever, "default_dense_k", dense_k) or dense_k))
    default_bm25_k = max(1, int(getattr(retriever, "default_bm25_k", bm25_k) or bm25_k))
    effective_dense_k = forced_dense_k or config_forced_dense_k or (
        default_dense_k if int(dense_k or 0) == 5 and default_dense_k != 5 else dense_k
    )
    effective_bm25_k = forced_bm25_k or config_forced_bm25_k or (
        default_bm25_k if int(bm25_k or 0) == 5 and default_bm25_k != 5 else bm25_k
    )
    effective_max_evidence = forced_max_evidence or config_forced_max_evidence or max_evidence
    configured_rerank_candidate_k = max(1, int(retriever.config.get("rerank_candidate_k", 8) or 8))
    multiquery_rerank_total_candidate_k = max(
        1,
        int(
            retriever.config.get(
                "multiquery_rerank_total_candidate_k",
                configured_rerank_candidate_k,
            )
            or configured_rerank_candidate_k
        ),
    )

    def _per_query_rerank_candidate_k(query_count: int) -> int:
        query_count = max(1, int(query_count or 1))
        if query_count <= 1:
            return configured_rerank_candidate_k
        per_query = (multiquery_rerank_total_candidate_k + query_count - 1) // query_count
        return max(1, min(configured_rerank_candidate_k, per_query))

    if len(query_list) == 1:
        effective_query, context_choices, context_entities, exclude_benchmark, exclude_index = _augment_query_with_current_example(
            query_list[0],
            goal=goal,
            entities=entities,
        )
        anchor_query = ""
        current_context = get_current_benchmark_example_context()
        if current_context is not None and current_context.use_example_question:
            anchor_query = _build_anchor_choice_query(
                question=current_context.question,
                choices=context_choices if context_choices is not None else choices,
                goal=goal,
            )
        executed_queries = list(query_list)
        if (
            anchor_query
            and anchor_query not in effective_query
            and bool(retriever.config.get("anchor_choice_union_enabled", False))
        ):
            union_queries = [effective_query, anchor_query]
            payloads = [
                _search_with_retriever(
                    retriever=retriever,
                    question=subquery,
                    choices=context_choices if context_choices is not None else choices,
                    entities=context_entities,
                    require_rag=bool(require_rag),
                    max_evidence=effective_max_evidence,
                    dense_k=effective_dense_k,
                    bm25_k=effective_bm25_k,
                    rerank_candidate_k=_per_query_rerank_candidate_k(len(union_queries)),
                    exclude_benchmark=exclude_benchmark,
                    exclude_question_index=exclude_index,
                )
                for subquery in union_queries
            ]
            executed_queries.append("anchor-choice context: " + anchor_query)
            payload = _merge_multi_query_payloads(
                queries=executed_queries,
                payloads=payloads,
                max_evidence=effective_max_evidence,
                dense_k=effective_dense_k,
                bm25_k=effective_bm25_k,
            )
        else:
            if (
                anchor_query
                and anchor_query not in effective_query
                and _should_lead_with_anchor_context(anchor_query, query_list[0])
            ):
                effective_query = anchor_query
                executed_queries.append("anchor-choice context: " + anchor_query)
            payload = _search_with_retriever(
                retriever=retriever,
                question=effective_query,
                choices=context_choices if context_choices is not None else choices,
                entities=context_entities,
                require_rag=bool(require_rag),
                max_evidence=effective_max_evidence,
                dense_k=effective_dense_k,
                bm25_k=effective_bm25_k,
                rerank_candidate_k=_per_query_rerank_candidate_k(1),
                exclude_benchmark=exclude_benchmark,
                exclude_question_index=exclude_index,
            )
        payload["queries_executed"] = executed_queries
    else:
        payloads = [
            (lambda augmented: _search_with_retriever(
                retriever=retriever,
                question=augmented[0],
                choices=augmented[1] if augmented[1] is not None else choices,
                entities=augmented[2],
                require_rag=bool(require_rag),
                max_evidence=effective_max_evidence,
                dense_k=effective_dense_k,
                bm25_k=effective_bm25_k,
                rerank_candidate_k=_per_query_rerank_candidate_k(len(query_list)),
                exclude_benchmark=augmented[3],
                exclude_question_index=augmented[4],
            ))(_augment_query_with_current_example(subquery, goal=goal, entities=entities))
            for subquery in query_list
        ]
        payload = _merge_multi_query_payloads(
            queries=query_list,
            payloads=payloads,
            max_evidence=effective_max_evidence,
            dense_k=effective_dense_k,
            bm25_k=effective_bm25_k,
        )
    overlay_payload = _overlay_support_payload(
        payload,
        config=retriever.config,
        max_evidence=effective_max_evidence,
    )
    payload = _merge_overlay_payload(payload, overlay_payload, max_evidence=effective_max_evidence)
    payload = _filter_model_visible_local_evidence(
        payload,
        max_evidence=effective_max_evidence,
        config=retriever.config,
    )

    evidence_items = payload.get("evidence_items") or []
    return {
        "success": bool(evidence_items),
        "status": "ok" if evidence_items else "no_results",
        "data": {
            **payload,
            "rendered_context": _render_context(payload),
            "returned_evidence_count": len(evidence_items),
            "requested_max_evidence": int(max_evidence),
            "effective_max_evidence": int(effective_max_evidence),
            "configured_rerank_candidate_k": int(configured_rerank_candidate_k),
            "multiquery_rerank_total_candidate_k": int(multiquery_rerank_total_candidate_k),
            "effective_rerank_candidate_k_per_query": int(_per_query_rerank_candidate_k(len(query_list))),
            "goal": goal,
        },
        "sources": [
            str(payload.get("knowledge_base_stats", {}).get("database_path", "")),
            *[str(item) for item in (payload.get("knowledge_base_stats", {}).get("source_files") or [])],
        ],
    }


class SearchLocalTool(FunctionTool):
    name = "search_local"
    description = (
        "Search the local private evidence knowledge base for entity-grounded evidence relevant to the question and optional choices."
    )
    input_hint = 'requires {"query":"<string>" or ["<string>", "..."],"goal":"patient/decision goal","entities":["plan entities"],"choices":["optional choice lines"],"require_rag":true,"dense_k":8,"bm25_k":8,"max_evidence":10}'
    tool_function = staticmethod(search_local)
    passes_context = True

    def build_kwargs(self, params: dict[str, Any]) -> dict[str, Any]:
        query_value: Any = params.get("query")
        if isinstance(query_value, list):
            cleaned_queries = self.clean_string_list(query_value)
            if not cleaned_queries:
                raise ValueError("Missing required parameter: query")
            query_argument: Any = cleaned_queries[:6]
        else:
            query_argument = self.require_str(params, "query")
        return {
            "query": query_argument,
            "choices": params.get("choices"),
            "goal": self.optional_str(params, "goal"),
            "entities": self.clean_string_list(params.get("entities"))[:20],
            "require_rag": bool(params.get("require_rag", True)),
            "dense_k": self.bounded_int(params, "dense_k", default=8, minimum=1, maximum=20),
            "bm25_k": self.bounded_int(params, "bm25_k", default=8, minimum=1, maximum=20),
            "max_evidence": self.bounded_int(params, "max_evidence", default=10, minimum=1, maximum=20),
            "max_question_hits": self.bounded_int(
                params,
                "max_question_hits",
                default=3,
                minimum=1,
                maximum=10,
            ) if "max_question_hits" in params else None,
        }
