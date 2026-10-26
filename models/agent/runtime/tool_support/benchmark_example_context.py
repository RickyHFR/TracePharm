from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
import csv
import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import sys
from threading import local
from typing import Any, Iterator

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[4]
_CURRENT_EXAMPLE_CONTEXT: ContextVar["BenchmarkExampleContext | None"] = ContextVar(
    "benchmark_example_context",
    default=None,
)
_THREAD_LOCAL_CONTEXT = local()


def _search_local_config_path() -> Path:
    raw = str(os.environ.get("SEARCH_LOCAL_CONFIG_PATH", "") or "").strip()
    if raw:
        path = Path(raw)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return path.resolve()
    return (PROJECT_ROOT / "models" / "agent" / "runtime" / "config" / "search_local.yaml").resolve()


@dataclass(slots=True)
class BenchmarkExampleContext:
    benchmark: str
    index: int
    question: str
    choices: Any
    answer_type: str
    eval_group: str
    routing_metadata: dict[str, Any]
    key_entities: list[str]
    key_entity_info: list[list[Any]]
    use_example_question: bool
    use_example_choices: bool
    use_key_entities: bool
    exclude_current_example: bool


def _safe_json_list(raw: Any) -> list[Any]:
    text = str(raw or "").strip()
    if not text:
        return []
    try:
        value = json.loads(text)
    except Exception:
        return []
    return value if isinstance(value, list) else []


@lru_cache(maxsize=1)
def _load_example_payloads() -> dict[tuple[str, int], dict[str, Any]]:
    config = yaml.safe_load(_search_local_config_path().read_text(encoding="utf-8")) or {}
    if not isinstance(config, dict):
        return {}

    raw_paths = list(config.get("source_csv_paths") or [])
    entity_map: dict[tuple[str, int], dict[str, Any]] = {}
    csv.field_size_limit(sys.maxsize)

    for raw_path in raw_paths:
        csv_path = Path(str(raw_path))
        if not csv_path.is_absolute():
            csv_path = (PROJECT_ROOT / csv_path).resolve()
        if not csv_path.exists():
            continue

        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                benchmark = str(row.get("benchmark", "") or "").strip()
                if not benchmark:
                    continue
                try:
                    index = int(row.get("index", "") or 0)
                except Exception:
                    continue
                key_entities = [str(item).strip() for item in _safe_json_list(row.get("key_entities_json")) if str(item).strip()]
                key_entity_info = []
                for item in _safe_json_list(row.get("key_entity_info_json")):
                    if isinstance(item, list):
                        key_entity_info.append(list(item))
                if key_entities or key_entity_info:
                    entity_map[(benchmark, index)] = {
                        "key_entities": key_entities,
                        "key_entity_info": key_entity_info,
                    }
    return entity_map


def _lookup_key_entities(benchmark: str, index: int) -> list[str]:
    payload = _load_example_payloads().get((str(benchmark).strip(), int(index)), {})
    return list(payload.get("key_entities", []) or [])


def _lookup_key_entity_info(benchmark: str, index: int) -> list[list[Any]]:
    payload = _load_example_payloads().get((str(benchmark).strip(), int(index)), {})
    return [list(item) for item in (payload.get("key_entity_info", []) or []) if isinstance(item, list)]


def _first_int_or_none(*values: Any) -> int | None:
    for value in values:
        if value is None:
            continue
        try:
            return int(value)
        except Exception:
            continue
    return None


def build_benchmark_example_payload(example: Any, adapter_config: dict[str, Any]) -> dict[str, Any] | None:
    example_benchmark = str(getattr(example, "benchmark", "") or "").strip()
    if not example_benchmark:
        return None

    example_index = int(getattr(example, "index", 0) or 0)
    question = str(getattr(example, "question", "") or "").strip()
    choices = getattr(example, "choices", None)
    answer_type = str(getattr(example, "answer_type", "") or "").strip()
    raw_metadata = getattr(example, "metadata", {}) or {}
    metadata = raw_metadata if isinstance(raw_metadata, dict) else {}
    ablation_sample = metadata.get("ablation_sample") if isinstance(metadata.get("ablation_sample"), dict) else {}
    eval_group = (
        str(metadata.get("eval_group", "") or "").strip()
        or str(metadata.get("original_benchmark", "") or "").strip()
        or str(ablation_sample.get("source_benchmark", "") or "").strip()
        or example_benchmark
    )
    local_benchmark = (
        str(ablation_sample.get("source_benchmark", "") or "").strip()
        or str(metadata.get("eval_group", "") or "").strip()
        or str(metadata.get("original_benchmark", "") or "").strip()
        or example_benchmark
    )
    local_index = _first_int_or_none(
        ablation_sample.get("source_index"),
        ablation_sample.get("source_row"),
        metadata.get("source_index"),
        metadata.get("original_index"),
        example_index,
    )
    if local_index is None:
        local_index = example_index
    # Keep routing metadata answer-free.  This lets the runtime choose the right
    # evidence lane without exposing labels, accepted answers, or rationales.
    routing_metadata = {
        key: metadata.get(key)
        for key in (
            "eval_group",
            "original_benchmark",
            "original_index",
            "source_file",
            "source_id",
            "question_type",
            "task_type",
            "source_dataset",
            "medical_task",
            "filter_focus",
            "primary_body_system",
            "primary_organ_system",
            "primary_special_population",
        )
        if key in metadata
    }
    routing_metadata["normalized_benchmark"] = example_benchmark
    routing_metadata["normalized_index"] = example_index
    routing_metadata["local_evidence_benchmark"] = local_benchmark
    routing_metadata["local_evidence_index"] = local_index
    if ablation_sample:
        routing_metadata["source_benchmark"] = ablation_sample.get("source_benchmark")
        routing_metadata["source_index"] = ablation_sample.get("source_index")
    overrides = adapter_config.get("agent_config_overrides") if isinstance(adapter_config.get("agent_config_overrides"), dict) else {}
    raw_enabled_tools = adapter_config.get("enabled_tools") or overrides.get("enabled_tools") or []
    enabled_tools = {
        str(item).strip()
        for item in raw_enabled_tools
        if str(item).strip()
    }
    requires_current_example_context = "get_current_example_key_entity_info" in enabled_tools
    requested_key_entities = bool(adapter_config.get("search_local_use_key_entities", False)) or requires_current_example_context
    allow_unsafe_answer_conditioned_context = bool(adapter_config.get("allow_answer_conditioned_example_context", False))
    use_key_entities = bool(requested_key_entities and allow_unsafe_answer_conditioned_context)
    leave_one_out = bool(adapter_config.get("search_local_leave_one_out", "search_local" in enabled_tools))

    return {
        "benchmark": local_benchmark,
        "index": local_index,
        "question": question,
        "choices": choices,
        "answer_type": answer_type,
        "eval_group": eval_group,
        "routing_metadata": routing_metadata,
        "key_entities": _lookup_key_entities(benchmark=local_benchmark, index=local_index) if use_key_entities else [],
        "key_entity_info": _lookup_key_entity_info(benchmark=local_benchmark, index=local_index) if use_key_entities else [],
        "use_example_question": bool(adapter_config.get("search_local_use_example_question", False)),
        "use_example_choices": bool(adapter_config.get("search_local_use_example_choices", False)),
        "use_key_entities": use_key_entities,
        "exclude_current_example": leave_one_out,
        "unsafe_key_entity_context_requested_but_blocked": bool(requested_key_entities and not use_key_entities),
    }


@contextmanager
def bind_benchmark_example_context(example: Any, adapter_config: dict[str, Any]) -> Iterator[None]:
    payload = build_benchmark_example_payload(example=example, adapter_config=adapter_config)
    if not payload:
        yield
        return
    use_example_question = bool(payload.get("use_example_question", False))
    use_example_choices = bool(payload.get("use_example_choices", False))
    use_key_entities = bool(payload.get("use_key_entities", False))
    exclude_current_example = bool(payload.get("exclude_current_example", False))
    if not (use_example_question or use_example_choices or use_key_entities or exclude_current_example):
        yield
        return

    benchmark = str(payload.get("benchmark", "") or "").strip()
    index = int(payload.get("index", 0) or 0)
    question = str(payload.get("question", "") or "").strip()
    choices = payload.get("choices")
    answer_type = str(payload.get("answer_type", "") or "").strip()
    eval_group = str(payload.get("eval_group", "") or "").strip()
    routing_metadata = payload.get("routing_metadata") if isinstance(payload.get("routing_metadata"), dict) else {}
    key_entities = list(payload.get("key_entities", []) or [])
    key_entity_info = [list(item) for item in (payload.get("key_entity_info", []) or []) if isinstance(item, list)]

    token = _CURRENT_EXAMPLE_CONTEXT.set(
        BenchmarkExampleContext(
            benchmark=benchmark,
            index=index,
            question=question,
            choices=choices,
            answer_type=answer_type,
            eval_group=eval_group,
            routing_metadata=dict(routing_metadata),
            key_entities=key_entities,
            key_entity_info=key_entity_info,
            use_example_question=use_example_question,
            use_example_choices=use_example_choices,
            use_key_entities=use_key_entities,
            exclude_current_example=exclude_current_example,
        )
    )
    _THREAD_LOCAL_CONTEXT.current_example = _CURRENT_EXAMPLE_CONTEXT.get()
    try:
        yield
    finally:
        _CURRENT_EXAMPLE_CONTEXT.reset(token)
        _THREAD_LOCAL_CONTEXT.current_example = _CURRENT_EXAMPLE_CONTEXT.get()


def get_current_benchmark_example_context() -> BenchmarkExampleContext | None:
    context = _CURRENT_EXAMPLE_CONTEXT.get()
    if context is not None:
        return context
    thread_context = getattr(_THREAD_LOCAL_CONTEXT, "current_example", None)
    if isinstance(thread_context, BenchmarkExampleContext):
        return thread_context
    return None
