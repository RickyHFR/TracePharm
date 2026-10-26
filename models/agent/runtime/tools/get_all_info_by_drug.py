from __future__ import annotations

import importlib.util
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from threading import BoundedSemaphore, Lock
from typing import Any
from urllib.parse import quote_plus

from agent.telemetry import record_tool_substep
from llm.deepseek_client import DeepSeekClient

from .base_tool import FunctionTool

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


_GLOBAL_SELECTOR_CONCURRENCY = 32
_SELECTOR_SEMAPHORE = BoundedSemaphore(_GLOBAL_SELECTOR_CONCURRENCY)
_SELECTOR_CLIENTS: dict[str, DeepSeekClient] = {}
_SELECTOR_CLIENTS_LOCK = Lock()
_SELECTOR_SLOT_DIR = Path(tempfile.gettempdir()) / "nature_get_all_info_by_drug_selector_slots"
_CHILD_ENV = "AGENT_GET_ALL_INFO_BY_DRUG_CHILD"
_ISOLATE_ENV = "AGENT_GET_ALL_INFO_BY_DRUG_ISOLATE"
_SUBPROCESS_TIMEOUT_ENV = "AGENT_GET_ALL_INFO_BY_DRUG_SUBPROCESS_TIMEOUT_SECONDS"
_CHILD_JSON_START = "__GET_ALL_INFO_BY_DRUG_JSON_START__"
_CHILD_JSON_END = "__GET_ALL_INFO_BY_DRUG_JSON_END__"


_FDA_ALWAYS_USELESS_KEYS = {
    "_meta",
    "effective_time",
    "id",
    "openfda",
    "package_label_principal_display_panel",
    "set_id",
    "spl_product_data_elements",
    "version",
}
_FDA_FALLBACK_PRIORITY = [
    "indications_and_usage",
    "contraindications",
    "drug_interactions",
    "dosage_and_administration",
    "warnings_and_cautions",
    "warnings",
    "use_in_specific_populations",
    "pregnancy",
    "pediatric_use",
    "geriatric_use",
    "clinical_studies",
    "clinical_pharmacology",
    "pharmacokinetics",
    "mechanism_of_action",
    "adverse_reactions",
]


@lru_cache(maxsize=1)
def _load_curemas_module() -> Any:
    module_path = Path(__file__).resolve().parents[4] / "reference" / "cureMAS_tools.py"
    spec = importlib.util.spec_from_file_location("reference_curemas_tools", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load cureMAS tool module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=2)
def _load_qwen_tokenizer(model_name: str) -> Any:
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(model_name, local_files_only=True, trust_remote_code=False)
    except Exception:
        return None


def _runtime_config() -> dict[str, Any]:
    path = Path(__file__).resolve().parents[1] / "config" / "agent.yaml"
    if not path.exists():
        return {}
    try:
        import yaml

        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _env_flag(name: str, default: bool) -> bool:
    raw = str(os.environ.get(name, "") or "").strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    return default


def _should_isolate_get_all_info_process() -> bool:
    if _env_flag(_CHILD_ENV, False):
        return False
    # macOS can load incompatible OpenMP runtimes when the reference multi-source
    # drug helper and the FAISS local retriever are initialized in the same Python
    # process.  The official ablation configs set this explicitly; the platform
    # default protects standalone local smoke probes as well.
    return _env_flag(_ISOLATE_ENV, sys.platform == "darwin")


def _subprocess_timeout_seconds() -> float:
    try:
        return max(1.0, float(os.environ.get(_SUBPROCESS_TIMEOUT_ENV, "240") or 240))
    except Exception:
        return 240.0


def _json_roundtrip_safe(value: Any) -> Any:
    try:
        return json.loads(json.dumps(value, ensure_ascii=False, default=str))
    except Exception:
        return str(value)


def _trim_child_message(message: dict[str, Any]) -> dict[str, Any]:
    trimmed: dict[str, Any] = {
        "role": str(message.get("role", "") or ""),
        "content": str(message.get("content", "") or "")[:12000],
    }
    payload = message.get("_tool_payload")
    if isinstance(payload, dict):
        trimmed["_tool_payload"] = _json_roundtrip_safe(payload)
    return trimmed


def _child_context_payload(context: Any) -> Any:
    if not isinstance(context, dict):
        return _json_roundtrip_safe(context)
    messages = context.get("message_context") or []
    if not isinstance(messages, list):
        return _json_roundtrip_safe(context)

    selected_indices: set[int] = set(range(max(0, len(messages) - 8), len(messages)))
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            continue
        if str(message.get("role", "")).strip().lower() == "user" and str(message.get("content", "")).strip():
            selected_indices.add(index)
            break

    selected_messages = [
        _trim_child_message(messages[index])
        for index in sorted(selected_indices)
        if 0 <= index < len(messages) and isinstance(messages[index], dict)
    ]
    return {"message_context": selected_messages}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _runtime_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _pythonpath_for_child(env: dict[str, str]) -> str:
    entries = [str(_runtime_root()), str(_repo_root())]
    existing = str(env.get("PYTHONPATH", "") or "")
    if existing:
        entries.extend(part for part in existing.split(os.pathsep) if part)
    deduped: list[str] = []
    for entry in entries:
        if entry not in deduped:
            deduped.append(entry)
    return os.pathsep.join(deduped)


def _parse_child_result(stdout: str) -> dict[str, Any]:
    start = stdout.rfind(_CHILD_JSON_START)
    end = stdout.rfind(_CHILD_JSON_END)
    if start < 0 or end < 0 or end <= start:
        raise ValueError("isolated child did not emit a delimited JSON result")
    start += len(_CHILD_JSON_START)
    payload = stdout[start:end].strip()
    parsed = json.loads(payload)
    if not isinstance(parsed, dict):
        raise ValueError("isolated child emitted a non-object JSON result")
    return parsed


def _tail(text: str, limit: int = 1200) -> str:
    cleaned = str(text or "").strip()
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[-limit:]


def _get_all_info_by_drug_subprocess(drug_name: str, context: Any = None) -> dict[str, Any]:
    started = time.perf_counter()
    payload = {
        "drug_name": str(drug_name or ""),
        "context": _child_context_payload(context),
    }
    env = os.environ.copy()
    env[_CHILD_ENV] = "1"
    env["PYTHONPATH"] = _pythonpath_for_child(env)
    try:
        completed = subprocess.run(
            [sys.executable, "-m", "tools.get_all_info_by_drug"],
            input=json.dumps(payload, ensure_ascii=False),
            text=True,
            capture_output=True,
            cwd=str(_repo_root()),
            env=env,
            timeout=_subprocess_timeout_seconds(),
            check=False,
        )
        record_tool_substep(
            "get_all_info_by_drug",
            "process_isolated_fetch",
            time.perf_counter() - started,
            {"returncode": completed.returncode},
        )
        if completed.returncode != 0:
            return {
                "success": False,
                "error": (
                    "Isolated drug-information process failed "
                    f"(exit={completed.returncode}): {_tail(completed.stderr or completed.stdout)}"
                ),
                "drug_name": drug_name,
            }
        return _parse_child_result(completed.stdout)
    except subprocess.TimeoutExpired:
        record_tool_substep(
            "get_all_info_by_drug",
            "process_isolated_fetch",
            time.perf_counter() - started,
            {"timeout": True},
        )
        return {
            "success": False,
            "error": f"Isolated drug-information process timed out after {_subprocess_timeout_seconds():.0f}s",
            "drug_name": drug_name,
        }
    except Exception as exc:
        record_tool_substep(
            "get_all_info_by_drug",
            "process_isolated_fetch",
            time.perf_counter() - started,
            {"exception": type(exc).__name__},
        )
        return {
            "success": False,
            "error": f"Failed to run isolated drug-information process: {exc}",
            "drug_name": drug_name,
        }


def _count_tokens(text: str, tokenizer_name: str) -> int:
    tokenizer = _load_qwen_tokenizer(tokenizer_name)
    if tokenizer is not None:
        try:
            return len(tokenizer.encode(text, add_special_tokens=False))
        except Exception:
            pass
    return max(1, len(text) // 4)


def _truncate_to_token_limit(text: str, tokenizer_name: str, max_tokens: int) -> str:
    cleaned = str(text or "").strip()
    if not cleaned:
        return ""
    limit = max(1, int(max_tokens))
    tokenizer = _load_qwen_tokenizer(tokenizer_name)
    if tokenizer is not None:
        try:
            token_ids = tokenizer.encode(cleaned, add_special_tokens=False)
            if len(token_ids) <= limit:
                return cleaned
            return tokenizer.decode(token_ids[:limit], skip_special_tokens=True).strip()
        except Exception:
            pass
    approx_chars = max(32, limit * 4)
    return cleaned[:approx_chars].strip()


def _normalize_label_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        cleaned = []
        for item in value:
            normalized = _normalize_label_value(item)
            if normalized in ("", [], {}, None):
                continue
            cleaned.append(normalized)
        return cleaned
    if isinstance(value, dict):
        cleaned_dict = {}
        for key, item in value.items():
            normalized = _normalize_label_value(item)
            if normalized in ("", [], {}, None):
                continue
            cleaned_dict[str(key)] = normalized
        return cleaned_dict
    return value


def _render_field_text(key: str, value: Any) -> str:
    normalized = _normalize_label_value(value)
    if normalized in ("", [], {}, None):
        return ""
    if isinstance(normalized, list):
        body = "\n".join(f"- {str(item).strip()}" for item in normalized if str(item).strip())
    elif isinstance(normalized, dict):
        body = json.dumps(normalized, ensure_ascii=False, indent=2)
    else:
        body = str(normalized).strip()
    if not body:
        return ""
    return f"{key}:\n{body}"


def _extract_question_context(context: Any) -> str:
    if isinstance(context, dict):
        context = context.get("message_context") or []
    if not isinstance(context, list):
        return str(context or "").strip()
    for message in context:
        if str(message.get("role", "")).strip().lower() != "user":
            continue
        content = str(message.get("content", "")).strip()
        if content:
            return content
    return ""


def _strip_tool_call_lines(text: str) -> str:
    lines = []
    for line in str(text or "").splitlines():
        if line.strip().startswith("TOOL_CALL:"):
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def _extract_reasoning_context(context: Any, max_messages: int = 4) -> str:
    if isinstance(context, dict):
        context = context.get("message_context") or []
    if not isinstance(context, list):
        return ""
    snippets: list[str] = []
    for message in context:
        role = str(message.get("role", "")).strip().lower()
        if role not in {"assistant", "user"}:
            continue
        content = _strip_tool_call_lines(message.get("content", ""))
        if not content:
            continue
        snippets.append(f"[{role}] {content}")
    return "\n\n".join(snippets[-max_messages:])


def _extract_previous_tool_state(drug_name: str, context: Any) -> dict[str, Any]:
    state = {"used_keys": [], "round": 0}
    if isinstance(context, dict):
        context = context.get("message_context") or []
    if not isinstance(context, list):
        return state

    normalized_name = str(drug_name or "").strip().lower()
    for message in context:
        payload = message.get("_tool_payload")
        if not isinstance(payload, dict):
            continue
        if str(payload.get("drug_name", "")).strip().lower() != normalized_name:
            continue
        selected = payload.get("fda_label_used_keys") or payload.get("fda_label_selected_keys") or []
        if isinstance(selected, list):
            for key in selected:
                text = str(key).strip()
                if text and text not in state["used_keys"]:
                    state["used_keys"].append(text)
        try:
            state["round"] = max(state["round"], int(payload.get("fda_label_retrieval_round", 0) or 0))
        except Exception:
            pass
    return state


def _select_fallback_keys(available_keys: list[str], question_context: str) -> list[str]:
    lowered = str(question_context or "").lower()
    heuristic_groups = [
        (("dose", "dosage", "administer", "frequency", "renal", "hepatic"), ["dosage_and_administration", "use_in_specific_populations"]),
        (("contraind", "avoid", "unsafe", "allergy"), ["contraindications", "warnings_and_cautions", "warnings"]),
        (("pregnan", "breast", "lactat", "trimester"), ["pregnancy", "use_in_specific_populations"]),
        (("interaction", "coadmin", "together with", "cyp"), ["drug_interactions", "clinical_pharmacology", "pharmacokinetics"]),
        (("indication", "used for", "treat", "appropriate", "most appropriate"), ["indications_and_usage", "contraindications"]),
        (("trial", "study", "endpoint", "efficacy"), ["clinical_studies", "clinical_pharmacology"]),
        (("mechanism", "moa", "target"), ["mechanism_of_action", "clinical_pharmacology"]),
    ]
    selected: list[str] = []
    for triggers, candidate_keys in heuristic_groups:
        if not any(trigger in lowered for trigger in triggers):
            continue
        for key in candidate_keys:
            if key in available_keys and key not in selected:
                selected.append(key)
        if selected:
            break

    if selected:
        return selected[:4]

    for key in _FDA_FALLBACK_PRIORITY:
        if key in available_keys and key not in selected:
            selected.append(key)
        if len(selected) >= 4:
            break
    return selected or available_keys[:2]


def _select_relevant_fda_keys(
    *,
    drug_name: str,
    question_context: str,
    reasoning_context: str,
    key_previews: dict[str, str],
) -> tuple[list[str], str]:
    available_keys = list(key_previews.keys())
    if not available_keys:
        return [], ""

    preview_lines = []
    for key in available_keys:
        preview_lines.append(f"- {key}\n{key_previews[key]}")
    prompt = (
        f"Drug: {drug_name}\n\n"
        f"Question context:\n{question_context or '(empty)'}\n\n"
        f"Prior reasoning context:\n{reasoning_context or '(none)'}\n\n"
        "Candidate FDA label keys with short previews:\n"
        f"{chr(10).join(preview_lines)}\n\n"
        "Select only FDA keys that are directly useful for answering the current benchmark question.\n"
        "Prefer keys that can determine the answer with high confidence, such as indications, contraindications, warnings, dosing, interactions, use in specific populations, pregnancy, clinical studies, or clinical pharmacology.\n"
        "Do not select metadata, packaging, display, or purely administrative keys.\n"
        "Return JSON with:\n"
        '{"selected_keys":["..."],"notes":"short reason"}'
    )

    config_path = _selector_config_path()
    raw = ""
    try:
        client = _get_shared_selector_client(config_path)
        with _SELECTOR_SEMAPHORE:
            with _acquire_cross_process_selector_slot():
                raw = client.chat_json(
                    [
                        {
                            "role": "system",
                            "content": "You are selecting FDA label fields for a medical question-answering tool. Return strict JSON only.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=400,
                )
        payload = json.loads(raw)
        selected = payload.get("selected_keys") or []
        if not isinstance(selected, list):
            selected = []
        cleaned = []
        for key in selected:
            text = str(key).strip()
            if text in key_previews and text not in cleaned:
                cleaned.append(text)
        if cleaned:
            return cleaned[:6], raw
    except Exception:
        pass

    return _select_fallback_keys(available_keys, question_context), raw


def _build_fda_preview_map(
    *,
    fda_data: dict[str, Any],
    tokenizer_name: str,
    preview_tokens: int,
    excluded_keys: set[str],
    used_keys: list[str],
) -> dict[str, str]:
    previews: dict[str, str] = {}
    for key, value in fda_data.items():
        if key in excluded_keys or key in used_keys:
            continue
        rendered = _render_field_text(key, value)
        if not rendered:
            continue
        previews[key] = _truncate_to_token_limit(rendered, tokenizer_name, preview_tokens)
    return previews


def _selector_config_path() -> str | None:
    return str(os.environ.get("AGENT_LLM_CONFIG_PATH", "") or "").strip() or None


def _get_shared_selector_client(config_path: str | None) -> DeepSeekClient:
    cache_key = str(config_path or "__default__")
    with _SELECTOR_CLIENTS_LOCK:
        client = _SELECTOR_CLIENTS.get(cache_key)
        if client is None:
            client = DeepSeekClient(config_path=config_path)
            _SELECTOR_CLIENTS[cache_key] = client
        return client


@contextmanager
def _acquire_cross_process_selector_slot(max_slots: int = _GLOBAL_SELECTOR_CONCURRENCY):
    import fcntl

    _SELECTOR_SLOT_DIR.mkdir(parents=True, exist_ok=True)
    acquired_file = None
    while acquired_file is None:
        for slot_index in range(max_slots):
            slot_path = _SELECTOR_SLOT_DIR / f"slot_{slot_index:02d}.lock"
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


def _get_source_cache(module: Any) -> dict[str, Any]:
    cache = getattr(module, "_agent_runtime_drug_source_cache", None)
    if isinstance(cache, dict):
        return cache
    cache = {}
    setattr(module, "_agent_runtime_drug_source_cache", cache)
    return cache


_STRENGTH_TOKEN_RE = re.compile(r"\b\d+(?:\.\d+)?\s*(?:mg|mcg|g|ml|%|unit|units|iu)\b", re.IGNORECASE)
_FORM_TOKENS = {
    "tablet", "tablets", "capsule", "capsules", "oral", "injection", "injectable",
    "solution", "suspension", "extended", "release", "delayed", "iv", "po", "im",
}


def _normalize_drug_identity_name(value: Any) -> str:
    text = _STRENGTH_TOKEN_RE.sub(" ", str(value or "").lower())
    text = re.sub(r"[^a-z0-9]+", " ", text)
    tokens = [token for token in text.split() if token and token not in _FORM_TOKENS]
    return " ".join(tokens).strip()


def _as_name_list(value: Any) -> list[str]:
    values = value if isinstance(value, list) else [value]
    result: list[str] = []
    for item in values:
        text = str(item or "").strip()
        if text and text not in result:
            result.append(text)
    return result


def _fda_identity_candidates(record: dict[str, Any]) -> list[str]:
    openfda = record.get("openfda") if isinstance(record.get("openfda"), dict) else {}
    names: list[str] = []
    for key in ("brand_name", "generic_name", "substance_name"):
        for item in _as_name_list(openfda.get(key)):
            if item not in names:
                names.append(item)
    return names


def _identity_match(requested_name: str, candidate_names: list[str]) -> dict[str, Any]:
    requested = _normalize_drug_identity_name(requested_name)
    requested_tokens = set(requested.split())
    best_score = 0.0
    best_name = ""
    for candidate_name in candidate_names:
        candidate = _normalize_drug_identity_name(candidate_name)
        candidate_tokens = set(candidate.split())
        if not candidate or not requested:
            score = 0.0
        elif candidate == requested:
            score = 1.0
        elif requested_tokens and requested_tokens.issubset(candidate_tokens) and len(candidate_tokens - requested_tokens) <= 2:
            score = 0.88
        elif candidate_tokens and candidate_tokens.issubset(requested_tokens) and len(requested_tokens - candidate_tokens) <= 2:
            score = 0.84
        else:
            union = requested_tokens | candidate_tokens
            score = (len(requested_tokens & candidate_tokens) / len(union)) if union else 0.0
        if score > best_score:
            best_score = score
            best_name = candidate_name
    return {
        "requested_name": requested_name,
        "requested_normalized": requested,
        "matched_name": best_name,
        "candidate_names": candidate_names,
        "match_confidence": round(best_score, 4),
        "match_status": "matched" if best_score >= 0.78 else "rejected_identity_mismatch",
    }


def _fda_product_identity(drug_name: str, record: dict[str, Any]) -> dict[str, Any]:
    """Match the labeled product, not an individual ingredient of a combination."""
    openfda = record.get("openfda") or {}
    identity = _identity_match(drug_name, _fda_identity_candidates(record))
    generic_names = _as_name_list(openfda.get("generic_name"))
    substances = {_normalize_drug_identity_name(name) for name in _as_name_list(openfda.get("substance_name"))}
    separator = r"\b(?:and|with)\b|[+/;]"
    combination = len(substances) > 1 or any(re.search(separator, name, re.I) for name in generic_names)
    requested = _normalize_drug_identity_name(drug_name)
    brand_match = requested in {_normalize_drug_identity_name(name) for name in _as_name_list(openfda.get("brand_name"))}
    if combination and not re.search(separator, drug_name, re.I) and not (brand_match and requested not in substances):
        identity.update(match_confidence=0.0, match_status="rejected_combination_product")
    return identity


def _fetch_fda_label(
    module: Any,
    drug_name: str,
) -> tuple[dict[str, Any] | None, float, dict[str, Any], str]:
    started = time.perf_counter()
    params = [
        {"base_url": "https://api.fda.gov/drug/label.json", "limit": 5, "search": f'(openfda.brand_name:("{drug_name}"))', "kind": "brand"},
        {"base_url": "https://api.fda.gov/drug/label.json", "limit": 5, "search": f'(openfda.generic_name:("{drug_name}"))', "kind": "generic"},
        {"base_url": "https://api.fda.gov/drug/label.json", "limit": 5, "search": f'"{drug_name}"', "kind": "full_text_fallback"},
    ]
    fda_data: dict[str, Any] | None = None
    best_identity: dict[str, Any] = {
        "requested_name": drug_name,
        "match_status": "not_found",
        "match_confidence": 0.0,
        "matched_name": "",
        "candidate_names": [],
    }
    selected_url = ""
    for param in params:
        base_url = param["base_url"]
        query_param = {key: value for key, value in param.items() if key not in {"base_url", "kind"}}
        try:
            query = module.urlencode(query_param, quote_via=module.quote)
            url = f"{base_url}?{query}"
            response = module._request_with_retry("get", url, timeout=10)
            if response is None:
                continue
            data = response.json()
            if isinstance(data, dict) and data.get("results"):
                candidates = []
                for result in list(data.get("results") or []):
                    if not isinstance(result, dict):
                        continue
                    identity = _fda_product_identity(drug_name, result)
                    identity["query_kind"] = param["kind"]
                    candidates.append((float(identity["match_confidence"]), result, identity))
                if not candidates:
                    continue
                candidates.sort(key=lambda item: item[0], reverse=True)
                score, candidate, identity = candidates[0]
                if score > float(best_identity.get("match_confidence", 0.0) or 0.0):
                    best_identity = identity
                threshold = 0.78 if param["kind"] == "full_text_fallback" else 0.72
                if score < threshold:
                    continue
                fda_data = dict(candidate)
                fda_data["_meta"] = data.get("meta")
                best_identity = identity
                selected_url = url
                break
        except Exception:
            continue
    return fda_data, time.perf_counter() - started, best_identity, selected_url


def _fetch_opentargets(module: Any, drug_name: str) -> tuple[dict[str, Any] | None, float]:
    started = time.perf_counter()
    formatted_ot: dict[str, Any] | None = None
    try:
        base_url = "https://api.platform.opentargets.org/api/v4/graphql"
        drug_query = """
        query DrugSearch($q: String!) {
            search(queryString: $q, entityNames: ["drug"], page: {index: 0, size: 5}) {
                hits { id name }
            }
        }
        """
        response = module.requests.post(base_url, json={"query": drug_query, "variables": {"q": drug_name}}, timeout=10)
        if response.status_code == 200:
            hits = response.json().get("data", {}).get("search", {}).get("hits", [])
            ranked_hits = []
            for hit in hits:
                if not isinstance(hit, dict):
                    continue
                identity = _identity_match(drug_name, [str(hit.get("name", "") or "")])
                ranked_hits.append((float(identity["match_confidence"]), hit, identity))
            ranked_hits.sort(key=lambda item: item[0], reverse=True)
            if ranked_hits and ranked_hits[0][0] >= 0.78:
                _, selected_hit, selected_identity = ranked_hits[0]
                drug_id = selected_hit.get("id")
                detail_query = """
                query DrugDetail($chemblId: String!) {
                    drug(chemblId: $chemblId) {
                        id
                        name
                        mechanismsOfAction { rows { mechanismOfAction targets { id approvedSymbol } } }
                        indications { rows { disease { id name } } }
                        maximumClinicalTrialPhase
                    }
                }
                """
                detail_response = module.requests.post(
                    base_url,
                    json={"query": detail_query, "variables": {"chemblId": drug_id}},
                    timeout=10,
                )
                if detail_response.status_code == 200:
                    drug_detail = detail_response.json().get("data", {}).get("drug", {})
                    if drug_detail:
                        formatted_ot = {"identity": selected_identity, "id": drug_id, "name": selected_hit.get("name")}
                        mechs = drug_detail.get("mechanismsOfAction", {}).get("rows", [])
                        mech_lines = []
                        for mech in mechs:
                            moa = mech.get("mechanismOfAction", "Unknown")
                            targets = [target.get("approvedSymbol") for target in mech.get("targets", []) if target.get("approvedSymbol")]
                            target_str = f" (Targets: {', '.join(targets)})" if targets else ""
                            mech_lines.append(f"{moa}{target_str}")
                        if mech_lines:
                            formatted_ot["mechanismsOfAction"] = "; ".join(mech_lines)

                        inds = drug_detail.get("indications", {}).get("rows", [])
                        ind_names = [item.get("disease", {}).get("name") for item in inds if item.get("disease", {}).get("name")]
                        if ind_names:
                            formatted_ot["indications"] = ", ".join(ind_names)

                        if drug_detail.get("maximumClinicalTrialPhase"):
                            formatted_ot["maxPhase"] = drug_detail.get("maximumClinicalTrialPhase")

                        if not formatted_ot:
                            formatted_ot = None
    except Exception:
        formatted_ot = None
    return formatted_ot, time.perf_counter() - started


def _fetch_primekg(module: Any, drug_name: str) -> tuple[dict[str, Any] | None, float]:
    started = time.perf_counter()
    formatted_pkg: dict[str, Any] | None = None
    if str(os.environ.get("AGENT_TOOLS_USE_PRIMEKG", "1") or "1").strip().lower() not in {"1", "true", "yes", "on"}:
        return None, time.perf_counter() - started
    try:
        if "_query_primekg_drug_info" in getattr(module, "__dict__", {}):
            primekg_info = module._query_primekg_drug_info(drug_name)
            if primekg_info:
                formatted_pkg = {}
                if primekg_info.get("targets"):
                    formatted_pkg["targets"] = ", ".join(primekg_info["targets"])
                if primekg_info.get("indications"):
                    formatted_pkg["indications"] = ", ".join(primekg_info["indications"])
                if primekg_info.get("interactions"):
                    formatted_pkg["interactions"] = ", ".join(primekg_info["interactions"])
                if not formatted_pkg:
                    formatted_pkg = None
    except Exception:
        formatted_pkg = None
    return formatted_pkg, time.perf_counter() - started


def _get_all_info_by_drug_impl(drug_name: str, context: Any = None) -> dict[str, Any]:
    try:
        module = _load_curemas_module()
        cache = _get_source_cache(module)
        cache_key = str(drug_name or "").strip().lower()
        raw_payload = cache.get(cache_key)
        record_tool_substep("get_all_info_by_drug", "cache_lookup", 0.0, {"cache_hit": raw_payload is not None})
        if raw_payload is None and cache_key in cache:
            return {
                "success": False,
                "error": f"Drug information not found: {drug_name}",
                "drug_name": drug_name,
            }

        if raw_payload is None:
            with ThreadPoolExecutor(max_workers=3) as executor:
                fda_future = executor.submit(_fetch_fda_label, module, drug_name)
                ot_future = executor.submit(_fetch_opentargets, module, drug_name)
                pkg_future = executor.submit(_fetch_primekg, module, drug_name)
                fda_data, fda_elapsed, fda_identity, fda_source_url = fda_future.result()
                opentargets_data, ot_elapsed = ot_future.result()
                primekg_data, pkg_elapsed = pkg_future.result()

            record_tool_substep(
                "get_all_info_by_drug",
                "fetch_fda_label",
                fda_elapsed,
                {"found": bool(fda_data)},
            )
            record_tool_substep(
                "get_all_info_by_drug",
                "fetch_opentargets",
                ot_elapsed,
                {"found": bool(opentargets_data)},
            )
            record_tool_substep(
                "get_all_info_by_drug",
                "fetch_primekg",
                pkg_elapsed,
                {"found": bool(primekg_data)},
            )

            raw_payload = {
                "drug_name": drug_name,
                "fda_label_raw": fda_data,
                "fda_identity": fda_identity,
                "fda_source_url": fda_source_url,
                "opentargets": opentargets_data,
                "primekg": primekg_data,
            }
            if not fda_data and not opentargets_data and not primekg_data:
                cache[cache_key] = None
                return {
                    "success": False,
                    "error": f"Drug information not found: {drug_name}",
                    "drug_name": drug_name,
                }
            cache[cache_key] = raw_payload

        assemble_started = time.perf_counter()
        result: dict[str, Any] = {"drug_name": drug_name, "sources": []}
        fda_data = raw_payload.get("fda_label_raw")
        fda_identity = raw_payload.get("fda_identity") if isinstance(raw_payload.get("fda_identity"), dict) else {}
        fda_source_url = str(raw_payload.get("fda_source_url", "") or "").strip()
        opentargets_data = raw_payload.get("opentargets")
        primekg_data = raw_payload.get("primekg")

        if isinstance(fda_data, dict):
            runtime_config = _runtime_config()
            tokenizer_name = str(runtime_config.get("qwen_tokenizer_name", "Qwen/Qwen2.5-7B-Instruct") or "Qwen/Qwen2.5-7B-Instruct").strip()
            question_context = _extract_question_context(context)
            reasoning_context = _extract_reasoning_context(context)
            previous_state = _extract_previous_tool_state(drug_name, context)
            used_keys = list(previous_state.get("used_keys", []))
            retrieval_round = int(previous_state.get("round", 0) or 0) + 1
            excluded_keys = {key for key in _FDA_ALWAYS_USELESS_KEYS if key in fda_data}
            previews = _build_fda_preview_map(
                fda_data=fda_data,
                tokenizer_name=tokenizer_name,
                preview_tokens=100,
                excluded_keys=excluded_keys,
                used_keys=used_keys,
            )
            selection_started = time.perf_counter()
            selected_keys, selector_raw = _select_relevant_fda_keys(
                drug_name=drug_name,
                question_context=question_context,
                reasoning_context=reasoning_context,
                key_previews=previews,
            )
            record_tool_substep(
                "get_all_info_by_drug",
                "select_fda_keys",
                time.perf_counter() - selection_started,
                {
                    "candidate_key_count": len(previews),
                    "selected_key_count": len(selected_keys),
                    "retrieval_round": retrieval_round,
                },
            )
            filtered_fda = {
                key: _normalize_label_value(fda_data.get(key))
                for key in selected_keys
                if key in fda_data and _normalize_label_value(fda_data.get(key)) not in ("", [], {}, None)
            }
            used_key_union: list[str] = []
            for key in used_keys + list(filtered_fda.keys()):
                if key not in used_key_union:
                    used_key_union.append(key)
            remaining_keys = [key for key in previews.keys() if key not in filtered_fda]
            result["fda_label"] = filtered_fda
            result["fda_identity"] = fda_identity
            result["fda_source_url"] = fda_source_url
            set_id = str(fda_data.get("set_id", "") or "").strip()
            if set_id:
                result["dailymed_source_url"] = f"https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid={quote_plus(set_id)}"
            result["fda_label_selected_keys"] = list(filtered_fda.keys())
            result["fda_label_used_keys"] = used_key_union
            result["fda_label_retrieval_round"] = retrieval_round
            result["fda_label_total_key_count"] = len(fda_data)
            result["fda_label_candidate_key_count"] = len(previews)
            result["fda_label_excluded_keys"] = sorted(excluded_keys)
            result["fda_label_remaining_key_count"] = len(remaining_keys)
            result["fda_label_remaining_keys_preview"] = remaining_keys[:20]
            result["fda_label_key_previews"] = {key: previews[key] for key in list(filtered_fda.keys())}
            result["question_context"] = question_context
            result["reasoning_context"] = reasoning_context
            result["fda_selector_raw"] = selector_raw
            result["sources"].append("FDA")
        if opentargets_data:
            result["opentargets"] = opentargets_data
            result["sources"].append("OpenTargets")
        if primekg_data:
            result["primekg"] = primekg_data
            result["sources"].append("PrimeKG")

        if not result.get("fda_label") and not result.get("opentargets") and not result.get("primekg"):
            record_tool_substep(
                "get_all_info_by_drug",
                "assemble_response",
                time.perf_counter() - assemble_started,
                {"found_any_source": False},
            )
            return {
                "success": False,
                "error": f"Drug information not found: {drug_name}",
                "drug_name": drug_name,
            }

        record_tool_substep(
            "get_all_info_by_drug",
            "assemble_response",
            time.perf_counter() - assemble_started,
            {"found_any_source": True, "source_count": len(result["sources"])},
        )
        output = {"success": True}
        output.update(result)
        return output
    except Exception as exc:
        return {
            "success": False,
            "error": f"Failed to fetch drug information: {exc}",
            "drug_name": drug_name,
        }


def get_all_info_by_drug(drug_name: str, context: Any = None) -> dict[str, Any]:
    if _should_isolate_get_all_info_process():
        return _get_all_info_by_drug_subprocess(drug_name, context=context)
    return _get_all_info_by_drug_impl(drug_name, context=context)


def _main_child() -> int:
    request = json.loads(sys.stdin.read() or "{}")
    result = _get_all_info_by_drug_impl(
        str(request.get("drug_name", "") or ""),
        context=request.get("context"),
    )
    sys.stdout.write(f"{_CHILD_JSON_START}\n")
    sys.stdout.write(json.dumps(result, ensure_ascii=False))
    sys.stdout.write(f"\n{_CHILD_JSON_END}\n")
    sys.stdout.flush()
    return 0


class GetAllInfoByDrugTool(FunctionTool):
    name = "get_all_info_by_drug"
    description = (
        "Retrieve treatment-focused drug information, centered on drug-label evidence, from the cureMAS multi-source implementation."
    )
    input_hint = 'requires {"drug_name":"<string>"}'
    tool_function = staticmethod(get_all_info_by_drug)
    passes_context = True

    def build_kwargs(self, params: dict[str, Any]) -> dict[str, Any]:
        return {"drug_name": self.require_str(params, "drug_name")}


if __name__ == "__main__":
    raise SystemExit(_main_child())
