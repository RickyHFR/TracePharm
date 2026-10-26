from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Iterable, List, Tuple


_TOOL_ARG_DEFAULTS: dict[str, dict[str, Any]] = {
    "search_local": {
        "require_rag": True,
        "dense_k": 5,
        "bm25_k": 5,
        "max_evidence": 10,
    },
    "web_search": {
        "max_results": 10,
    },
}


def _clean_argument_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        return text or None
    if isinstance(value, list):
        cleaned: list[Any] = []
        for item in value:
            normalized = _clean_argument_value(item)
            if normalized is None:
                continue
            cleaned.append(normalized)
        return cleaned or None
    if isinstance(value, dict):
        cleaned_dict: dict[str, Any] = {}
        for key, item in value.items():
            normalized = _clean_argument_value(item)
            if normalized is None:
                continue
            cleaned_dict[str(key)] = normalized
        return cleaned_dict or None
    return value


def compact_tool_arguments(tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    defaults = _TOOL_ARG_DEFAULTS.get(tool_name, {})
    compacted: dict[str, Any] = {}
    for key, value in arguments.items():
        cleaned = _clean_argument_value(value)
        if cleaned is None:
            continue
        if key in defaults and cleaned == defaults[key]:
            continue
        compacted[key] = cleaned
    return compacted


def format_tool_call_payload(tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "name": tool_name,
        "arguments": compact_tool_arguments(tool_name, arguments),
    }


def render_tool_call_block(tool_calls: Iterable[Tuple[str, Dict[str, Any]]]) -> str:
    rendered_lines = [
        "TOOL_CALL: "
        + json.dumps(
            format_tool_call_payload(tool_name, arguments),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        for tool_name, arguments in tool_calls
    ]
    return "\n".join(rendered_lines)


def _render_search_local_result(payload: Dict[str, Any]) -> str:
    data = payload.get("data")
    if not isinstance(data, dict):
        return "No local evidence was found."

    rendered = str(data.get("rendered_context", "") or "").strip()
    if rendered:
        return rendered

    lines: list[str] = []
    query_text = str(data.get("query_text", "") or "").strip()
    if query_text:
        lines.append(f"Local evidence search for: {query_text}")

    extracted_entities = data.get("extracted_entities") or []
    if extracted_entities:
        lines.append("Extracted entities: " + ", ".join(str(item) for item in extracted_entities[:12]))

    resolved_entities = data.get("resolved_entities") or []
    if resolved_entities:
        lines.append("Resolved entities: " + ", ".join(str(item) for item in resolved_entities[:12]))

    evidence_items = data.get("evidence_items") or []
    if evidence_items:
        lines.append("Evidence sentences:")
        for index, item in enumerate(evidence_items[:8], start=1):
            if not isinstance(item, dict):
                continue
            sentence = str(item.get("sentence", "") or "").strip()
            title = str(item.get("title", "") or "").strip()
            url = str(item.get("url", "") or "").strip()
            header = " | ".join(part for part in [title, url] if part)
            if header and sentence:
                lines.append(f"{index}. {header}\n{sentence}")
            elif sentence:
                lines.append(f"{index}. {sentence}")
    elif not lines:
        return "No local evidence was found."
    else:
        lines.append("No local evidence was found.")

    return "\n".join(lines)


def _render_web_search_result(payload: Dict[str, Any]) -> str:
    data = payload.get("data")
    if not isinstance(data, dict):
        return "No search results were returned."
    rendered = str(data.get("rendered_context", "") or "").strip()
    return rendered or "No search results were returned."


def _render_web_visit_result(payload: Dict[str, Any]) -> str:
    data = payload.get("data")
    if not isinstance(data, dict):
        return "The webpage could not be summarized."
    rendered = str(data.get("rendered_context", "") or "").strip()
    return rendered or "The webpage could not be summarized."


def _truncate_text(value: Any, max_chars: int = 320) -> str:
    text = str(value or "").strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def _render_full_label_value(value: Any) -> str:
    if isinstance(value, list):
        return "\n".join(f"- {str(item).strip()}" for item in value if str(item).strip())
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, indent=2)
    return str(value or "").strip()


def _first_nonempty_string(values: Any) -> str:
    if isinstance(values, list):
        for item in values:
            text = str(item or "").strip()
            if text:
                return text
        return ""
    return str(values or "").strip()


def _render_drug_info_result(payload: Dict[str, Any]) -> str:
    drug_name = str(payload.get("drug_name", "") or "").strip()
    lines: list[str] = []
    if drug_name:
        lines.append(f"Drug information for: {drug_name}")

    sources = [str(item).strip() for item in (payload.get("sources") or []) if str(item).strip()]
    if sources:
        lines.append("Sources: " + ", ".join(sources))

    fda_label = payload.get("fda_label")
    selected_keys = [
        str(item).strip()
        for item in (payload.get("fda_label_selected_keys") or [])
        if str(item).strip()
    ]
    if isinstance(fda_label, dict):
        keys_to_render = selected_keys or [str(key).strip() for key in fda_label.keys() if str(key).strip()]
        if keys_to_render:
            lines.append("FDA selected keys: " + ", ".join(keys_to_render))
        for key in keys_to_render:
            value = fda_label.get(key)
            body = _render_full_label_value(value)
            if body:
                lines.append(f"FDA {key.replace('_', ' ')}:\n{body}")

    opentargets = payload.get("opentargets")
    if isinstance(opentargets, dict):
        mechanism = str(opentargets.get("mechanismsOfAction", "") or "").strip()
        indications = str(opentargets.get("indications", "") or "").strip()
        max_phase = str(opentargets.get("maxPhase", "") or "").strip()
        if mechanism:
            lines.append(f"OpenTargets mechanism: {_truncate_text(mechanism, max_chars=240)}")
        if indications:
            lines.append(f"OpenTargets indications: {_truncate_text(indications, max_chars=240)}")
        if max_phase:
            lines.append(f"OpenTargets max phase: {max_phase}")

    primekg = payload.get("primekg")
    if isinstance(primekg, dict):
        targets = str(primekg.get("targets", "") or "").strip()
        indications = str(primekg.get("indications", "") or "").strip()
        interactions = str(primekg.get("interactions", "") or "").strip()
        if targets:
            lines.append(f"PrimeKG targets: {_truncate_text(targets, max_chars=240)}")
        if indications:
            lines.append(f"PrimeKG indications: {_truncate_text(indications, max_chars=240)}")
        if interactions:
            lines.append(f"PrimeKG interactions: {_truncate_text(interactions, max_chars=240)}")

    return "\n".join(lines) if lines else "No compact drug information was returned."


def _render_trial_info_result(payload: Dict[str, Any]) -> str:
    lines: list[str] = []
    nct_id = str(payload.get("nct_id", "") or "").strip()
    if nct_id:
        lines.append(f"Trial: {nct_id}")
    title = str(payload.get("brief_title", "") or payload.get("official_title", "") or "").strip()
    if title:
        lines.append(f"Title: {_truncate_text(title, max_chars=500)}")
    status = str(payload.get("recruitment_status", "") or "").strip()
    sponsor = str(payload.get("sponsor", "") or "").strip()
    ingredients = [str(item).strip() for item in (payload.get("ingredients") or []) if str(item).strip()]
    sources = [str(item).strip() for item in (payload.get("sources") or []) if str(item).strip()]
    if status:
        lines.append(f"Recruitment status: {status}")
    if sponsor:
        lines.append(f"Sponsor: {sponsor}")
    if ingredients:
        lines.append("Source-derived ingredient candidates: " + ", ".join(ingredients[:20]))

    dates = payload.get("study_dates")
    if isinstance(dates, dict):
        date_parts: list[str] = []
        for key in ("start", "primary_completion", "completion"):
            item = dates.get(key)
            if isinstance(item, dict) and str(item.get("date") or "").strip():
                date_parts.append(f"{key}={item.get('date')} ({item.get('type') or 'type not stated'})")
        if date_parts:
            lines.append("Study dates: " + "; ".join(date_parts))

    interventions = payload.get("interventions") or []
    if isinstance(interventions, list) and interventions:
        lines.append("Registry interventions:")
        for index, item in enumerate(interventions[:20], start=1):
            if not isinstance(item, dict):
                continue
            parts = [str(item.get("type") or "").strip(), str(item.get("name") or "").strip()]
            arm_labels = [str(value).strip() for value in item.get("arm_group_labels") or [] if str(value).strip()]
            aliases = [str(value).strip() for value in item.get("other_names") or [] if str(value).strip()]
            if arm_labels:
                parts.append("arms=" + ", ".join(arm_labels[:8]))
            if aliases:
                parts.append("aliases=" + ", ".join(aliases[:6]))
            lines.append(f"{index}. " + "; ".join(part for part in parts if part))

    regimen_arms = payload.get("regimen_arms") or []
    if isinstance(regimen_arms, list) and regimen_arms:
        lines.append("Regimen arms (registry-linked; efficacy is not guessed):")
        for index, arm in enumerate(regimen_arms[:16], start=1):
            if not isinstance(arm, dict):
                continue
            interventions_text = ", ".join(
                str(value).strip() for value in arm.get("intervention_names") or [] if str(value).strip()
            )
            description = _truncate_text(arm.get("description"), max_chars=360)
            parts = [
                str(arm.get("label") or "").strip(),
                str(arm.get("type") or "").strip(),
                f"interventions={interventions_text}" if interventions_text else "",
                f"description={description}" if description else "",
            ]
            lines.append(f"{index}. " + "; ".join(part for part in parts if part))

    registered = payload.get("registered_outcomes") or {}
    if isinstance(registered, dict):
        primary = registered.get("primary") or []
        if primary:
            lines.append("Registered primary outcomes:")
            for index, outcome in enumerate(primary[:8], start=1):
                if not isinstance(outcome, dict):
                    continue
                measure = _truncate_text(outcome.get("measure"), max_chars=360)
                time_frame = _truncate_text(outcome.get("time_frame"), max_chars=240)
                lines.append(f"{index}. {measure}" + (f"; time frame={time_frame}" if time_frame else ""))

    reported = payload.get("reported_outcomes") or []
    if isinstance(reported, list) and reported:
        lines.append("Reported arm-level outcome results:")
        for index, outcome in enumerate(reported[:8], start=1):
            if not isinstance(outcome, dict):
                continue
            heading_parts = [
                str(outcome.get("type") or "").strip(),
                _truncate_text(outcome.get("title"), max_chars=420),
                f"units={outcome.get('units')}" if outcome.get("units") else "",
                f"parameter={outcome.get('param_type')}" if outcome.get("param_type") else "",
                f"time frame={_truncate_text(outcome.get('time_frame'), max_chars=220)}"
                if outcome.get("time_frame")
                else "",
            ]
            lines.append(f"Outcome {index}: " + "; ".join(part for part in heading_parts if part))
            measurement_count = 0
            for result_class in outcome.get("classes") or []:
                if not isinstance(result_class, dict):
                    continue
                for category in result_class.get("categories") or []:
                    if not isinstance(category, dict):
                        continue
                    category_title = _truncate_text(category.get("title"), max_chars=220)
                    for measurement in category.get("measurements") or []:
                        if not isinstance(measurement, dict) or measurement_count >= 12:
                            continue
                        group = str(measurement.get("group_title") or measurement.get("group_id") or "unknown group")
                        value = str(measurement.get("value") or "not reported")
                        interval = ""
                        if measurement.get("lower_limit") or measurement.get("upper_limit"):
                            interval = f"; interval={measurement.get('lower_limit') or '?'} to {measurement.get('upper_limit') or '?'}"
                        spread = f"; spread={measurement.get('spread')}" if measurement.get("spread") else ""
                        category_prefix = f"{category_title}: " if category_title else ""
                        lines.append(f"  - {category_prefix}{group} = {value}{spread}{interval}")
                        measurement_count += 1
            for analysis in list(outcome.get("analyses") or [])[:3]:
                if not isinstance(analysis, dict):
                    continue
                parts = [
                    f"groups={','.join(str(item) for item in analysis.get('group_ids') or [])}",
                    f"{analysis.get('param_type')}={analysis.get('param_value')}"
                    if analysis.get("param_value")
                    else "",
                    f"p={analysis.get('p_value')}" if analysis.get("p_value") else "",
                    f"method={analysis.get('method')}" if analysis.get("method") else "",
                ]
                lines.append("  - analysis: " + "; ".join(part for part in parts if part))

    references = payload.get("publication_references") or []
    if isinstance(references, list) and references:
        lines.append("Registry-linked publication references:")
        for index, reference in enumerate(references[:10], start=1):
            if not isinstance(reference, dict):
                continue
            parts = [
                f"PMID {reference.get('pmid')}" if reference.get("pmid") else "",
                str(reference.get("type") or "").strip(),
                _truncate_text(reference.get("citation"), max_chars=500),
                str(reference.get("url") or "").strip(),
            ]
            lines.append(f"{index}. " + "; ".join(part for part in parts if part))

    if payload.get("reported_outcomes_truncated"):
        lines.append(
            "Reported outcomes were bounded for context: "
            f"returned {len(reported)}, registry total {payload.get('reported_outcome_count')}."
        )
    if sources:
        lines.append("Sources: " + ", ".join(sources[:3]))
    return "\n".join(lines) if lines else "No compact trial information was returned."


def _render_list_items(label: str, items: Any, fields: List[str], max_items: int = 8) -> str:
    if not isinstance(items, list) or not items:
        return f"{label}: none found"
    lines = [f"{label}:"]
    for index, item in enumerate(items[:max_items], start=1):
        if not isinstance(item, dict):
            lines.append(f"{index}. {_truncate_text(item, max_chars=240)}")
            continue
        parts = []
        for field in fields:
            value = str(item.get(field, "") or "").strip()
            if value:
                parts.append(f"{field}={value}")
        if parts:
            lines.append(f"{index}. " + "; ".join(parts))
    return "\n".join(lines)


def _render_source_line(payload: Dict[str, Any]) -> str:
    urls: list[str] = []
    for item in payload.get("source_records") or []:
        if isinstance(item, dict):
            value = str(item.get("url") or item.get("source_url") or "").strip()
            if value.startswith(("http://", "https://")):
                urls.append(value)
    for item in payload.get("sources") or []:
        value = str(item or "").strip()
        if value.startswith(("http://", "https://")):
            urls.append(value)
    unique_urls = list(dict.fromkeys(urls))[:3]
    return "Sources: " + ", ".join(unique_urls) if unique_urls else ""


def _with_source_line(body: str, payload: Dict[str, Any]) -> str:
    source_line = _render_source_line(payload)
    return body + ("\n" + source_line if source_line else "")


def _selection_context(payload: Dict[str, Any]) -> str:
    parts: list[str] = []
    if payload.get("as_of_date"):
        parts.append(f"as_of_date={payload.get('as_of_date')}")
    if payload.get("latest_approval_date"):
        parts.append(f"latest_linked_approval_date={payload.get('latest_approval_date')}")
    if payload.get("latest_approval_only"):
        parts.append("latest_approval_only=true")
    if payload.get("earliest_patent_expiry_date") or payload.get("latest_patent_expiry_date"):
        parts.append(
            "returned_expiry_range="
            f"{payload.get('earliest_patent_expiry_date') or '?'} to "
            f"{payload.get('latest_patent_expiry_date') or '?'}"
        )
    return "Selection context: " + "; ".join(parts) if parts else ""


def _render_structured_top_level_result(tool_name: str, payload: Dict[str, Any]) -> str:
    if tool_name == "get_all_info_by_drug":
        return _render_drug_info_result(payload)
    if tool_name == "get_trial_info":
        return _render_trial_info_result(payload)
    if tool_name == "get_drug_patents":
        ingredients = ", ".join(str(item).strip() for item in (payload.get("ingredients") or []) if str(item).strip())
        heading = f"Patent information for: {ingredients}" if ingredients else "Patent information"
        return _with_source_line(
            heading
            + ("\n" + _selection_context(payload) if _selection_context(payload) else "")
            + "\n"
            + _render_list_items(
                "Patents",
                payload.get("patents"),
                [
                    "product_name",
                    "active_ingredient",
                    "application_number",
                    "product_number",
                    "approval_date",
                    "number",
                    "expiry_date",
                    "use_code",
                    "notes",
                ],
            ),
            payload,
        )
    if tool_name == "get_drug_approvals":
        ingredients = ", ".join(str(item).strip() for item in (payload.get("ingredients") or []) if str(item).strip())
        heading = f"Approval information for: {ingredients}" if ingredients else "Approval information"
        return _with_source_line(
            heading
            + ("\n" + _selection_context(payload) if _selection_context(payload) else "")
            + "\n"
            + _render_list_items(
                "Approvals",
                payload.get("approvals"),
                [
                    "product_name",
                    "active_ingredient",
                    "application_number",
                    "product_number",
                    "approval_date",
                    "marketing_authorisation_company",
                    "applicant_abbreviation",
                    "dosage_form_route",
                    "strength",
                ],
            ),
            payload,
        )
    if tool_name == "get_drug_exlusivities":
        ingredients = ", ".join(str(item).strip() for item in (payload.get("ingredients") or []) if str(item).strip())
        heading = f"Exclusivity information for: {ingredients}" if ingredients else "Exclusivity information"
        return _with_source_line(
            heading
            + ("\n" + _selection_context(payload) if _selection_context(payload) else "")
            + "\n"
            + _render_list_items(
                "Exclusivities",
                payload.get("exclusivities"),
                [
                    "product_name",
                    "active_ingredient",
                    "application_number",
                    "product_number",
                    "approval_date",
                    "type",
                    "end_date",
                    "notes",
                ],
            ),
            payload,
        )
    return ""


def render_tool_result_body(tool_name: str, payload: Dict[str, Any]) -> str:
    if not bool(payload.get("success", False)):
        error_text = str(payload.get("error", "") or "Tool execution failed.").strip()
        return f"Error: {error_text}"

    if tool_name == "search_local":
        return _render_search_local_result(payload)
    if tool_name == "web_search":
        return _render_web_search_result(payload)
    if tool_name == "web_visit":
        return _render_web_visit_result(payload)

    data = payload.get("data")
    if isinstance(data, dict):
        rendered = str(data.get("rendered_context", "") or "").strip()
        if rendered:
            return rendered

    structured = _render_structured_top_level_result(tool_name, payload).strip()
    if structured:
        return structured

    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def render_tool_result_message(tool_name: str, payload: Dict[str, Any]) -> str:
    header = json.dumps({"name": tool_name}, ensure_ascii=False, separators=(",", ":"))
    body = render_tool_result_body(tool_name, payload).strip()
    if not body:
        return f"TOOL_RESULT: {header}"
    return f"TOOL_RESULT: {header}\n{body}"


def bound_model_visible_text(text: Any, *, max_chars: int) -> tuple[str, Dict[str, Any]]:
    """Deterministically cap text sent back to a model and fingerprint the full text.

    The caller can persist the returned audit mapping while keeping the complete
    raw tool payload in a separate audit channel.  Character counting is used
    deliberately: it is deterministic across platforms and independent of a
    tokenizer implementation or model revision.
    """

    rendered = str(text or "")
    limit = int(max_chars)
    if limit < 160:
        raise ValueError("max_model_visible_tool_result_chars must be at least 160")

    full_bytes = rendered.encode("utf-8")
    full_sha256 = hashlib.sha256(full_bytes).hexdigest()
    truncated = len(rendered) > limit
    if truncated:
        marker = (
            "\n...[MODEL_VISIBLE_TOOL_RESULT_TRUNCATED "
            f"full_sha256={full_sha256} full_chars={len(rendered)} "
            f"full_utf8_bytes={len(full_bytes)}]"
        )
        prefix_chars = max(0, limit - len(marker))
        visible = rendered[:prefix_chars].rstrip() + marker
        # rstrip() can only make the result shorter, but keep the invariant
        # explicit in case the marker format changes later.
        visible = visible[:limit]
    else:
        visible = rendered

    return visible, {
        "schema_version": 1,
        "limit_unit": "unicode_codepoints",
        "max_model_visible_chars": limit,
        "truncated": truncated,
        "full_rendered_sha256": full_sha256,
        "full_rendered_chars": len(rendered),
        "full_rendered_utf8_bytes": len(full_bytes),
        "model_visible_chars": len(visible),
        "model_visible_utf8_bytes": len(visible.encode("utf-8")),
    }
