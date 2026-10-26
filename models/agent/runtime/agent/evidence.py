from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from typing import Any
from urllib.parse import urlparse


_TIER_1_DOMAINS = {
    "fda.gov",
    "accessdata.fda.gov",
    "dailymed.nlm.nih.gov",
    "clinicaltrials.gov",
    "cdc.gov",
    "who.int",
    "nice.org.uk",
    "ema.europa.eu",
    "ec.europa.eu",
    "canada.ca",
    "tga.gov.au",
    "pmda.go.jp",
    "gov.uk",
    "medlineplus.gov",
}
_TIER_2_DOMAINS = {
    "nih.gov",
    "ncbi.nlm.nih.gov",
    "pubmed.ncbi.nlm.nih.gov",
    "nejm.org",
    "jamanetwork.com",
    "thelancet.com",
    "bmj.com",
    "cochranelibrary.com",
    "nature.com",
    "science.org",
    "cell.com",
}

_TIER_1_SOURCE_TYPES = {
    "official_label",
    "regulator",
    "regulatory_database",
    "fda_orange_book",
    "guideline",
    "trial_registry",
    "government_public_health",
}
_TIER_2_SOURCE_TYPES = {
    "peer_reviewed",
    "systematic_review",
    "structured_biomedical_database",
}


def _text(value: Any, limit: int = 8000) -> str:
    return str(value or "").strip()[:limit]


def _domain(url: str) -> str:
    try:
        host = urlparse(url).netloc.lower().split(":", 1)[0]
    except Exception:
        return ""
    return host[4:] if host.startswith("www.") else host


def authority_tier(url: str, source_type: str = "") -> tuple[int, str]:
    domain = _domain(url)
    source_type = str(source_type or "").lower()
    if source_type == "search_snippet":
        return 3, "uninspected_search_snippet"
    if source_type in _TIER_1_SOURCE_TYPES:
        return 1, "governed_primary_source"
    if any(domain == item or domain.endswith("." + item) for item in _TIER_1_DOMAINS):
        return 1, "governed_primary_source"
    if source_type in _TIER_2_SOURCE_TYPES:
        return 2, "curated_or_peer_reviewed_source"
    if any(domain == item or domain.endswith("." + item) for item in _TIER_2_DOMAINS):
        return 2, "peer_reviewed_or_public_health_source"
    if url:
        return 3, "other_identified_source"
    return 4, "unidentified_source"


def _record_id(tool_name: str, url: str, content_kind: str, content: str) -> str:
    payload = f"{tool_name}\n{url}\n{content_kind}\n{content}".encode("utf-8", errors="ignore")
    return "ev_" + hashlib.sha256(payload).hexdigest()[:16]


def make_evidence_record(
    *,
    tool_name: str,
    quote: Any = "",
    fact: Any = "",
    url: Any = "",
    title: Any = "",
    source_type: str = "",
    content_kind: str = "",
    entity: Any = "",
    goal: Any = "",
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    # Selected evidence is an indivisible source unit. Metadata may be bounded,
    # but cutting source content can remove a late exception or dose threshold.
    quote_text = str(quote or "").strip()
    fact_text = str(fact or "").strip()
    if not quote_text and not fact_text:
        return None
    content_kind = content_kind or ("verbatim_quote" if quote_text else "structured_fact")
    content = quote_text or fact_text
    url_text = _text(url, 2000)
    tier, tier_label = authority_tier(url_text, source_type=source_type)
    return {
        "record_id": _record_id(tool_name, url_text, content_kind, content),
        "tool": tool_name,
        "title": _text(title, 500),
        "url": url_text,
        "source_type": source_type or "unknown",
        "authority_tier": tier,
        "authority_label": tier_label,
        "quote": quote_text,
        "fact": fact_text,
        "content_kind": content_kind,
        "entity": _text(entity, 300),
        "goal": _text(goal, 600),
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "metadata": dict(metadata or {}),
    }


def _records_from_search_local(data: dict[str, Any], goal: str) -> list[dict[str, Any]]:
    records = []
    for item in list(data.get("evidence_items") or [])[:20]:
        if not isinstance(item, dict):
            continue
        sentence = str(item.get("sentence") or "").strip()
        parts = [(sentence, "verbatim_quote")]
        if sentence.startswith("[Source-attributed summary; non-verbatim]"):
            summary, _, quote = sentence.partition("[Verbatim source excerpt]")
            parts = [(summary, "source_attributed_summary"), (quote, "verbatim_quote")]
        for content, content_kind in parts:
            is_quote = content_kind == "verbatim_quote"
            record = make_evidence_record(
                tool_name="search_local",
                quote=content if is_quote else "",
                fact=content if not is_quote else "",
                url=item.get("url"),
                title=item.get("title"),
                source_type=str(item.get("source_type") or (
                    "curated_local_quote" if is_quote else "curated_local_summary"
                )),
                content_kind=content_kind,
                entity=item.get("entity_name"),
                goal=goal,
                metadata={
                    "doc_id": item.get("doc_id"),
                    "benchmark": item.get("benchmark"),
                    "question_index": item.get("question_index"),
                    "source_csv": item.get("source_csv"),
                    "contradiction": item.get("contradiction", False),
                },
            )
            if record:
                records.append(record)
    return records


def _records_from_web_search(data: dict[str, Any], goal: str) -> list[dict[str, Any]]:
    result_groups: list[list[Any]] = []
    if isinstance(data.get("results"), list):
        result_groups.append(data["results"])
    for batch in list(data.get("batches") or []):
        if isinstance(batch, dict) and isinstance(batch.get("results"), list):
            result_groups.append(batch["results"])
    records = []
    for group in result_groups:
        for item in group[:10]:
            if not isinstance(item, dict):
                continue
            record = make_evidence_record(
                tool_name="web_search",
                fact=item.get("snippet"),
                url=item.get("link"),
                title=item.get("title"),
                source_type="search_snippet",
                goal=goal,
                metadata={
                    "rank": item.get("rank"),
                    "date": item.get("date"),
                    "page_quote_verified": False,
                },
            )
            if record:
                records.append(record)
    return records


def _records_from_web_visit(data: dict[str, Any], goal: str) -> list[dict[str, Any]]:
    items = list(data.get("results") or []) if isinstance(data.get("results"), list) else [data]
    records = []
    for item in items:
        if not isinstance(item, dict):
            continue
        # Item-level failures include pages rejected by the label-free
        # benchmark-mirror filter.  Never admit them to the ledger even if a
        # malformed provider payload happens to carry an evidence-looking
        # field alongside the failure marker.
        if item.get("success") is False:
            continue
        quote_verified = bool(item.get("quote_verified", False))
        verified_quote = str(item.get("evidence") or "").strip() if quote_verified else ""
        # The model-generated page summary and rationale are not citations.
        # They remain only in the raw tool payload for audit and must not enter
        # this ledger because the ledger is shown to the reflection model.
        if not verified_quote:
            continue
        record = make_evidence_record(
            tool_name="web_visit",
            quote=verified_quote,
            url=item.get("url"),
            title="",
            source_type="webpage_quote",
            goal=item.get("goal") or goal,
            metadata={
                "quote_verified": True,
                "quote_verification_method": "exact_page_span",
                "content_sha256": _text(item.get("content_sha256"), 128),
                "unverified_evidence_discarded": bool(item.get("unverified_evidence_discarded", False)),
            },
        )
        if record:
            records.append(record)
    return records


def _source_entries(data: dict[str, Any], default_source_type: str) -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []
    raw_source_records = data.get("source_records") or []
    if isinstance(raw_source_records, dict):
        source_records = [raw_source_records]
    elif isinstance(raw_source_records, (list, tuple)):
        source_records = list(raw_source_records)
    else:
        source_records = []
    for item in source_records:
        if not isinstance(item, dict):
            continue
        url = _text(item.get("url") or item.get("source_url"), 2000)
        if url:
            entries.append(
                {
                    "url": url,
                    "title": _text(item.get("title"), 500),
                    "source_type": _text(item.get("source_type"), 100) or default_source_type,
                }
            )
    raw_sources = data.get("sources") or []
    if isinstance(raw_sources, (str, dict)):
        sources = [raw_sources]
    elif isinstance(raw_sources, (list, tuple)):
        sources = list(raw_sources)
    else:
        sources = []
    for item in sources:
        if isinstance(item, dict):
            url = _text(item.get("url") or item.get("source_url"), 2000)
            title = _text(item.get("title"), 500)
            source_type = _text(item.get("source_type"), 100) or default_source_type
        else:
            url = _text(item, 2000)
            title = ""
            source_type = default_source_type
        if url:
            entries.append({"url": url, "title": title, "source_type": source_type})

    deduped: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for entry in entries:
        key = (entry["url"], entry["source_type"])
        if key not in seen:
            seen.add(key)
            deduped.append(entry)
    return deduped


def _first_source(
    data: dict[str, Any],
    *,
    default_source_type: str,
    default_url: str = "",
    default_title: str = "",
) -> dict[str, str]:
    entries = _source_entries(data, default_source_type)
    if entries:
        return entries[0]
    return {
        "url": default_url,
        "title": default_title,
        "source_type": default_source_type,
    }


def _entity_list(data: dict[str, Any], key: str = "ingredients") -> str:
    value = data.get(key)
    if isinstance(value, list):
        return ", ".join(_text(item, 100) for item in value if _text(item, 100))[:300]
    return _text(value, 300)


def _records_from_orange_book(tool_name: str, data: dict[str, Any], goal: str) -> list[dict[str, Any]]:
    collection_by_tool = {
        "get_drug_approvals": "approvals",
        "get_drug_patents": "patents",
        "get_drug_exlusivities": "exclusivities",
        "get_drug_exclusivities": "exclusivities",
    }
    collection = collection_by_tool[tool_name]
    source = _first_source(
        data,
        default_source_type="fda_orange_book",
        default_url="https://www.fda.gov/drugs/drug-approvals-and-databases/orange-book-data-files",
        default_title="FDA Orange Book Data Files",
    )
    entity = _entity_list(data)
    records: list[dict[str, Any]] = []
    for index, item in enumerate(list(data.get(collection) or [])[:20]):
        if not isinstance(item, dict):
            continue
        item_source_url = _text(item.get("source_url"), 2000) or source["url"]
        item_source_type = _text(item.get("source_type"), 100) or source["source_type"]
        item_entity = _text(item.get("active_ingredient"), 300) or entity
        fact_text = json.dumps(item, ensure_ascii=False)
        record = make_evidence_record(
            tool_name=tool_name,
            fact=fact_text,
            url=item_source_url,
            title=source["title"] or "FDA Orange Book",
            source_type=item_source_type,
            entity=item_entity,
            goal=goal,
            metadata={
                "collection": collection,
                "item_index": index,
                "application_number": _text(item.get("application_number"), 80),
                "product_number": _text(item.get("product_number"), 80),
                "linked_approval_date": _text(item.get("approval_date"), 80),
            },
        )
        if record:
            records.append(record)
    return records


def _records_from_trial(data: dict[str, Any], goal: str) -> list[dict[str, Any]]:
    nct_id = _text(data.get("nct_id"), 100)
    source = _first_source(
        data,
        default_source_type="trial_registry",
        default_url=f"https://clinicaltrials.gov/study/{nct_id}" if nct_id else "",
        default_title=f"ClinicalTrials.gov study {nct_id}" if nct_id else "ClinicalTrials.gov study",
    )
    facts = [
        ("recruitment_status", "recruitment status", data.get("recruitment_status")),
        ("sponsor", "lead sponsor", data.get("sponsor")),
        ("ingredients", "interventions/ingredients", _entity_list(data)),
    ]
    if data.get("study_dates"):
        facts.append(("study_dates", "study dates", json.dumps(data.get("study_dates"), ensure_ascii=False)))
    records: list[dict[str, Any]] = []
    for field, label, value in facts:
        value_text = _text(value, 1200)
        if not value_text:
            continue
        record = make_evidence_record(
            tool_name="get_trial_info",
            fact=f"{nct_id} {label}: {value_text}".strip(),
            url=source["url"],
            title=source["title"],
            source_type=source["source_type"],
            entity=nct_id,
            goal=goal,
            metadata={"field": field},
        )
        if record:
            records.append(record)

    # Keep each selected registry unit intact, including dose exceptions,
    # denominators, confidence intervals, analysis methods and group definitions.
    # These are structured source facts, not model-generated summaries.
    sections = [
        ("intervention", data.get("interventions") or [], 16),
        ("regimen_arm", data.get("regimen_arms") or [], 16),
        ("reported_outcome", data.get("reported_outcomes") or [], 12),
        ("publication_reference", data.get("publication_references") or [], 12),
    ]
    for kind, outcomes in (data.get("registered_outcomes") or {}).items():
        sections.append((f"registered_{kind}_outcome", outcomes, 12))
    for field, items, limit in sections:
        for index, item in enumerate(list(items)[:limit]):
            if not isinstance(item, dict):
                continue
            source_record = item.get("source_record", item)
            record = make_evidence_record(
                tool_name="get_trial_info",
                fact=f"{nct_id} {field}: " + json.dumps(source_record, ensure_ascii=False),
                url=source["url"],
                title=source["title"],
                source_type=source["source_type"],
                entity=nct_id,
                goal=goal,
                metadata={"field": field, "item_index": index},
            )
            if record:
                records.append(record)
    return records




def _records_from_drug_info(result: dict[str, Any], goal: str) -> list[dict[str, Any]]:
    records = []
    drug = _text(result.get("drug_name"), 300)
    identity = result.get("fda_identity") if isinstance(result.get("fda_identity"), dict) else {}
    source_url = _text(result.get("fda_source_url"), 2000)
    for key, value in dict(result.get("fda_label") or {}).items():
        quote = json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else str(value)
        record = make_evidence_record(
            tool_name="get_all_info_by_drug",
            quote=quote,
            url=source_url,
            title=f"FDA label: {drug} — {key}",
            source_type="official_label",
            entity=drug,
            goal=goal,
            metadata={"field": key, "identity": identity},
        )
        if record:
            records.append(record)
    for source_key in ("opentargets", "primekg"):
        value = result.get(source_key)
        if not value:
            continue
        record = make_evidence_record(
            tool_name="get_all_info_by_drug",
            fact=json.dumps(value, ensure_ascii=False),
            url="https://platform.opentargets.org/" if source_key == "opentargets" else "https://primekg.org/",
            title=f"{source_key}: {drug}",
            source_type="structured_biomedical_database",
            entity=drug,
            goal=goal,
        )
        if record:
            records.append(record)
    return records


def normalize_tool_evidence(tool_name: str, result: Any, goal: str = "") -> list[dict[str, Any]]:
    if not isinstance(result, dict) or not bool(result.get("success", False)):
        return []
    data = result.get("data") if isinstance(result.get("data"), dict) else result
    if tool_name == "search_local":
        records = _records_from_search_local(data, goal)
    elif tool_name == "web_search":
        records = _records_from_web_search(data, goal)
    elif tool_name == "web_visit":
        records = _records_from_web_visit(data, goal)
    elif tool_name == "get_all_info_by_drug":
        records = _records_from_drug_info(result, goal)
    elif tool_name in {
        "get_drug_approvals",
        "get_drug_patents",
        "get_drug_exlusivities",
        "get_drug_exclusivities",
    }:
        records = _records_from_orange_book(tool_name, data, goal)
    elif tool_name == "get_trial_info":
        records = _records_from_trial(data, goal)
    else:
        rendered = data.get("rendered_context") if isinstance(data, dict) else ""
        summary = data.get("summary") if isinstance(data, dict) else ""
        source = _first_source(data, default_source_type="structured_tool_output")
        record = make_evidence_record(
            tool_name=tool_name,
            fact=rendered or summary,
            url=source["url"],
            title=source["title"],
            source_type=source["source_type"],
            goal=goal,
        )
        records = [record] if record else []

    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for record in records:
        record_id = str(record.get("record_id", ""))
        if not record_id or record_id in seen:
            continue
        seen.add(record_id)
        deduped.append(record)
    return deduped
