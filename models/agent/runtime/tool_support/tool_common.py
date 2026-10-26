from __future__ import annotations

import os
from typing import Any, Dict, List, Optional
import re

from .trial_providers import CtGovProvider, OpenFDAProvider, OrangeBookProvider, _session_with_retry
from .support_tools import (
    AuthoritativeWebSearchProvider,
    DeterministicMedicationCalculator,
    PrimeKGProvider,
    normalize_feature_terms,
    normalize_medical_query_text,
)


OPENTARGETS_GRAPHQL_URL = "https://api.platform.opentargets.org/api/v4/graphql"
HPO_OLS_SEARCH_URL = "https://www.ebi.ac.uk/ols4/api/search"
SESSION = _session_with_retry()
GUIDANCE_SEARCH = AuthoritativeWebSearchProvider()
DOSE_CALCULATOR = DeterministicMedicationCalculator()
PRIMEKG_PROVIDER = PrimeKGProvider()


def _parse_bool(value: Any, default: bool = True) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return default


def is_primekg_enabled() -> bool:
    env_value = os.getenv("AGENT_TOOLS_USE_PRIMEKG")
    if env_value is not None:
        return _parse_bool(env_value, default=True)

    config_value = None
    agent_config_path = os.path.join(os.path.dirname(__file__), "..", "config", "agent.yaml")
    agent_config_path = os.path.abspath(agent_config_path)
    if os.path.exists(agent_config_path):
        try:
            import yaml

            with open(agent_config_path, "r", encoding="utf-8") as handle:
                payload = yaml.safe_load(handle) or {}
            if isinstance(payload, dict):
                config_value = payload.get("use_primekg")
        except Exception:
            config_value = None
    return _parse_bool(config_value, default=True)


def normalize_space(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def truncate(text: Any, max_chars: int = 640) -> str:
    cleaned = normalize_space(text)
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[: max_chars - 3].rstrip() + "..."


def unique(items: List[Any], limit: Optional[int] = None) -> List[str]:
    out: List[str] = []
    for item in items:
        text = normalize_space(item)
        if text and text not in out:
            out.append(text)
        if limit is not None and len(out) >= limit:
            break
    return out


def render_lines(lines: List[str], max_chars: int = 640) -> str:
    content = "\n".join(line for line in lines if normalize_space(line))
    return truncate(content, max_chars=max_chars)


def compact_sources(sources: List[Any], limit: int = 4) -> List[str]:
    return unique([str(source) for source in sources], limit=limit)


def compact_payload(
    summary: str,
    bullets: List[str],
    sources: List[Any],
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    rendered_context = render_lines([summary, *bullets], max_chars=520)
    data = {"rendered_context": rendered_context}
    if extra:
        data.update(extra)
    return {
        "success": True,
        "data": data,
        "sources": compact_sources(sources),
    }


def merge_rendered_sections(sections: List[str], max_chars: int = 520) -> str:
    deduped: List[str] = []
    for section in sections:
        text = normalize_space(section)
        if text and text not in deduped:
            deduped.append(text)
    return render_lines(deduped, max_chars=max_chars)


def safe_primekg_lookup(entity_name: str, allowed_types: List[str], limit: int = 3) -> Dict[str, Any]:
    if not is_primekg_enabled():
        return {
            "success": False,
            "entity_name": entity_name,
            "primekg_status": "disabled_by_config",
            "snippets": [],
            "sources": [],
        }
    try:
        return PRIMEKG_PROVIDER.lookup_entity_context(entity_name=entity_name, allowed_types=allowed_types, limit=limit)
    except Exception as exc:
        return {"success": False, "error": str(exc), "primekg_status": "lookup_failed", "snippets": []}


def post_graphql(query: str, variables: Dict[str, Any]) -> Dict[str, Any]:
    resp = SESSION.post(
        OPENTARGETS_GRAPHQL_URL,
        json={"query": query, "variables": variables},
        timeout=25,
    )
    resp.raise_for_status()
    payload = resp.json()
    errors = payload.get("errors") or []
    if errors:
        raise RuntimeError(str(errors[0].get("message", "Unknown GraphQL error")))
    return payload.get("data") or {}


def search_disease(query_text: str, size: int = 5) -> List[Dict[str, Any]]:
    query = """
    query SearchDisease($q: String!, $size: Int!) {
      search(queryString: $q, entityNames: ["disease"], page: {index: 0, size: $size}) {
        hits {
          id
          name
          entity
        }
      }
    }
    """
    data = post_graphql(query, {"q": query_text, "size": size})
    search = data.get("search") or {}
    hits = search.get("hits") or []
    return [hit for hit in hits if isinstance(hit, dict)]


def get_disease_detail(disease_id: str) -> Dict[str, Any]:
    query = """
    query Disease($id: String!) {
      disease(efoId: $id) {
        id
        name
        description
        therapeuticAreas { id name }
        phenotypes { rows { phenotypeHPO { id name } } }
        knownDrugs {
          count
          rows {
            phase
            status
            mechanismOfAction
            drug { id name maximumClinicalTrialPhase }
          }
        }
        associatedTargets(page: {index: 0, size: 5}) {
          rows {
            score
            target { approvedSymbol approvedName }
          }
        }
      }
    }
    """
    data = post_graphql(query, {"id": disease_id})
    disease = data.get("disease")
    if not isinstance(disease, dict):
        raise RuntimeError(f"Disease not found in OpenTargets: {disease_id}")
    return disease


def search_hpo(term: str, rows: int = 3) -> List[Dict[str, Any]]:
    resp = SESSION.get(
        HPO_OLS_SEARCH_URL,
        params={"q": term, "ontology": "hp", "rows": rows},
        timeout=25,
    )
    resp.raise_for_status()
    payload = resp.json()
    docs = (((payload or {}).get("response") or {}).get("docs")) or []
    return [doc for doc in docs if isinstance(doc, dict)]


def extract_openfda_text(drug_name: str) -> str:
    provider = OpenFDAProvider()
    base = provider.fetch_drug_info(drug_name)
    if not isinstance(base, dict):
        return ""
    record = ((base.get("results") or []) or [None])[0]
    if not isinstance(record, dict):
        return ""

    fields = [
        "indications_and_usage",
        "dosage_and_administration",
        "contraindications",
        "warnings",
        "warnings_and_cautions",
        "adverse_reactions",
        "drug_interactions",
        "use_in_specific_populations",
        "pregnancy",
        "lactation",
    ]
    parts: List[str] = []
    for field in fields:
        value = record.get(field)
        if isinstance(value, list):
            text = " ".join(normalize_space(item) for item in value if normalize_space(item))
        else:
            text = normalize_space(value)
        if text:
            parts.append(f"{field}: {text}")
    return " ".join(parts)


def matches_terms(text: str, terms: List[str]) -> List[str]:
    haystack = text.lower()
    matches = []
    for term in terms:
        needle = normalize_space(term).lower()
        if needle and needle in haystack and needle not in matches:
            matches.append(needle)
    return matches
