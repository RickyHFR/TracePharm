from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
from typing import Any, Iterable
import unicodedata
from urllib.parse import parse_qs, unquote, urlparse

from bs4 import BeautifulSoup

from .trial_providers import _session_with_retry


PROJECT_ROOT = Path(__file__).resolve().parents[4]
PRIMEKG_CACHE_DIR = Path(os.getenv("PRIMEKG_CACHE_DIR", str(PROJECT_ROOT / ".runs/cache/primekg"))).resolve()
PRIMEKG_CACHE_DIR.mkdir(parents=True, exist_ok=True)
PRIMEKG_ENTITY_CACHE_DIR = PRIMEKG_CACHE_DIR / "entity_cache"
PRIMEKG_ENTITY_CACHE_DIR.mkdir(parents=True, exist_ok=True)

PRIMEKG_CSV_URL = "https://dataverse.harvard.edu/api/access/datafile/6180620"
PRIMEKG_DEFAULT_PATHS = [
    os.getenv("PRIMEKG_CSV_PATH", ""),
    str(PRIMEKG_CACHE_DIR / "kg.csv"),
    str(PROJECT_ROOT / "PrimeKG" / "kg.csv"),
]

AUTHORITATIVE_DOMAINS = [
    "fda.gov",
    "nih.gov",
    "ncbi.nlm.nih.gov",
    "cdc.gov",
    "who.int",
    "nice.org.uk",
    "acog.org",
    "heart.org",
    "clinicaltrials.gov",
    "ema.europa.eu",
]

MULTILINGUAL_REPLACEMENTS = {
    "antibiotico": "antibiotic",
    "antibioticos": "antibiotics",
    "amoxicilina": "amoxicillin",
    "dose recommandee": "recommended dose",
    "dosis recomendada": "recommended dose",
    "dosis": "dose",
    "dosificacion": "dosing",
    "encefalite japonesa": "japanese encephalitis",
    "encephalite japonaise": "japanese encephalitis",
    "embarazada": "pregnant",
    "gravidez": "pregnancy",
    "herbal remedy": "herbal remedy",
    "injecao": "injection",
    "injecao": "injection",
    "migraña": "migraine",
    "nao": "not",
    "newborn": "newborn",
    "neonato": "newborn",
    "pneumonie": "pneumonia",
    "recem nascido": "newborn",
    "recem-nascido": "newborn",
    "ressalte": "emphasize",
    "travelers": "travelers",
    "ulceras": "ulcers",
    "vitamina k": "vitamin k",
}

_SESSION = _session_with_retry()
_SPACE_RE = re.compile(r"\s+")
_UNIT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(mcg|μg|ug|mg|g|kg|ml|l)\b", re.IGNORECASE)
_FREQ_RE = re.compile(r"every\s+(\d+(?:\.\d+)?)(?:\s*-\s*(\d+(?:\.\d+)?))?\s*(minutes?|hours?|days?)", re.IGNORECASE)
_QFREQ_RE = re.compile(r"\bq(\d+(?:\.\d+)?)(h|hr|hrs|hour|hours|d|day|days|min|mins|minute|minutes)\b", re.IGNORECASE)


def _normalize_space(text: Any) -> str:
    return _SPACE_RE.sub(" ", str(text or "")).strip()


def strip_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def normalize_medical_query_text(text: str) -> str:
    normalized = strip_accents(_normalize_space(text)).lower()
    for source, target in sorted(MULTILINGUAL_REPLACEMENTS.items(), key=lambda item: len(item[0]), reverse=True):
        normalized = normalized.replace(source, target)
    return _normalize_space(normalized)


def normalize_feature_terms(features: Iterable[str]) -> list[str]:
    cleaned: list[str] = []
    for feature in features:
        text = normalize_medical_query_text(str(feature))
        if text and text not in cleaned:
            cleaned.append(text)
    return cleaned


def _truncate(text: Any, max_chars: int = 360) -> str:
    cleaned = _normalize_space(text)
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[: max_chars - 3].rstrip() + "..."


class DeterministicMedicationCalculator:
    UNIT_FACTORS = {
        "mcg": 0.001,
        "μg": 0.001,
        "ug": 0.001,
        "mg": 1.0,
        "g": 1000.0,
        "kg": 1000000.0,
    }

    def summarize(self, query: str) -> dict[str, Any]:
        normalized = normalize_medical_query_text(query)
        conversions: list[str] = []
        for amount_text, unit_text in _UNIT_RE.findall(normalized):
            unit = unit_text.lower()
            amount = float(amount_text)
            if unit in self.UNIT_FACTORS:
                mg_value = amount * self.UNIT_FACTORS[unit]
                conversions.append(f"{amount:g} {unit_text} = {mg_value:g} mg")
            elif unit == "l":
                conversions.append(f"{amount:g} L = {amount * 1000:g} mL")
            elif unit == "ml":
                conversions.append(f"{amount:g} mL = {amount / 1000:g} L")

        frequencies: list[str] = []
        for low_text, high_text, unit_text in _FREQ_RE.findall(normalized):
            low = float(low_text)
            high = float(high_text) if high_text else low
            unit = unit_text.lower()
            if unit.startswith("minute"):
                per_day_low = 1440 / high
                per_day_high = 1440 / low
            elif unit.startswith("hour"):
                per_day_low = 24 / high
                per_day_high = 24 / low
            else:
                per_day_low = 1 / high
                per_day_high = 1 / low
            if abs(per_day_low - per_day_high) < 1e-9:
                frequencies.append(f"every {low:g} {unit_text} ~= {per_day_low:g} times/day")
            else:
                frequencies.append(
                    f"every {low:g}-{high:g} {unit_text} ~= {per_day_low:g}-{per_day_high:g} times/day"
                )

        for amount_text, unit_text in _QFREQ_RE.findall(normalized):
            interval = float(amount_text)
            unit = unit_text.lower()
            if unit.startswith("h"):
                per_day = 24 / interval
                frequencies.append(f"q{amount_text}{unit_text} ~= {per_day:g} times/day")
            elif unit.startswith("d"):
                frequencies.append(f"q{amount_text}{unit_text} ~= {1 / interval:g} times/day")
            else:
                frequencies.append(f"q{amount_text}{unit_text} ~= {1440 / interval:g} times/day")

        lines = []
        if conversions:
            lines.append("Unit conversions: " + "; ".join(dict.fromkeys(conversions)))
        if frequencies:
            lines.append("Frequency conversions: " + "; ".join(dict.fromkeys(frequencies)))
        if not lines:
            lines.append("No deterministic unit or frequency conversion pattern was detected in the query.")

        return {
            "success": True,
            "data": {
                "rendered_context": _truncate(" ".join(lines), max_chars=320),
                "normalized_query": normalized,
                "conversions": list(dict.fromkeys(conversions)),
                "frequencies": list(dict.fromkeys(frequencies)),
            },
            "sources": ["deterministic-local-calculator"],
        }


class AuthoritativeWebSearchProvider:
    def __init__(self, allowed_domains: list[str] | None = None) -> None:
        self.allowed_domains = allowed_domains or AUTHORITATIVE_DOMAINS
        self.session = _SESSION

    def _allowed(self, url: str, preferred_domains: list[str] | None = None) -> bool:
        netloc = urlparse(url).netloc.lower()
        domains = preferred_domains or self.allowed_domains
        return any(netloc == domain or netloc.endswith(f".{domain}") for domain in domains)

    def _resolve_duckduckgo_url(self, raw_url: str) -> str:
        parsed = urlparse(raw_url)
        if parsed.netloc.endswith("duckduckgo.com"):
            uddg = parse_qs(parsed.query).get("uddg")
            if uddg:
                return unquote(uddg[0])
        return raw_url

    def _fetch_excerpt(self, url: str, max_chars: int = 220) -> str:
        try:
            response = self.session.get(url, timeout=15)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            text_blocks = []
            for tag in soup.find_all(["h1", "h2", "p"]):
                text = _normalize_space(tag.get_text(" ", strip=True))
                if text:
                    text_blocks.append(text)
                if len(" ".join(text_blocks)) >= max_chars:
                    break
            return _truncate(" ".join(text_blocks), max_chars=max_chars)
        except Exception:
            return ""

    def search(self, query: str, max_results: int = 3, preferred_domains: list[str] | None = None) -> list[dict[str, str]]:
        normalized = normalize_medical_query_text(query)
        domains = preferred_domains or self.allowed_domains
        site_filter = " OR ".join(f"site:{domain}" for domain in domains[:6])
        full_query = f"{normalized} ({site_filter})" if site_filter else normalized
        response = self.session.get(
            "https://html.duckduckgo.com/html/",
            params={"q": full_query},
            timeout=20,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        results: list[dict[str, str]] = []
        for anchor in soup.select("a.result__a"):
            href = self._resolve_duckduckgo_url(anchor.get("href") or "")
            if not href or not self._allowed(href, preferred_domains=domains):
                continue
            snippet_tag = anchor.find_parent(class_="result")
            snippet = ""
            if snippet_tag is not None:
                snippet_node = snippet_tag.select_one(".result__snippet")
                if snippet_node is not None:
                    snippet = _truncate(snippet_node.get_text(" ", strip=True), max_chars=180)
            excerpt = self._fetch_excerpt(href)
            results.append(
                {
                    "title": _truncate(anchor.get_text(" ", strip=True), max_chars=120),
                    "url": href,
                    "snippet": snippet,
                    "excerpt": excerpt,
                }
            )
            if len(results) >= max_results:
                break
        return results


class PrimeKGProvider:
    def __init__(self) -> None:
        self.session = _SESSION
        self._header_cache: dict[str, list[str]] = {}

    def _resolve_local_csv_path(self) -> Path | None:
        for candidate in PRIMEKG_DEFAULT_PATHS:
            if not candidate:
                continue
            path = Path(candidate).expanduser().resolve()
            if path.exists():
                return path
        return None

    def _cache_path(self, entity_name: str, allowed_types: list[str] | None = None) -> Path:
        suffix = ",".join(sorted(allowed_types or []))
        key = hashlib.md5(f"{normalize_medical_query_text(entity_name)}|{suffix}".encode("utf-8")).hexdigest()
        return PRIMEKG_ENTITY_CACHE_DIR / f"{key}.json"

    def _load_cache(self, path: Path) -> dict[str, Any] | None:
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def _save_cache(self, path: Path, payload: dict[str, Any]) -> None:
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _row_matches(self, row: dict[str, str], normalized_entity: str, allowed_types: set[str]) -> tuple[bool, str]:
        x_type = normalize_medical_query_text(row.get("x_type", ""))
        y_type = normalize_medical_query_text(row.get("y_type", ""))
        if allowed_types and x_type not in allowed_types and y_type not in allowed_types:
            return False, ""

        x_name = normalize_medical_query_text(row.get("x_name", ""))
        y_name = normalize_medical_query_text(row.get("y_name", ""))
        for candidate in [x_name, y_name]:
            if not candidate:
                continue
            if candidate == normalized_entity or normalized_entity in candidate or candidate in normalized_entity:
                return True, candidate
        return False, ""

    def _format_row(self, row: dict[str, str], matched_name: str) -> str:
        relation = _normalize_space(row.get("display_relation") or row.get("relation"))
        x_name = _normalize_space(row.get("x_name"))
        y_name = _normalize_space(row.get("y_name"))
        x_type = _normalize_space(row.get("x_type"))
        y_type = _normalize_space(row.get("y_type"))
        return f"{x_name} [{x_type}] --{relation or 'related to'}--> {y_name} [{y_type}]"

    def _scan_rows(self, rows: Iterable[dict[str, str]], entity_name: str, allowed_types: list[str] | None, limit: int) -> dict[str, Any]:
        normalized_entity = normalize_medical_query_text(entity_name)
        type_filter = {normalize_medical_query_text(item) for item in (allowed_types or []) if item}
        snippets: list[str] = []
        sources: list[str] = []
        rows_scanned = 0
        exact_match_count = 0

        for row in rows:
            rows_scanned += 1
            matched, matched_name = self._row_matches(row, normalized_entity, type_filter)
            if not matched:
                continue
            if matched_name == normalized_entity:
                exact_match_count += 1
            snippet = self._format_row(row, matched_name)
            if snippet not in snippets:
                snippets.append(snippet)
            if len(snippets) >= limit and exact_match_count >= max(1, limit // 2):
                break

        return {
            "success": bool(snippets),
            "entity_name": entity_name,
            "snippets": snippets[:limit],
            "sources": sources or [PRIMEKG_CSV_URL],
            "rows_scanned": rows_scanned,
            "primekg_status": "live_query",
        }

    def _scan_local_csv(self, csv_path: Path, entity_name: str, allowed_types: list[str] | None, limit: int) -> dict[str, Any]:
        fast_result = self._fast_scan_local_csv(csv_path, entity_name=entity_name, allowed_types=allowed_types, limit=limit)
        if fast_result is not None:
            fast_result["sources"] = [str(csv_path)]
            fast_result["primekg_status"] = "local_csv_rg"
            return fast_result
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            result = self._scan_rows(reader, entity_name=entity_name, allowed_types=allowed_types, limit=limit)
        result["sources"] = [str(csv_path)]
        result["primekg_status"] = "local_csv"
        return result

    def _csv_fieldnames(self, csv_path: Path) -> list[str]:
        key = str(csv_path)
        if key in self._header_cache:
            return list(self._header_cache[key])
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            fieldnames = next(reader, [])
        self._header_cache[key] = list(fieldnames)
        return list(fieldnames)

    def _fast_scan_local_csv(
        self,
        csv_path: Path,
        entity_name: str,
        allowed_types: list[str] | None,
        limit: int,
    ) -> dict[str, Any] | None:
        normalized_entity = normalize_medical_query_text(entity_name)
        if not normalized_entity:
            return None

        try:
            rg_result = subprocess.run(
                [
                    "rg",
                    "-i",
                    "-F",
                    "-m",
                    str(max(20, limit * 25)),
                    entity_name,
                    str(csv_path),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
        except Exception:
            return None

        if rg_result.returncode not in {0, 1}:
            return None

        matched_lines = [line for line in rg_result.stdout.splitlines() if line.strip()]
        if not matched_lines:
            return None

        fieldnames = self._csv_fieldnames(csv_path)
        if not fieldnames:
            return None

        reader = csv.DictReader(matched_lines, fieldnames=fieldnames)
        result = self._scan_rows(reader, entity_name=entity_name, allowed_types=allowed_types, limit=limit)
        if not result.get("success"):
            return None
        result["rows_scanned_fast_path"] = len(matched_lines)
        return result

    def _scan_remote_csv(self, entity_name: str, allowed_types: list[str] | None, limit: int) -> dict[str, Any]:
        response = self.session.get(
            PRIMEKG_CSV_URL,
            stream=True,
            timeout=90,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        response.raise_for_status()
        try:
            reader = csv.DictReader(
                line for line in response.iter_lines(decode_unicode=True) if line
            )
            return self._scan_rows(reader, entity_name=entity_name, allowed_types=allowed_types, limit=limit)
        finally:
            response.close()

    def lookup_entity_context(
        self,
        entity_name: str,
        allowed_types: list[str] | None = None,
        limit: int = 4,
    ) -> dict[str, Any]:
        cache_path = self._cache_path(entity_name, allowed_types=allowed_types)
        local_csv = self._resolve_local_csv_path()
        cached = self._load_cache(cache_path)
        if cached and not (
            local_csv is not None and str(cached.get("primekg_status", "")).strip() == "live_query"
        ):
            return cached

        if local_csv is not None:
            payload = self._scan_local_csv(local_csv, entity_name=entity_name, allowed_types=allowed_types, limit=limit)
        else:
            payload = self._scan_remote_csv(entity_name=entity_name, allowed_types=allowed_types, limit=limit)

        self._save_cache(cache_path, payload)
        return payload
