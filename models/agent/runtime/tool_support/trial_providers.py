# -*- coding: utf-8 -*-
"""
真实数据提供者模块
从ClinicalTrials.gov API和FDA Orange Book获取真实数据
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Protocol, Tuple, Iterable
from collections import OrderedDict
import os
import sys
import io
import re
import json
import math
import zipfile
import csv
import hashlib
import time
import fcntl
from functools import lru_cache
from threading import Lock
import requests
from requests.adapters import HTTPAdapter, Retry
from dateutil import parser as dtparser
from bs4 import BeautifulSoup
from urllib.parse import urlencode, quote




# ---------- 常量 ----------
CTGOV_STUDY_URL = "https://clinicaltrials.gov/api/v2/studies/{}"
CTGOV_PAGE_URL = "https://clinicaltrials.gov/study/{}"
OPENFDA_LABEL_URL = "https://api.fda.gov/drug/label.json"

# 官方页（主数据页 + Orange Book 主页，两者任一可解析出下载链接）
ORANGE_BOOK_PAGE = "https://www.fda.gov/drugs/drug-approvals-and-databases/orange-book-data-files"
ORANGE_BOOK_MAIN = "https://www.fda.gov/drugs/drug-approvals-and-databases/approved-drug-products-therapeutic-equivalence-evaluations-orange-book"
_DEFAULT_ORANGE_BOOK_DATA_DIR = os.path.join(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")),
    ".runs",
    "cache",
    "orangebook",
)

# 允许的下载链接模式：绝对或相对
MEDIA_ABS_RE = re.compile(r'https?://www\.fda\.gov/media/\d+/download(?:\?[^"\s>]*)?', re.I)
MEDIA_REL_RE = re.compile(r'^/media/\d+/download(?:\?[^"\s>]*)?$', re.I)

# Keep downloaded data in the ignored run directory; deployments may override it.
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
_DEFAULT_CACHE_DIR = os.path.join(_PROJECT_ROOT, ".runs", "cache", "orangebook")
CACHE_DIR = os.getenv("ORANGE_BOOK_CACHE_DIR", _DEFAULT_CACHE_DIR)
CACHE_MAX_AGE = 24 * 60 * 60 * 365  # 缓存有效期：一年
MAX_RETRIES = 3  # 最大重试次数
RETRY_DELAY = 2  # 重试延迟（秒）

KEY_LOG_DIR = os.path.join(_PROJECT_ROOT, "records")
KEY_LOG_FILE = os.path.join(KEY_LOG_DIR, "openfda_key_selection_log.json")

# ---------- 数据模型 ----------
@dataclass
class Patent:
    jurisdiction: str
    number: str
    expiry_date: Optional[str]
    notes: Optional[str] = None
    application_number: Optional[str] = None
    product_number: Optional[str] = None
    product_name: Optional[str] = None
    active_ingredient: Optional[str] = None
    approval_date: Optional[str] = None
    marketing_authorisation_company: Optional[str] = None
    use_code: Optional[str] = None
    drug_substance_flag: Optional[str] = None
    drug_product_flag: Optional[str] = None
    delist_flag: Optional[str] = None
    submission_date: Optional[str] = None

@dataclass
class Exclusivity:
    region: str
    type: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    notes: Optional[str] = None
    application_number: Optional[str] = None
    product_number: Optional[str] = None
    product_name: Optional[str] = None
    active_ingredient: Optional[str] = None
    approval_date: Optional[str] = None
    marketing_authorisation_company: Optional[str] = None

@dataclass
class Approval:
    region: str
    product_name: str
    active_ingredient: str
    approval_date: Optional[str] = None
    status: Optional[str] = "Approved"
    marketing_authorisation_company: Optional[str] = None
    application_number: Optional[str] = None
    product_number: Optional[str] = None
    application_type: Optional[str] = None
    dosage_form_route: Optional[str] = None
    strength: Optional[str] = None
    applicant_abbreviation: Optional[str] = None

@dataclass
class TrialCommercialProfile:
    nct_id: str
    ingredients: List[str]
    sponsor: str
    is_open_for_enrollment: bool
    recruitment_status: str
    patents: List[Patent]
    exclusivities: List[Exclusivity]
    approvals: List[Approval]
    sources: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

# ---------- Provider 接口 ----------
class TrialProvider(Protocol):
    def can_handle(self, nct_id: str) -> bool: ...
    def fetch_basic_trial(self, nct_id: str) -> Dict[str, Any]: ...

class DrugIntelProvider(Protocol):
    def resolve_by_ingredients(self, ingredients: List[str]) -> Dict[str, Any]: ...

# ---------- 工具函数 ----------
def _session_with_retry() -> requests.Session:
    s = requests.Session()
    retries = Retry(
        total=3, 
        backoff_factor=0.6,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"])
    )
    s.headers.update({
        "User-Agent": "trial-profile-bot/1.1 (+github.com/your-org) Python-requests"
    })
    s.mount("https://", HTTPAdapter(max_retries=retries))
    return s

def _normalize_date(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    value = str(s).strip()
    # ClinicalTrials.gov legitimately publishes year- or month-precision dates.
    # dateutil fills missing components from today's date, which fabricates a
    # day (for example ``2019-01`` could become January 11).  Preserve stated
    # precision for already ISO-like partial dates.
    if re.fullmatch(r"\d{4}(?:-\d{2})?", value):
        return value
    try:
        return dtparser.parse(value).date().isoformat()
    except Exception:
        return value

def _salt_stripper(name: str) -> str:
    # 只保留真正的盐类/酸根后缀，不包含元素名（magnesium、sodium、potassium、calcium）
    salts = {
        "hydrochloride", "sulfate", "phosphate", "mesylate", "maleate", "bitartrate",
        "acetate", "fumarate", "succinate", "tosylate", "besylate", "citrate", "oxalate"
    }
    x = name.lower()
    x = re.sub(r"[()]", " ", x)
    tokens = [t for t in re.split(r"[,\s;]+", x) if t]
    tokens = [t for t in tokens if t not in salts]
    return " ".join(tokens)


def _normalized_drug_tokens(name: str) -> tuple[str, ...]:
    """Normalize a drug identity without using benchmark-specific aliases.

    Orange Book combination products are separated by semicolons.  Matching is
    performed component-by-component below; this helper therefore only removes
    punctuation, common dosage-form words, and salt suffixes.  Token equality is
    deliberately preferred over raw substring matching (for example, ``alpha``
    must not match ``alphabetin``).
    """

    value = _salt_stripper(str(name or ""))
    value = re.sub(r"\b(?:tablet|tablets|capsule|capsules|injection|solution|oral|iv)\b", " ", value)
    value = re.sub(r"[^a-z0-9]+", " ", value.lower())
    return tuple(token for token in value.split() if token)


def _orange_book_components(value: str) -> list[str]:
    return [item.strip() for item in re.split(r"\s*;\s*", str(value or "")) if item.strip()]


def _drug_token_identity_match(left: tuple[str, ...], right: tuple[str, ...]) -> bool:
    if not left or not right:
        return False
    if left == right:
        return True
    shorter, longer = sorted((left, right), key=len)
    if len(longer) - len(shorter) > 2:
        return False
    return longer[: len(shorter)] == shorter or longer[-len(shorter) :] == shorter

def _match_ob_ingredient(targets: Iterable[str], ob_ingredient: str) -> bool:
    ob_components = [_normalized_drug_tokens(item) for item in _orange_book_components(ob_ingredient)]
    ob_components = [item for item in ob_components if item]
    if not ob_components:
        return False

    for target in targets:
        target_tokens = _normalized_drug_tokens(target)
        if not target_tokens:
            continue
        for component_tokens in ob_components:
            # A registry intervention often adds a brand name in parentheses or
            # a short presentation suffix. Permit a whole-token prefix/suffix
            # match, never an arbitrary character substring.
            if _drug_token_identity_match(target_tokens, component_tokens):
                return True
    return False


def _clean_registry_name(value: str) -> str:
    value = re.sub(r"\s+", " ", str(value or "")).strip(" ,;:")
    return value[:300]


def _intervention_identity_candidates(name: str, other_names: Iterable[str] = ()) -> List[str]:
    """Return source-derived identity candidates, with no trial-specific map."""

    raw_candidates: list[str] = [name, *list(other_names or [])]
    for raw in list(raw_candidates):
        raw_candidates.extend(re.findall(r"\(([^()]{2,100})\)", str(raw or "")))
        raw_candidates.append(re.sub(r"\([^()]*\)", " ", str(raw or "")))
        raw_candidates.extend(re.split(r"\s+(?:\+|vs\.?|versus)\s+|\s*/\s*", str(raw or ""), flags=re.I))

    out: list[str] = []
    seen: set[str] = set()
    for candidate in raw_candidates:
        cleaned = _clean_registry_name(candidate)
        normalized = " ".join(_normalized_drug_tokens(cleaned))
        if not cleaned or not normalized or normalized in seen:
            continue
        seen.add(normalized)
        out.append(cleaned)
    return out


def _guess_ingredients_from_interventions(interventions: Iterable[Dict[str, Any]]) -> List[str]:
    out: list[str] = []
    seen: set[str] = set()
    for intervention in interventions:
        if not isinstance(intervention, dict):
            continue
        intervention_type = str(intervention.get("type") or "").upper()
        if intervention_type and intervention_type not in {"DRUG", "BIOLOGICAL", "DIETARY_SUPPLEMENT", "OTHER"}:
            continue
        candidates = _intervention_identity_candidates(
            str(intervention.get("name") or ""),
            intervention.get("otherNames") or [],
        )
        for candidate in candidates:
            key = " ".join(_normalized_drug_tokens(candidate))
            if key and key not in seen:
                seen.add(key)
                out.append(candidate)
    return out


def _short_text(value: Any, limit: int = 1200) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return text[:limit]


def _date_struct(module: Dict[str, Any], key: str) -> Dict[str, str]:
    raw = module.get(key) if isinstance(module.get(key), dict) else {}
    date = _normalize_date(str(raw.get("date") or "")) or ""
    return {
        "date": date,
        "type": _short_text(raw.get("type"), 80),
    }


def _protocol_outcomes(module: Dict[str, Any], key: str) -> List[Dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for item in list(module.get(key) or []):
        if not isinstance(item, dict):
            continue
        records.append(
            {
                "measure": _short_text(item.get("measure"), 400),
                "description": _short_text(item.get("description"), 1200),
                "time_frame": _short_text(item.get("timeFrame"), 400),
                "source_record": dict(item),
            }
        )
    return records


def _reported_outcomes(results_section: Dict[str, Any]) -> List[Dict[str, Any]]:
    module = results_section.get("outcomeMeasuresModule") or {}
    records: list[dict[str, Any]] = []
    for measure in list(module.get("outcomeMeasures") or []):
        if not isinstance(measure, dict):
            continue
        groups: list[dict[str, str]] = []
        group_titles: dict[str, str] = {}
        for group in list(measure.get("groups") or []):
            if not isinstance(group, dict):
                continue
            group_id = _short_text(group.get("id"), 80)
            title = _short_text(group.get("title"), 300)
            groups.append(
                {
                    "id": group_id,
                    "title": title,
                    "description": _short_text(group.get("description"), 1000),
                }
            )
            if group_id:
                group_titles[group_id] = title

        classes: list[dict[str, Any]] = []
        for result_class in list(measure.get("classes") or []):
            if not isinstance(result_class, dict):
                continue
            categories: list[dict[str, Any]] = []
            for category in list(result_class.get("categories") or []):
                if not isinstance(category, dict):
                    continue
                measurements: list[dict[str, Any]] = []
                for measurement in list(category.get("measurements") or []):
                    if not isinstance(measurement, dict):
                        continue
                    group_id = _short_text(measurement.get("groupId"), 80)
                    measurements.append(
                        {
                            "group_id": group_id,
                            "group_title": group_titles.get(group_id, ""),
                            "value": _short_text(measurement.get("value"), 120),
                            "spread": _short_text(measurement.get("spread"), 120),
                            "lower_limit": _short_text(measurement.get("lowerLimit"), 120),
                            "upper_limit": _short_text(measurement.get("upperLimit"), 120),
                            "comment": _short_text(measurement.get("comment"), 500),
                        }
                    )
                categories.append(
                    {
                        "title": _short_text(category.get("title"), 300),
                        "measurements": measurements,
                    }
                )
            classes.append({"title": _short_text(result_class.get("title"), 300), "categories": categories})

        analyses: list[dict[str, Any]] = []
        for analysis in list(measure.get("analyses") or []):
            if not isinstance(analysis, dict):
                continue
            analyses.append(
                {
                    "group_ids": [_short_text(item, 80) for item in list(analysis.get("groupIds") or [])],
                    "non_inferiority_type": _short_text(analysis.get("nonInferiorityType"), 120),
                    "param_type": _short_text(analysis.get("paramType"), 120),
                    "param_value": _short_text(analysis.get("paramValue"), 120),
                    "dispersion_type": _short_text(analysis.get("dispersionType"), 120),
                    "dispersion_value": _short_text(analysis.get("dispersionValue"), 120),
                    "p_value": _short_text(analysis.get("pValue"), 120),
                    "ci_percent": _short_text(analysis.get("ciPctValue"), 120),
                    "ci_lower_limit": _short_text(analysis.get("ciLowerLimit"), 120),
                    "ci_upper_limit": _short_text(analysis.get("ciUpperLimit"), 120),
                    "method": _short_text(analysis.get("method"), 300),
                }
            )
        records.append(
            {
                "type": _short_text(measure.get("type"), 80),
                "title": _short_text(measure.get("title"), 500),
                "description": _short_text(measure.get("description"), 1200),
                "time_frame": _short_text(measure.get("timeFrame"), 400),
                "units": _short_text(measure.get("units"), 120),
                "param_type": _short_text(measure.get("paramType"), 120),
                "dispersion_type": _short_text(measure.get("dispersionType"), 120),
                "groups": groups,
                "classes": classes,
                "analyses": analyses,
                "source_record": dict(measure),
            }
        )
    return records

# ---------- ClinicalTrials.gov v2 ----------
class CtGovProvider:
    """从ClinicalTrials.gov API获取真实临床试验数据"""
    
    def __init__(self):
        self.session = _session_with_retry()
    
    def can_handle(self, nct_id: str) -> bool:
        return bool(re.fullmatch(r"NCT\d{8}", str(nct_id or "").strip().upper()))

    def parse_study(self, data: Dict[str, Any], nct_id: str) -> Dict[str, Any]:
        """Parse the public v2 schema into a compact, source-faithful record.

        The previous implementation returned only a flat intervention-name
        list.  That is insufficient for regimen-comparison tasks because the
        arm assignment and reported outcome measurements carry the efficacy
        relation.  This parser retains those relations without deciding which
        arm is "better"; that judgment remains question-dependent.
        """

        psec = data.get("protocolSection", {}) or {}
        if not psec:
            raise ValueError(f"Study not found or invalid response: {nct_id}")

        identification_mod = psec.get("identificationModule", {}) or {}
        status_mod = psec.get("statusModule", {}) or {}
        overall = _short_text(status_mod.get("overallStatus"), 100)
        sponsor_mod = psec.get("sponsorCollaboratorsModule", {}) or {}
        sponsor = _short_text(
            sponsor_mod.get("leadSponsor", {}).get("name")
            or sponsor_mod.get("leadSponsor", {}).get("class"),
            400,
        )

        inter_mod = psec.get("armsInterventionsModule", {}) or {}
        raw_interventions = [item for item in list(inter_mod.get("interventions") or []) if isinstance(item, dict)]
        interventions: list[dict[str, Any]] = []
        intervention_by_arm: dict[str, list[dict[str, Any]]] = {}
        for intervention in raw_interventions:
            name = _clean_registry_name(intervention.get("name") or "")
            other_names = [
                _clean_registry_name(item)
                for item in list(intervention.get("otherNames") or [])
                if _clean_registry_name(item)
            ]
            candidates = _intervention_identity_candidates(name, other_names)
            arm_labels = [
                _clean_registry_name(item)
                for item in list(intervention.get("armGroupLabels") or [])
                if _clean_registry_name(item)
            ]
            record = {
                "type": _short_text(intervention.get("type"), 80),
                "name": name,
                "description": _short_text(intervention.get("description"), 1200),
                "other_names": other_names,
                "arm_group_labels": arm_labels,
                "ingredient_candidates": candidates,
                "source_record": dict(intervention),
            }
            interventions.append(record)
            for label in arm_labels:
                intervention_by_arm.setdefault(label.casefold(), []).append(record)

        arms: list[dict[str, Any]] = []
        for arm in list(inter_mod.get("armGroups") or []):
            if not isinstance(arm, dict):
                continue
            label = _clean_registry_name(arm.get("label") or "")
            linked = intervention_by_arm.get(label.casefold(), [])
            arm_ingredients: list[str] = []
            for intervention in linked:
                for candidate in intervention.get("ingredient_candidates") or []:
                    if candidate and candidate not in arm_ingredients:
                        arm_ingredients.append(candidate)
            listed_names = [
                _clean_registry_name(item)
                for item in list(arm.get("interventionNames") or [])
                if _clean_registry_name(item)
            ]
            if not listed_names:
                listed_names = [item.get("name", "") for item in linked if item.get("name")]
            arms.append(
                {
                    "label": label,
                    "type": _short_text(arm.get("type"), 80),
                    "description": _short_text(arm.get("description"), 1200),
                    "intervention_names": listed_names,
                    "ingredient_candidates": arm_ingredients,
                    "source_record": {
                        "arm": dict(arm),
                        "interventions": [item["source_record"] for item in linked],
                    },
                }
            )

        outcomes_mod = psec.get("outcomesModule", {}) or {}
        design_mod = psec.get("designModule", {}) or {}
        description_mod = psec.get("descriptionModule", {}) or {}
        references_mod = psec.get("referencesModule", {}) or {}
        references: list[dict[str, str]] = []
        for reference in list(references_mod.get("references") or []):
            if not isinstance(reference, dict):
                continue
            pmid = _short_text(reference.get("pmid"), 40)
            references.append(
                {
                    "pmid": pmid,
                    "type": _short_text(reference.get("type"), 80),
                    "citation": _short_text(reference.get("citation"), 1200),
                    "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else "",
                }
            )
        reported_outcomes = _reported_outcomes(data.get("resultsSection", {}) or {})
        ingredients = _guess_ingredients_from_interventions(raw_interventions)

        return {
            "nct_id": str(nct_id).strip().upper(),
            "brief_title": _short_text(identification_mod.get("briefTitle"), 500),
            "official_title": _short_text(identification_mod.get("officialTitle"), 800),
            "brief_summary": _short_text(description_mod.get("briefSummary"), 1800),
            "study_type": _short_text(design_mod.get("studyType"), 100),
            "phases": [_short_text(item, 80) for item in list(design_mod.get("phases") or [])],
            "sponsor": sponsor,
            "recruitment_status": overall,
            "study_dates": {
                "start": _date_struct(status_mod, "startDateStruct"),
                "primary_completion": _date_struct(status_mod, "primaryCompletionDateStruct"),
                "completion": _date_struct(status_mod, "completionDateStruct"),
            },
            "ingredients": ingredients,
            "interventions": interventions,
            "regimen_arms": arms,
            "registered_outcomes": {
                "primary": _protocol_outcomes(outcomes_mod, "primaryOutcomes"),
                "secondary": _protocol_outcomes(outcomes_mod, "secondaryOutcomes"),
                "other": _protocol_outcomes(outcomes_mod, "otherOutcomes"),
            },
            "reported_outcomes": reported_outcomes,
            "publication_references": references,
            "has_results": bool(data.get("hasResults") or reported_outcomes),
            "sources": [CTGOV_PAGE_URL.format(str(nct_id).strip().upper())],
            "source_api_url": CTGOV_STUDY_URL.format(str(nct_id).strip().upper()),
            "source_schema": "ClinicalTrials.gov API v2",
        }
    
    def fetch_basic_trial(self, nct_id: str) -> Dict[str, Any]:
        """获取临床试验的方案、arm 与公开结果结构。"""
        normalized_nct_id = str(nct_id or "").strip().upper()
        if not self.can_handle(normalized_nct_id):
            raise ValueError(f"Unsupported NCT identifier format: {nct_id}")
        r = self.session.get(CTGOV_STUDY_URL.format(normalized_nct_id), timeout=30)
        r.raise_for_status()
        return self.parse_study(r.json(), normalized_nct_id)

# ---------- 缓存工具函数 ----------
def _ensure_cache_dir():
    """确保缓存目录存在"""
    os.makedirs(CACHE_DIR, exist_ok=True)
    return CACHE_DIR

def _get_cache_key(url: str) -> str:
    """根据URL生成缓存键（文件名）"""
    return hashlib.md5(url.encode()).hexdigest() + ".zip"

def _get_cache_path(url: str) -> str:
    """获取缓存文件路径"""
    cache_dir = _ensure_cache_dir()
    cache_key = _get_cache_key(url)
    return os.path.join(cache_dir, cache_key)

def _get_lock_path(url: str) -> str:
    """获取锁文件路径"""
    lock_dir = _ensure_cache_dir()
    cache_key = _get_cache_key(url)
    return os.path.join(lock_dir, cache_key + ".lock")

def _is_cache_valid(cache_path: str) -> bool:
    """检查缓存是否有效（存在且未过期）"""
    if not os.path.exists(cache_path):
        return False
    try:
        age = time.time() - os.path.getmtime(cache_path)
        return age < CACHE_MAX_AGE
    except Exception:
        return False

def _save_to_cache(url: str, content: bytes) -> None:
    """保存内容到缓存"""
    cache_dir = _ensure_cache_dir()
    cache_path = os.path.join(cache_dir, _get_cache_key(url))
    try:
        with open(cache_path, 'wb') as f:
            f.write(content)
    except Exception:
        # 如果保存失败，不影响主流程，只是下次无法使用缓存
        pass

def _find_cache_files() -> List[str]:
    """查找缓存目录中的所有ZIP文件（不需要知道URL）"""
    if not os.path.exists(CACHE_DIR):
        return []
    try:
        cache_files = [f for f in os.listdir(CACHE_DIR) if f.endswith('.zip')]
        return [os.path.join(CACHE_DIR, f) for f in cache_files]
    except Exception:
        return []

def _load_from_cache(url: str) -> Optional[bytes]:
    """从缓存加载内容（需要知道URL）"""
    cache_path = os.path.join(CACHE_DIR, _get_cache_key(url))
    if _is_cache_valid(cache_path):
        try:
            with open(cache_path, 'rb') as f:
                content = f.read()
                # 调试信息：仅在环境变量启用时打印
                if os.getenv("ORANGE_BOOK_DEBUG_CACHE"):
                    print(f"[缓存] 从缓存加载: {cache_path} ({len(content)} bytes)", file=sys.stderr)
                return content
        except Exception as e:
            # 调试信息：仅在环境变量启用时打印
            if os.getenv("ORANGE_BOOK_DEBUG_CACHE"):
                print(f"[缓存] 读取缓存文件失败: {cache_path}, 错误: {e}", file=sys.stderr)
            pass
    else:
        # 调试信息：仅在环境变量启用时打印
        if os.getenv("ORANGE_BOOK_DEBUG_CACHE"):
            if os.path.exists(cache_path):
                print(f"[缓存] 缓存文件已过期: {cache_path}", file=sys.stderr)
            else:
                print(f"[缓存] 缓存文件不存在: {cache_path}, 缓存目录: {CACHE_DIR}", file=sys.stderr)
    return None

def _load_from_cache_any() -> Optional[bytes]:
    """从缓存目录中加载任意ZIP文件（不需要知道URL，用于离线场景）"""
    cache_files = _find_cache_files()
    if not cache_files:
        if os.getenv("ORANGE_BOOK_DEBUG_CACHE"):
            print(f"[缓存] 缓存目录中没有ZIP文件: {CACHE_DIR}", file=sys.stderr)
        return None
    
    # 如果有多个缓存文件，使用第一个（通常只有一个）
    if len(cache_files) > 1:
        if os.getenv("ORANGE_BOOK_DEBUG_CACHE"):
            print(f"[缓存] 找到多个缓存文件，使用第一个: {cache_files[0]}", file=sys.stderr)
    
    cache_path = cache_files[0]
    if _is_cache_valid(cache_path):
        try:
            with open(cache_path, 'rb') as f:
                content = f.read()
                # 调试信息：仅在环境变量启用时打印
                if os.getenv("ORANGE_BOOK_DEBUG_CACHE"):
                    print(f"[缓存] 从缓存加载: {cache_path} ({len(content)} bytes)", file=sys.stderr)
                return content
        except Exception as e:
            if os.getenv("ORANGE_BOOK_DEBUG_CACHE"):
                print(f"[缓存] 读取缓存文件失败: {cache_path}, 错误: {e}", file=sys.stderr)
            pass
    return None

def _acquire_lock(lock_path: str, timeout: int = 300) -> Optional[Any]:
    """获取文件锁（用于多进程同步）"""
    lock_file = None
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            lock_file = open(lock_path, 'w')
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return lock_file
        except (IOError, OSError):
            # 锁被占用，等待后重试
            if lock_file:
                lock_file.close()
            time.sleep(0.5)
    return None

def _release_lock(lock_file: Any) -> None:
    """释放文件锁"""
    if lock_file:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            lock_file.close()
        except Exception:
            pass


@lru_cache(maxsize=16)
def _read_tilde_path_cached(path: str, mtime_ns: int, size: int) -> Tuple[Tuple[str, ...], ...]:
    """Read an immutable table snapshot once per file version."""

    del mtime_ns, size
    with open(path, "r", encoding="utf-8-sig", errors="ignore", newline="") as handle:
        return tuple(tuple(row) for row in csv.reader(handle, delimiter="~"))


@lru_cache(maxsize=16)
def _file_sha256_cached(path: str, mtime_ns: int, size: int) -> str:
    del mtime_ns, size
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _file_sha256(path: str) -> str:
    stat = os.stat(path)
    return _file_sha256_cached(path, stat.st_mtime_ns, stat.st_size)


_ORANGE_BOOK_INDEX_CACHE: "OrderedDict[Tuple[int, int, int], Tuple[Any, Any, Any, Any]]" = OrderedDict()
_ORANGE_BOOK_INDEX_LOCK = Lock()


def _build_orange_book_index(
    products: Iterable[Iterable[str]],
    patents: Iterable[Iterable[str]],
    exclusivities: Iterable[Iterable[str]],
) -> Tuple[
    Dict[Tuple[str, ...], List[Dict[str, Any]]],
    Dict[Tuple[str, str], List[Dict[str, str]]],
    Dict[Tuple[str, str], List[Dict[str, str]]],
]:
    """Build reusable identity and application/product indexes from FDA rows."""

    product_index: Dict[Tuple[str, ...], List[Dict[str, Any]]] = {}
    for row in products:
        row = list(row)
        if len(row) < 14 or row[5].strip() not in {"N", "A"}:
            continue
        ingredient = row[0].strip()
        if not ingredient:
            continue
        product = {
            "application_number": row[6].strip(),
            "product_number": row[7].strip(),
            "active_ingredient": ingredient,
            "product_name": row[2].strip(),
            "approval_date": _normalize_date(row[9].strip()),
            "applicant_abbreviation": row[3].strip(),
            "application_type": row[5].strip(),
            "marketing_authorisation_company": row[13].strip() or row[3].strip(),
            "dosage_form_route": row[1].strip(),
            "strength": row[4].strip(),
            "reference_listed_drug": row[10].strip(),
            "reference_standard": row[11].strip(),
            "marketing_status": row[12].strip(),
        }
        for component in _orange_book_components(ingredient):
            tokens = _normalized_drug_tokens(component)
            if tokens:
                product_index.setdefault(tokens, []).append(product)

    patent_map: Dict[Tuple[str, str], List[Dict[str, str]]] = {}
    for row in patents:
        row = list(row)
        if len(row) < 5 or row[0].strip() not in {"N", "A"}:
            continue
        application_number = row[1].strip()
        product_number = row[2].strip()
        patent_number = row[3].strip()
        if not (application_number and product_number and patent_number):
            continue
        patent_map.setdefault((application_number, product_number), []).append(
            {
                "number": patent_number,
                "expiry_date": _normalize_date(row[4].strip()) or "",
                "drug_substance_flag": row[5].strip() if len(row) > 5 else "",
                "drug_product_flag": row[6].strip() if len(row) > 6 else "",
                "use_code": row[7].strip() if len(row) > 7 else "",
                "delist_flag": row[8].strip() if len(row) > 8 else "",
                "submission_date": (_normalize_date(row[9].strip()) or "") if len(row) > 9 else "",
            }
        )

    exclusivity_map: Dict[Tuple[str, str], List[Dict[str, str]]] = {}
    for row in exclusivities:
        row = list(row)
        if len(row) < 5 or row[0].strip() not in {"N", "A"}:
            continue
        application_number = row[1].strip()
        product_number = row[2].strip()
        code = row[3].strip()
        if not (application_number and product_number and code):
            continue
        exclusivity_map.setdefault((application_number, product_number), []).append(
            {"type": code, "end_date": _normalize_date(row[4].strip()) or ""}
        )
    return product_index, patent_map, exclusivity_map


def _get_orange_book_index(
    products: Any,
    patents: Any,
    exclusivities: Any,
) -> Tuple[
    Dict[Tuple[str, ...], List[Dict[str, Any]]],
    Dict[Tuple[str, str], List[Dict[str, str]]],
    Dict[Tuple[str, str], List[Dict[str, str]]],
]:
    key = (id(products), id(patents), id(exclusivities))
    with _ORANGE_BOOK_INDEX_LOCK:
        cached = _ORANGE_BOOK_INDEX_CACHE.get(key)
        if cached is not None and cached[0] is products and cached[1] is patents and cached[2] is exclusivities:
            _ORANGE_BOOK_INDEX_CACHE.move_to_end(key)
            return cached[3]

    index = _build_orange_book_index(products, patents, exclusivities)
    with _ORANGE_BOOK_INDEX_LOCK:
        _ORANGE_BOOK_INDEX_CACHE[key] = (products, patents, exclusivities, index)
        _ORANGE_BOOK_INDEX_CACHE.move_to_end(key)
        while len(_ORANGE_BOOK_INDEX_CACHE) > 4:
            _ORANGE_BOOK_INDEX_CACHE.popitem(last=False)
    return index


# ---------- Orange Book 提供者 ----------
class OrangeBookProvider:
    """
    从FDA Orange Book ZIP文件获取真实药物数据
    
    解析 Orange Book ZIP：
      1) 先看环境变量 ORANGE_BOOK_ZIP_URL（若已指定则直接下载）
      2) 解析 ORANGE_BOOK_PAGE（数据文件页）找 "compressed (.ZIP) data file" 或 media/{id}/download
      3) 若失败，解析 ORANGE_BOOK_MAIN（主页）里的 "Orange Book Data Files" 跳转再抓
      4) 支持相对链接，允许附带 query 参数
      5) 支持缓存机制，避免重复下载
      6) 支持错误重试机制
    """
    
    def __init__(
        self,
        page_url: str = ORANGE_BOOK_PAGE,
        main_url: str = ORANGE_BOOK_MAIN,
        data_dir: Optional[str] = None,
    ):
        self.page_url = page_url
        self.main_url = main_url
        self.data_dir = data_dir
        self.session = _session_with_retry()

    # ---- 解析策略 ----
    def _extract_media_links(self, html: str, base: str) -> List[str]:
        soup = BeautifulSoup(html, "html.parser")
        links = []
        for a in soup.find_all("a"):
            text = (a.get_text(strip=True) or "").lower()
            href = a.get("href") or ""
            if not href:
                continue
            # 文本优先：compressed (.ZIP) data file / Data Files
            is_text_hit = ("compressed" in text and "zip" in text) or ("data files" in text and "zip" in text)
            # 链接模式：绝对或相对 media/{id}/download
            is_media_abs = bool(MEDIA_ABS_RE.search(href))
            is_media_rel = bool(MEDIA_REL_RE.search(href))
            if is_text_hit or is_media_abs or is_media_rel:
                if is_media_rel:
                    href = requests.compat.urljoin(base, href)
                links.append(href)
        # 去重，保持顺序
        seen, out = set(), []
        for u in links:
            if u not in seen:
                seen.add(u)
                out.append(u)
        return out

    def _resolve_zip_url_from_page(self, url: str) -> Optional[str]:
        resp = self.session.get(url, timeout=40)
        resp.raise_for_status()
        html = resp.text
        cands = self._extract_media_links(html, base=url)
        return cands[0] if cands else None

    def _resolve_zip_url(self) -> str:
        # 0) 环境变量兜底
        env_url = os.getenv("ORANGE_BOOK_ZIP_URL", "").strip()
        if env_url:
            return env_url

        # 1) 直接在数据文件页解析
        url = self._resolve_zip_url_from_page(self.page_url)
        if url:
            return url

        # 2) 在 Orange Book 主页找到 "Orange Book Data Files" 再解析
        resp = self.session.get(self.main_url, timeout=40)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        data_files_href = None
        for a in soup.find_all("a"):
            if "data files" in (a.get_text(strip=True) or "").lower():
                data_files_href = a.get("href") or ""
                if data_files_href:
                    data_files_href = requests.compat.urljoin(self.main_url, data_files_href)
                    break
        if data_files_href:
            url = self._resolve_zip_url_from_page(data_files_href)
            if url:
                return url

        # 3) 失败：抛出带诊断信息的异常
        snippet = (resp.text or "")[:1000].replace("\n", " ")
        raise RuntimeError("Failed to locate Orange Book ZIP link on FDA pages. "
                           f"Tried: {self.page_url} and {self.main_url}. "
                           "If you are behind a proxy/CDN variant, set env ORANGE_BOOK_ZIP_URL to the direct media link. "
                           f"Page snippet: {snippet[:400]}...")

    # ---- 下载与解析 ----
    def _download_zip(self, url: Optional[str] = None) -> bytes:
        """
        下载ZIP文件，支持缓存和重试机制
        
        Args:
            url: ZIP文件URL（如果为None，则自动解析）
        
        Returns:
            ZIP文件内容（bytes）
        """
        # 1. 优先尝试从缓存目录中查找ZIP文件（不需要联网，用于离线场景）
        cached_content = _load_from_cache_any()
        if cached_content is not None:
            return cached_content
        
        # 2. 如果提供了URL或环境变量，尝试使用URL查找缓存
        zip_url = url or os.getenv("ORANGE_BOOK_ZIP_URL", "").strip()
        if zip_url:
            cached_content = _load_from_cache(zip_url)
            if cached_content is not None:
                return cached_content
        
        # 3. 如果没有提供URL，尝试解析（需要联网）
        if not zip_url:
            try:
                zip_url = self._resolve_zip_url()
                # 解析成功后，再次尝试从缓存加载
                cached_content = _load_from_cache(zip_url)
                if cached_content is not None:
                    return cached_content
            except Exception as e:
                # 如果解析失败（可能是网络问题），检查是否有缓存文件
                cache_files = _find_cache_files()
                if cache_files:
                    # 有缓存文件但无法解析URL，尝试直接使用缓存
                    error_msg = f"Failed to resolve Orange Book ZIP URL (network may be unavailable).\n"
                    error_msg += f"  Error: {str(e)}\n"
                    error_msg += f"  Cache directory: {CACHE_DIR}\n"
                    error_msg += f"  Found cache files: {len(cache_files)}\n"
                    error_msg += f"  💡 Tip: Found cache files but cannot verify URL. Trying to use cache directly...\n"
                    # 尝试使用第一个缓存文件
                    try:
                        with open(cache_files[0], 'rb') as f:
                            content = f.read()
                            if zipfile.is_zipfile(io.BytesIO(content)):
                                if os.getenv("ORANGE_BOOK_DEBUG_CACHE"):
                                    print(f"[缓存] 使用缓存文件（无法验证URL）: {cache_files[0]}", file=sys.stderr)
                                return content
                    except Exception:
                        pass
                    raise RuntimeError(error_msg + f"  ❌ Cache files exist but cannot be read. Please check cache directory: {CACHE_DIR}")
                else:
                    # 没有缓存文件且无法联网，直接报错
                    error_msg = f"Failed to resolve Orange Book ZIP URL and no cache found.\n"
                    error_msg += f"  Error: {str(e)}\n"
                    error_msg += f"  Cache directory: {CACHE_DIR}\n"
                    error_msg += f"  ❌ No cache files found in cache directory!\n"
                    error_msg += f"  💡 Tip: Run prefetch_cache.py on a machine with network access to download the ZIP file.\n"
                    error_msg += f"  💡 Tip: Copy the cache directory to this machine.\n"
                    error_msg += f"  💡 Tip: Or set ORANGE_BOOK_ZIP_URL environment variable to skip URL resolution.\n"
                    raise RuntimeError(error_msg)
        
        # 4. 如果到这里，说明需要下载（有URL但缓存不存在）
        lock_path = _get_lock_path(zip_url)
        lock_file = _acquire_lock(lock_path)
        try:
            # 再次检查缓存（可能在等待锁期间被其他进程下载）
            cached_content = _load_from_cache(zip_url)
            if cached_content is not None:
                return cached_content
            
            # 5. 下载文件（带重试机制）
            content = None
            last_error = None
            cache_key = _get_cache_key(zip_url)
            cache_path = os.path.join(CACHE_DIR, cache_key)
            
            for attempt in range(MAX_RETRIES):
                try:
                    r = self.session.get(zip_url, timeout=60, allow_redirects=True)
                    r.raise_for_status()
                    content = r.content
                    
                    # 类型判断（有时返回 HTML 维护页）
                    ctype = r.headers.get("Content-Type", "").lower()
                    if "zip" not in ctype and not zipfile.is_zipfile(io.BytesIO(content)):
                        raise RuntimeError(f"Unexpected content while downloading Orange Book ZIP (url={zip_url}, content-type={ctype})")
                    
                    # 下载成功，保存到缓存
                    _save_to_cache(zip_url, content)
                    return content
                    
                except Exception as e:
                    last_error = e
                    if attempt < MAX_RETRIES - 1:
                        # 等待后重试
                        wait_time = RETRY_DELAY * (attempt + 1)  # 指数退避
                        time.sleep(wait_time)
            
            # 所有重试都失败了，提供详细的错误信息
            error_msg = f"Failed to download Orange Book ZIP after {MAX_RETRIES} attempts.\n"
            error_msg += f"  URL: {zip_url}\n"
            error_msg += f"  Last error: {str(last_error)}\n"
            error_msg += f"  Cache directory: {CACHE_DIR}\n"
            error_msg += f"  Expected cache file: {cache_path}\n"
            if not os.path.exists(CACHE_DIR):
                error_msg += f"  ❌ Cache directory does not exist!\n"
                error_msg += f"  💡 Tip: Set ORANGE_BOOK_CACHE_DIR environment variable to point to the cache directory.\n"
                error_msg += f"  💡 Tip: Copy the cache directory from the machine where prefetch_cache.py was run.\n"
            elif not os.path.exists(cache_path):
                error_msg += f"  ❌ Cache file does not exist!\n"
                error_msg += f"  💡 Tip: Run prefetch_cache.py on a machine with network access to download the ZIP file.\n"
                error_msg += f"  💡 Tip: Copy the cache directory to this machine.\n"
            raise RuntimeError(error_msg)
        
        finally:
            # 释放锁
            if lock_file:
                _release_lock(lock_file)

    @staticmethod
    def _read_tilde_file(zf: zipfile.ZipFile, member_name: str) -> List[List[str]]:
        with zf.open(member_name) as f:
            text = io.TextIOWrapper(f, encoding="utf-8", errors="ignore")
            reader = csv.reader(text, delimiter='~')
            return [row for row in reader]

    @staticmethod
    def _read_tilde_path(path: str) -> Tuple[Tuple[str, ...], ...]:
        stat = os.stat(path)
        return _read_tilde_path_cached(path, stat.st_mtime_ns, stat.st_size)

    @staticmethod
    def _find_table_path(data_dir: str, stem: str) -> Optional[str]:
        candidates = [
            os.path.join(data_dir, f"{stem}.txt"),
            os.path.join(data_dir, f"{stem}.txt.txt"),
        ]
        return next((path for path in candidates if os.path.isfile(path)), None)

    def _load_tables(self) -> Tuple[List[List[str]], List[List[str]], List[List[str]], Dict[str, Any]]:
        """Load a frozen FDA snapshot when available, otherwise the official ZIP.

        The repository snapshot contains the raw FDA tables, not benchmark
        questions, labels, or rationales.  Keeping this fallback makes formal
        runs reproducible and permits offline fixture tests while preserving the
        FDA page as the authoritative provenance URL.
        """

        configured_dir = self.data_dir or os.getenv("ORANGE_BOOK_DATA_DIR", "").strip()
        candidate_dir = configured_dir or _DEFAULT_ORANGE_BOOK_DATA_DIR
        if candidate_dir and os.path.isdir(candidate_dir):
            products_path = self._find_table_path(candidate_dir, "products")
            patents_path = self._find_table_path(candidate_dir, "patent")
            exclusivity_path = self._find_table_path(candidate_dir, "exclusivity")
            if products_path and patents_path and exclusivity_path:
                return (
                    self._read_tilde_path(products_path),
                    self._read_tilde_path(patents_path),
                    self._read_tilde_path(exclusivity_path),
                    {
                        "kind": "frozen_fda_orange_book_snapshot",
                        "directory": os.path.abspath(candidate_dir),
                        "files": [
                            os.path.basename(products_path),
                            os.path.basename(patents_path),
                            os.path.basename(exclusivity_path),
                        ],
                        "sha256": {
                            os.path.basename(products_path): _file_sha256(products_path),
                            os.path.basename(patents_path): _file_sha256(patents_path),
                            os.path.basename(exclusivity_path): _file_sha256(exclusivity_path),
                        },
                    },
                )
            if configured_dir:
                raise RuntimeError(
                    "ORANGE_BOOK_DATA_DIR is missing products, patent, or exclusivity tilde-delimited tables."
                )

        raw = self._download_zip()
        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            names = {n.lower(): n for n in zf.namelist()}
            prod_name = next((v for k, v in names.items() if "products.txt" in k), None)
            pat_name = next((v for k, v in names.items() if "patent.txt" in k or "patents.txt" in k), None)
            ex_name = next((v for k, v in names.items() if "exclusivity.txt" in k), None)
            if not (prod_name and pat_name and ex_name):
                raise RuntimeError("Orange Book zip missing expected files (Products/Patent/Exclusivity).")
            return (
                self._read_tilde_file(zf, prod_name),
                self._read_tilde_file(zf, pat_name),
                self._read_tilde_file(zf, ex_name),
                {
                    "kind": "fda_orange_book_zip",
                    "files": [prod_name, pat_name, ex_name],
                },
            )

    def resolve_by_ingredients(self, ingredients: List[str]) -> Dict[str, Any]:
        """根据药物成分获取并保留 application/product 关联的 FDA 记录。"""
        normalized_targets = [str(item).strip() for item in ingredients if str(item).strip()]
        if not normalized_targets:
            return {
                "patents": [],
                "exclusivities": [],
                "approvals": [],
                "matched_products": [],
                "sources": [ORANGE_BOOK_PAGE],
                "source_snapshot": {},
            }
        products, patents, exclusiv, source_snapshot = self._load_tables()

        product_index, pat_map, ex_map = _get_orange_book_index(products, patents, exclusiv)
        target_tokens = [
            tokens
            for tokens in (_normalized_drug_tokens(item) for item in normalized_targets)
            if tokens
        ]
        key_products: List[Dict[str, Any]] = []
        seen_products: set[Tuple[str, str, str]] = set()
        for indexed_tokens, indexed_products in product_index.items():
            if not any(_drug_token_identity_match(target, indexed_tokens) for target in target_tokens):
                continue
            for product in indexed_products:
                identity = (
                    str(product.get("application_number") or ""),
                    str(product.get("product_number") or ""),
                    str(product.get("active_ingredient") or ""),
                )
                if identity in seen_products:
                    continue
                seen_products.add(identity)
                key_products.append(dict(product))

        approvals: List[Approval] = []
        for product in key_products:
            if product["application_type"] != "N":
                continue
            approvals.append(Approval(
                region="US",
                product_name=product["product_name"],
                active_ingredient=product["active_ingredient"],
                approval_date=product["approval_date"],
                status="Approved",
                marketing_authorisation_company=product["marketing_authorisation_company"] or None,
                application_number=product["application_number"] or None,
                product_number=product["product_number"] or None,
                application_type=product["application_type"] or None,
                dosage_form_route=product["dosage_form_route"] or None,
                strength=product["strength"] or None,
                applicant_abbreviation=product["applicant_abbreviation"] or None,
            ))
        approvals.sort(
            key=lambda item: (
                item.approval_date or "",
                item.application_number or "",
                item.product_number or "",
            ),
            reverse=True,
        )
        
        out_patents: List[Patent] = []
        out_excl: List[Exclusivity] = []
        for product in key_products:
            if product["application_type"] != "N":
                continue
            key = (product["application_number"], product["product_number"])
            for patent in pat_map.get(key, []):
                notes = f"Patent use code {patent['use_code']}" if patent.get("use_code") else None
                out_patents.append(
                    Patent(
                        jurisdiction="US",
                        number=patent["number"],
                        expiry_date=patent["expiry_date"] or None,
                        notes=notes,
                        application_number=product["application_number"] or None,
                        product_number=product["product_number"] or None,
                        product_name=product["product_name"] or None,
                        active_ingredient=product["active_ingredient"] or None,
                        approval_date=product["approval_date"],
                        marketing_authorisation_company=product["marketing_authorisation_company"] or None,
                        use_code=patent["use_code"] or None,
                        drug_substance_flag=patent["drug_substance_flag"] or None,
                        drug_product_flag=patent["drug_product_flag"] or None,
                        delist_flag=patent["delist_flag"] or None,
                        submission_date=patent["submission_date"] or None,
                    )
                )
            for exclusivity in ex_map.get(key, []):
                out_excl.append(
                    Exclusivity(
                        region="US",
                        type=exclusivity["type"],
                        end_date=exclusivity["end_date"] or None,
                        application_number=product["application_number"] or None,
                        product_number=product["product_number"] or None,
                        product_name=product["product_name"] or None,
                        active_ingredient=product["active_ingredient"] or None,
                        approval_date=product["approval_date"],
                        marketing_authorisation_company=product["marketing_authorisation_company"] or None,
                    )
                )

        def uniq(seq):
            seen = set()
            res = []
            for x in seq:
                key = tuple(asdict(x).items())
                if key not in seen:
                    seen.add(key)
                    res.append(x)
            return res

        approvals = uniq(approvals)
        out_patents = uniq(out_patents)
        out_excl = uniq(out_excl)
        out_patents.sort(key=lambda item: (item.approval_date or "", item.expiry_date or ""), reverse=True)
        out_excl.sort(key=lambda item: (item.approval_date or "", item.end_date or ""), reverse=True)
        key_products.sort(
            key=lambda item: (
                item.get("approval_date") or "",
                item.get("application_number") or "",
                item.get("product_number") or "",
            ),
            reverse=True,
        )

        return {
            "patents": out_patents,
            "exclusivities": out_excl,
            "approvals": approvals,
            "matched_products": key_products,
            "sources": [ORANGE_BOOK_PAGE],
            "source_snapshot": source_snapshot,
        }

# ---------- 聚合器 ----------
OPEN_STATUSES = {"Recruiting", "Enrolling by invitation", "Not yet recruiting", "Available"}

class TrialAggregator:
    """聚合临床试验和药物信息"""
    
    def __init__(self, trial_provider: Optional[TrialProvider] = None,
                 drug_provider: Optional[DrugIntelProvider] = None):
        self.trial_provider = trial_provider or CtGovProvider()
        self.drug_provider = drug_provider or OrangeBookProvider()

    def get_profile(self, nct_id: str) -> TrialCommercialProfile:
        """获取完整的临床试验商业档案"""
        if not self.trial_provider.can_handle(nct_id):
            raise ValueError("Unsupported NCT ID format")
        basic = self.trial_provider.fetch_basic_trial(nct_id)
        intel = self.drug_provider.resolve_by_ingredients(basic.get("ingredients", []))
        is_open = (basic.get("recruitment_status") or "") in OPEN_STATUSES
        return TrialCommercialProfile(
            nct_id=nct_id,
            ingredients=basic.get("ingredients") or [],
            sponsor=basic.get("sponsor") or "",
            is_open_for_enrollment=is_open,
            recruitment_status=basic.get("recruitment_status") or "",
            patents=intel.get("patents") or [],
            exclusivities=intel.get("exclusivities") or [],
            approvals=intel.get("approvals") or [],
            sources=[*basic.get("sources", []), *intel.get("sources", [])]
        )


@lru_cache(maxsize=2)
def _load_qwen_tokenizer(model_name: str):
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(model_name, local_files_only=True, trust_remote_code=False)
    except Exception:
        return None


# ---------- 新增：openFDA 药品标签 Provider ----------
class OpenFDAProvider:
    """
    使用 openFDA 药品标签接口查询单个药品信息。
    - 依次尝试 brand_name、generic_name、原始全文检索
    - 对动态字段做 LLM 选择
    - 在工具层做上下文压缩与切分
    - 支持基于历史工具结果的多轮补充检索
    """

    def __init__(self):
        self.session = _session_with_retry()
        self._last_llm_raw: Optional[str] = None
        self.runtime_config = self._load_runtime_config()
        self.chunk_token_limit = int(self.runtime_config.get("compression_chunk_tokens", 20000))
        self.retrieval_max_rounds = int(self.runtime_config.get("retrieval_max_rounds", 5))
        self.qwen_tokenizer_name = str(
            self.runtime_config.get("qwen_tokenizer_name", "Qwen/Qwen2.5-7B-Instruct")
        ).strip() or "Qwen/Qwen2.5-7B-Instruct"

    def _load_runtime_config(self) -> Dict[str, Any]:
        path = os.path.join(_PROJECT_ROOT, "config", "agent.yaml")
        if not os.path.exists(path):
            return {}
        try:
            import yaml

            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except Exception:
            return {}

    def _query_one(self, term: str) -> Optional[Dict[str, Any]]:
        params_list = [
            {
                "base_url": OPENFDA_LABEL_URL,
                "limit": 1,
                "search": f'(openfda.brand_name:("{term}"))',
            },
            {
                "base_url": OPENFDA_LABEL_URL,
                "limit": 1,
                "search": f'(openfda.generic_name:("{term}"))',
            },
            {
                "base_url": OPENFDA_LABEL_URL,
                "limit": 1,
                "search": f'"{term}"',
            },
        ]
        for params in params_list:
            base_url = params["base_url"]
            query_params = {k: v for k, v in params.items() if k != "base_url"}
            try:
                query = urlencode(query_params, quote_via=quote)
                url = f"{base_url}?{query}"
                resp = self.session.get(url, timeout=10)
                resp.raise_for_status()
                result = resp.json()
                if isinstance(result, dict) and result.get("results"):
                    return result
            except Exception:
                pass
        return None

    def _collect_dynamic_label_keys(self, record: Dict[str, Any]) -> List[str]:
        keys: List[str] = []
        for key in record.keys():
            if key != "openfda":
                keys.append(key)
        openfda = record.get("openfda")
        if isinstance(openfda, dict):
            for sub_key in openfda.keys():
                keys.append(f"openfda.{sub_key}")
        uniq: List[str] = []
        for key in keys:
            if key not in uniq:
                uniq.append(key)
        return uniq

    def _fallback_label_fields(self, dynamic_keys: List[str], minimum_keys: int = 1) -> List[str]:
        priority = [
            "boxed_warning",
            "warnings",
            "warnings_and_cautions",
            "contraindications",
            "indications_and_usage",
            "dosage_and_administration",
            "adverse_reactions",
            "drug_interactions",
            "use_in_specific_populations",
            "pregnancy",
            "lactation",
            "pregnancy_or_breast_feeding",
            "nursing_mothers",
            "information_for_patients",
            "patient_information",
            "spl_medguide",
            "do_not_use",
            "ask_doctor",
            "ask_doctor_or_pharmacist",
            "stop_use",
            "purpose",
            "description",
            "openfda.brand_name",
            "openfda.generic_name",
            "openfda.substance_name",
        ]
        selected: List[str] = []
        for key in priority:
            if key in dynamic_keys and key not in selected:
                selected.append(key)
            if len(selected) >= max(1, minimum_keys):
                return selected

        for key in dynamic_keys:
            if key.startswith("openfda."):
                continue
            if key not in selected:
                selected.append(key)
            if len(selected) >= max(1, minimum_keys):
                return selected

        return dynamic_keys[: max(1, minimum_keys)]

    def _select_relevant_label_fields(self, context: str, dynamic_keys: List[str]) -> List[str]:
        lowered = (context or "").lower()
        heuristic_groups = [
            (
                ("dose", "dosage", "administer", "administration", "how often", "frequency"),
                ["dosage_and_administration", "indications_and_usage"],
            ),
            (
                ("contraind", "avoid", "not use", "unsafe", "allergy"),
                ["contraindications", "warnings_and_cautions", "warnings"],
            ),
            (
                ("pregnan", "breast", "lactat", "nursing", "trimester"),
                ["pregnancy", "lactation", "use_in_specific_populations"],
            ),
            (
                ("interaction", "interact", "coadmin", "together with"),
                ["drug_interactions", "warnings_and_cautions"],
            ),
            (
                ("adverse", "side effect", "toxicity", "harm"),
                ["adverse_reactions", "warnings", "warnings_and_cautions"],
            ),
            (
                ("indication", "used for", "treat", "therapy", "condition"),
                ["indications_and_usage", "clinical_pharmacology"],
            ),
            (
                ("patient", "counsel", "tell the patient"),
                ["information_for_patients", "patient_information", "spl_medguide"],
            ),
        ]

        selected: List[str] = []
        for triggers, preferred_keys in heuristic_groups:
            if not any(trigger in lowered for trigger in triggers):
                continue
            for key in preferred_keys:
                if key in dynamic_keys and key not in selected:
                    selected.append(key)
            if len(selected) >= 2:
                break

        if selected:
            return selected[:2]

        return self._fallback_label_fields(dynamic_keys, minimum_keys=1)

    def _normalize_label_value(self, val: Any) -> Any:
        if val is None:
            return ""
        if isinstance(val, str) and val.strip() == "":
            return ""
        if isinstance(val, (list, dict)) and len(val) == 0:
            return ""
        return val

    def _extract_canonical_names(self, drug_name: str, record: Dict[str, Any]) -> List[str]:
        candidates: List[str] = []

        def _add(value: Any) -> None:
            if isinstance(value, list):
                for item in value:
                    _add(item)
                return
            text = str(value or "").strip()
            if not text:
                return
            parts = [part.strip() for part in re.split(r";|,|\band\b", text, flags=re.IGNORECASE) if part.strip()]
            for part in parts or [text]:
                if part and part.lower() not in {existing.lower() for existing in candidates}:
                    candidates.append(part)

        openfda = record.get("openfda") or {}
        if isinstance(openfda, dict):
            _add(openfda.get("generic_name"))
            _add(openfda.get("substance_name"))
            _add(openfda.get("brand_name"))
        _add(drug_name)
        return candidates[:8]

    def _context_to_text(self, context: Any) -> str:
        if isinstance(context, str):
            return context
        if not isinstance(context, list):
            return str(context or "")

        parts: List[str] = []
        for msg in context:
            role = str(msg.get("role", "")).strip().lower()
            content = str(msg.get("content", "")).strip()
            if not content:
                continue
            if role == "tool":
                continue
            parts.append(f"[{role}] {content}")
        return "\n\n".join(parts[-6:])

    def _parse_tool_payload_from_message(self, content: str) -> Optional[Dict[str, Any]]:
        if "TOOL_RESULT:" not in content:
            return None
        brace_idx = content.find("{")
        if brace_idx == -1:
            return None
        try:
            payload = json.loads(content[brace_idx:])
        except Exception:
            return None
        return payload if isinstance(payload, dict) else None

    def _extract_previous_tool_state(self, drug_name: str, context: Any) -> Dict[str, Any]:
        state = {"used_keys": [], "round": 0}
        if not isinstance(context, list):
            return state

        normalized_drug_name = drug_name.strip().lower()
        for msg in context:
            payload = msg.get("_tool_payload")
            if not isinstance(payload, dict):
                content = str(msg.get("content", ""))
                payload = self._parse_tool_payload_from_message(content)
            if not payload:
                continue
            if str(payload.get("drug_name", "")).strip().lower() != normalized_drug_name:
                continue
            data = payload.get("data")
            if not isinstance(data, dict):
                continue
            try:
                state["round"] = max(state["round"], int(data.get("retrieval_round", 0)))
            except Exception:
                pass
            prior_keys = data.get("used_keys") or data.get("selected_keys") or []
            if isinstance(prior_keys, list):
                for key in prior_keys:
                    key = str(key).strip()
                    if key and key not in state["used_keys"]:
                        state["used_keys"].append(key)
        return state

    def _render_field_text(self, key: str, value: Any) -> str:
        if isinstance(value, list):
            cleaned = [str(item).strip() for item in value if str(item).strip()]
            body = "\n".join(f"- {item}" for item in cleaned)
        elif isinstance(value, dict):
            body = json.dumps(value, ensure_ascii=False, indent=2)
        else:
            body = str(value).strip()

        if not body:
            return ""
        return f"{key}:\n{body}"

    def _count_tokens(self, text: str) -> int:
        tokenizer = _load_qwen_tokenizer(self.qwen_tokenizer_name)
        if tokenizer is not None:
            try:
                return len(tokenizer.encode(text, add_special_tokens=False))
            except Exception:
                pass
        return max(1, len(text) // 4)

    def _split_text_into_chunks(self, text: str) -> List[str]:
        limit = max(1, self.chunk_token_limit)
        tokenizer = _load_qwen_tokenizer(self.qwen_tokenizer_name)

        if tokenizer is not None:
            try:
                token_ids = tokenizer.encode(text, add_special_tokens=False)
                if len(token_ids) <= limit:
                    return [text.strip()]
                num_chunks = max(1, math.ceil(len(token_ids) / limit))
                per_chunk = max(1, math.ceil(len(token_ids) / num_chunks))
                chunks: List[str] = []
                for i in range(0, len(token_ids), per_chunk):
                    piece = tokenizer.decode(token_ids[i:i + per_chunk], skip_special_tokens=True).strip()
                    if piece:
                        chunks.append(piece)
                return chunks or [text.strip()]
            except Exception:
                pass

        approx_tokens = max(1, len(text) // 4)
        if approx_tokens <= limit:
            return [text.strip()]
        num_chunks = max(1, math.ceil(approx_tokens / limit))
        per_chunk_chars = max(1, math.ceil(len(text) / num_chunks))
        chunks = []
        for i in range(0, len(text), per_chunk_chars):
            piece = text[i:i + per_chunk_chars].strip()
            if piece:
                chunks.append(piece)
        return chunks or [text.strip()]

    def _build_chunk_map(self, drug_name: str, filtered: Dict[str, Any]) -> Dict[str, Any]:
        chunks: Dict[str, Any] = {}
        chunk_idx = 1

        for key, value in filtered.items():
            rendered = self._render_field_text(key, value)
            if not rendered:
                continue
            pieces = self._split_text_into_chunks(rendered)
            if len(pieces) == 1:
                content = pieces[0]
                chunks[f"Chunk{chunk_idx}"] = {
                    "fields": [key],
                    "token_count": self._count_tokens(content),
                    "content": content,
                }
                chunk_idx += 1
                continue

            total = len(pieces)
            for piece_idx, piece in enumerate(pieces, start=1):
                chunks[f"Chunk{chunk_idx}"] = {
                    "fields": [f"{key}[part {piece_idx}/{total}]"],
                    "token_count": self._count_tokens(piece),
                    "content": piece,
                }
                chunk_idx += 1

        return chunks

    def _render_chunk_map(self, drug_name: str, chunks: Dict[str, Any]) -> str:
        lines = [f"Drug: {drug_name}"]
        for chunk_name, chunk_info in chunks.items():
            if not isinstance(chunk_info, dict):
                continue
            fields = chunk_info.get("fields") or []
            token_count = chunk_info.get("token_count", "")
            content = str(chunk_info.get("content", "")).strip()
            lines.append(f"{chunk_name} | fields={fields} | tokens={token_count}")
            lines.append(content)
        return "\n\n".join(lines)

    def fetch_drug_info_filtered(self, drug_name: str, context: Any) -> Optional[Dict[str, Any]]:
        base = self.fetch_drug_info(drug_name)
        if not isinstance(base, dict):
            return None
        records = base.get("results") or []
        if not records:
            return None
        record = records[0]

        dynamic_keys = self._collect_dynamic_label_keys(record)
        question_context = self._context_to_text(context)
        previous_state = self._extract_previous_tool_state(drug_name, context)
        used_keys = list(previous_state.get("used_keys", []))
        retrieval_round = int(previous_state.get("round", 0)) + 1

        available_keys = [key for key in dynamic_keys if key not in used_keys]
        if retrieval_round > self.retrieval_max_rounds or not available_keys:
            return {
                "question_context": question_context,
                "retrieval_round": retrieval_round,
                "selected_keys": [],
                "used_keys": used_keys,
                "remaining_key_count": 0,
                "remaining_keys_preview": [],
                "has_more_context": False,
                "max_retrieval_rounds": self.retrieval_max_rounds,
                "chunk_token_limit": self.chunk_token_limit,
                "chunks": {},
                "rendered_context": f"Drug: {drug_name}\nNo additional openFDA keys are available for retrieval.",
            }

        selected_keys = self._select_relevant_label_fields(question_context, available_keys)
        if not selected_keys:
            selected_keys = self._fallback_label_fields(available_keys, minimum_keys=1)

        self._append_key_selection_log(drug_name, question_context, dynamic_keys, selected_keys)

        filtered: Dict[str, Any] = {}
        for key in selected_keys:
            value = None
            if key.startswith("openfda."):
                sub_key = key.split(".", 1)[1]
                openfda = record.get("openfda") or {}
                if isinstance(openfda, dict):
                    value = openfda.get(sub_key)
            else:
                value = record.get(key)
            filtered[key] = self._normalize_label_value(value)

        chunks = self._build_chunk_map(drug_name, filtered)
        all_used_keys: List[str] = []
        for key in used_keys + selected_keys:
            if key not in all_used_keys:
                all_used_keys.append(key)
        remaining_keys = [key for key in dynamic_keys if key not in all_used_keys]

        return {
            "question_context": question_context,
            "retrieval_round": retrieval_round,
            "selected_keys": selected_keys,
            "used_keys": all_used_keys,
            "remaining_key_count": len(remaining_keys),
            "remaining_keys_preview": remaining_keys[:20],
            "has_more_context": bool(remaining_keys) and retrieval_round < self.retrieval_max_rounds,
            "max_retrieval_rounds": self.retrieval_max_rounds,
            "chunk_token_limit": self.chunk_token_limit,
            "chunks": chunks,
            "rendered_context": self._render_chunk_map(drug_name, chunks),
            "canonical_names": self._extract_canonical_names(drug_name, record),
        }

    def fetch_drug_info(self, drug_name: str) -> Optional[Dict[str, Any]]:
        if not isinstance(drug_name, str) or not drug_name.strip():
            return None
        name = drug_name.strip()

        result = self._query_one(name)
        if result:
            return result

        clean = re.sub(r"[^\w\s]", "", name)
        words = [w for w in clean.split() if w]
        if len(words) > 1:
            prefix = " ".join(words[:-1])
            prefix_res = self._query_one(prefix)
            if prefix_res:
                return prefix_res

        return None

    def _append_key_selection_log(self, drug_name: str, context: str, original_keys: List[str], selected_keys: List[str]) -> None:
        try:
            os.makedirs(KEY_LOG_DIR, exist_ok=True)
            lock = _acquire_lock(KEY_LOG_FILE + ".lock", timeout=15)
            try:
                entries: List[Dict[str, Any]] = []
                if os.path.exists(KEY_LOG_FILE):
                    with open(KEY_LOG_FILE, "r", encoding="utf-8") as rf:
                        try:
                            entries = json.load(rf) or []
                        except Exception:
                            entries = []
                entry = {
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "drug_name": drug_name,
                    "original_keys": original_keys,
                    "selected_keys": selected_keys,
                    "context": context,
                    "raw": (self._last_llm_raw or ""),
                }
                with open(KEY_LOG_FILE, "w", encoding="utf-8") as wf:
                    json.dump(entries + [entry], wf, ensure_ascii=False, indent=2)
            finally:
                _release_lock(lock)
        except Exception:
            pass
