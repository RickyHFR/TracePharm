from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import date
import time
from typing import Any

from agent.telemetry import record_tool_substep
from .base_tool import FunctionTool
from tool_support.trial_providers import OrangeBookProvider


def _record_dict(item: Any) -> dict[str, Any]:
    if is_dataclass(item):
        return asdict(item)
    if isinstance(item, dict):
        return dict(item)
    return {
        key: getattr(item, key, None)
        for key in (
            "region",
            "type",
            "start_date",
            "end_date",
            "notes",
            "application_number",
            "product_number",
            "product_name",
            "active_ingredient",
            "approval_date",
            "marketing_authorisation_company",
        )
    }


def _validated_cutoff(value: str) -> str:
    value = str(value or "").strip()
    if not value:
        return ""
    date.fromisoformat(value)
    return value


def get_drug_exlusivities(
    ingredients: list[str],
    as_of_date: str = "",
    latest_approval_only: bool = False,
    max_records: int = 30,
) -> dict[str, Any]:
    try:
        cutoff = _validated_cutoff(as_of_date)
        provider = OrangeBookProvider()
        resolve_started = time.perf_counter()
        intel = provider.resolve_by_ingredients(ingredients)
        record_tool_substep(
            "get_drug_exlusivities",
            "resolve_by_ingredients",
            time.perf_counter() - resolve_started,
            {"ingredient_count": len(ingredients)},
        )
        exclusivities = [_record_dict(item) for item in (intel.get("exclusivities", []) or [])]
        approvals = [_record_dict(item) for item in (intel.get("approvals", []) or [])]
        if cutoff:
            exclusivities = [
                item
                for item in exclusivities
                if item.get("approval_date") and str(item.get("approval_date")) <= cutoff
            ]
            approvals = [
                item
                for item in approvals
                if item.get("approval_date") and str(item.get("approval_date")) <= cutoff
            ]
        latest_approval_date = max(
            (str(item.get("approval_date") or "") for item in approvals),
            default="",
        )
        if latest_approval_only and latest_approval_date:
            exclusivities = [
                item
                for item in exclusivities
                if str(item.get("approval_date") or "") == latest_approval_date
            ]
        exclusivities.sort(
            key=lambda item: (
                str(item.get("approval_date") or ""),
                str(item.get("end_date") or ""),
                str(item.get("type") or ""),
            ),
            reverse=True,
        )
        sources = [str(item).strip() for item in (intel.get("sources", []) or []) if str(item).strip()]
        source_url = sources[0] if sources else "https://www.fda.gov/drugs/drug-approvals-and-databases/orange-book-data-files"

        exclusivities_dict = []
        for item in exclusivities[: max(1, min(int(max_records), 50))]:
            exclusivities_dict.append(
                {
                    **item,
                    "source_url": source_url,
                    "source_type": "fda_orange_book",
                }
            )

        return {
            "success": True,
            "ingredients": ingredients,
            "as_of_date": cutoff or None,
            "latest_approval_only": bool(latest_approval_only),
            "latest_approval_date": latest_approval_date or None,
            "exclusivities": exclusivities_dict,
            "count": len(exclusivities),
            "returned_count": len(exclusivities_dict),
            "selection_policy": "Product-linked NDA exclusivity records sorted by linked approval date then exclusivity end date.",
            "sources": sources,
            "source_snapshot": intel.get("source_snapshot") or {},
            "source_records": [
                {
                    "url": source_url,
                    "title": "FDA Orange Book Data Files",
                    "source_type": "fda_orange_book",
                }
            ],
        }
    except Exception as exc:
        return {
            "success": False,
            "error": f"Failed to fetch exclusivity information: {exc}",
            "ingredients": ingredients,
        }


class GetDrugExlusivitiesTool(FunctionTool):
    name = "get_drug_exlusivities"
    description = "Get product-linked FDA Orange Book exclusivity records for one or more ingredients."
    input_hint = 'requires {"ingredients":["<string>", "..."]}; historical/latest-product questions should pass "as_of_date":"YYYY-MM-DD" and may set "latest_approval_only":true'
    tool_function = staticmethod(get_drug_exlusivities)

    def build_kwargs(self, params: dict[str, Any]) -> dict[str, Any]:
        ingredients = self.clean_string_list(params.get("ingredients"))
        if not ingredients:
            raise ValueError("Missing required parameter: ingredients")
        latest_only_value = params.get("latest_approval_only", False)
        if isinstance(latest_only_value, str):
            latest_only = latest_only_value.strip().lower() in {"1", "true", "yes"}
        else:
            latest_only = bool(latest_only_value)
        return {
            "ingredients": ingredients,
            "as_of_date": self.optional_str(params, "as_of_date"),
            "latest_approval_only": latest_only,
            "max_records": self.bounded_int(params, "max_records", default=30, minimum=1, maximum=50),
        }
