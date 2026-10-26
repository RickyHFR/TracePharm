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
            "product_name",
            "active_ingredient",
            "approval_date",
            "status",
            "marketing_authorisation_company",
            "application_number",
            "product_number",
            "application_type",
            "dosage_form_route",
            "strength",
            "applicant_abbreviation",
        )
    }


def _validated_cutoff(value: str) -> str:
    value = str(value or "").strip()
    if not value:
        return ""
    date.fromisoformat(value)
    return value


def get_drug_approvals(
    ingredients: list[str],
    as_of_date: str = "",
    max_records: int = 20,
) -> dict[str, Any]:
    try:
        cutoff = _validated_cutoff(as_of_date)
        provider = OrangeBookProvider()
        resolve_started = time.perf_counter()
        intel = provider.resolve_by_ingredients(ingredients)
        record_tool_substep(
            "get_drug_approvals",
            "resolve_by_ingredients",
            time.perf_counter() - resolve_started,
            {"ingredient_count": len(ingredients)},
        )
        approvals = [_record_dict(item) for item in (intel.get("approvals", []) or [])]
        if cutoff:
            approvals = [
                item
                for item in approvals
                if item.get("approval_date") and str(item.get("approval_date")) <= cutoff
            ]
        approvals.sort(
            key=lambda item: (
                str(item.get("approval_date") or ""),
                str(item.get("application_number") or ""),
                str(item.get("product_number") or ""),
            ),
            reverse=True,
        )
        sources = [str(item).strip() for item in (intel.get("sources", []) or []) if str(item).strip()]
        source_url = sources[0] if sources else "https://www.fda.gov/drugs/drug-approvals-and-databases/orange-book-data-files"

        approvals_dict = []
        for item in approvals[: max(1, min(int(max_records), 50))]:
            approvals_dict.append(
                {
                    **item,
                    "source_url": source_url,
                    "source_type": "fda_orange_book",
                }
            )
        latest_date = str(approvals[0].get("approval_date") or "") if approvals else ""
        latest_candidates = [item for item in approvals_dict if str(item.get("approval_date") or "") == latest_date]

        return {
            "success": True,
            "ingredients": ingredients,
            "as_of_date": cutoff or None,
            "approvals": approvals_dict,
            "count": len(approvals),
            "returned_count": len(approvals_dict),
            "latest_approval_date": latest_date or None,
            "latest_approval_candidates": latest_candidates,
            "selection_policy": "Matched NDA product approvals sorted by original product approval date descending; no lower-year cutoff and no single-row truncation.",
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
            "error": f"Failed to fetch approval information: {exc}",
            "ingredients": ingredients,
        }


class GetDrugApprovalsTool(FunctionTool):
    name = "get_drug_approvals"
    description = "Get product-linked FDA Orange Book NDA approvals, sorted latest first, for one or more ingredients."
    input_hint = 'requires {"ingredients":["<string>", "..."]}; for historical questions also pass "as_of_date":"YYYY-MM-DD"'
    tool_function = staticmethod(get_drug_approvals)

    def build_kwargs(self, params: dict[str, Any]) -> dict[str, Any]:
        ingredients = self.clean_string_list(params.get("ingredients"))
        if not ingredients:
            raise ValueError("Missing required parameter: ingredients")
        return {
            "ingredients": ingredients,
            "as_of_date": self.optional_str(params, "as_of_date"),
            "max_records": self.bounded_int(params, "max_records", default=20, minimum=1, maximum=50),
        }
