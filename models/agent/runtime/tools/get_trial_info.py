from __future__ import annotations

import time
from typing import Any

from agent.telemetry import record_tool_substep
from .base_tool import FunctionTool
from tool_support.trial_providers import CtGovProvider


def get_trial_info(nct_id: str, max_reported_outcomes: int = 20) -> dict[str, Any]:
    try:
        provider = CtGovProvider()
        if not provider.can_handle(nct_id):
            return {
                "success": False,
                "error": f"Unsupported NCT identifier format: {nct_id}",
                "nct_id": nct_id,
            }

        fetch_started = time.perf_counter()
        basic = provider.fetch_basic_trial(nct_id)
        record_tool_substep(
            "get_trial_info",
            "fetch_basic_trial",
            time.perf_counter() - fetch_started,
            {"nct_id": nct_id},
        )
        sources = [str(item).strip() for item in (basic.get("sources", []) or []) if str(item).strip()]
        source_url = sources[0] if sources else f"https://clinicaltrials.gov/study/{nct_id}"
        reported_outcomes = list(basic.get("reported_outcomes") or [])

        def outcome_priority(item: Any) -> int:
            if not isinstance(item, dict):
                return 2
            return 0 if str(item.get("type") or "").upper() == "PRIMARY" else 1

        reported_outcomes.sort(key=outcome_priority)
        max_outcomes = max(1, min(int(max_reported_outcomes), 50))
        basic["reported_outcomes"] = reported_outcomes[:max_outcomes]
        basic["reported_outcome_count"] = len(reported_outcomes)
        basic["reported_outcomes_truncated"] = len(reported_outcomes) > max_outcomes
        api_url = str(basic.get("source_api_url") or f"https://clinicaltrials.gov/api/v2/studies/{nct_id}")
        return {
            "success": True,
            "nct_id": nct_id,
            **basic,
            "sources": sources,
            "source_records": [
                {
                    "url": source_url,
                    "title": f"ClinicalTrials.gov study {nct_id}",
                    "source_type": "trial_registry",
                },
                {
                    "url": api_url,
                    "title": f"ClinicalTrials.gov API v2 record {nct_id}",
                    "source_type": "trial_registry",
                },
            ],
        }
    except Exception as exc:
        return {
            "success": False,
            "error": f"Failed to fetch trial information: {exc}",
            "nct_id": nct_id,
        }


class GetTrialInfoTool(FunctionTool):
    name = "get_trial_info"
    description = "Get source-linked ClinicalTrials.gov v2 protocol, regimen arms, interventions, dates, registered outcomes, and reported arm-level results for one NCT identifier."
    input_hint = 'requires {"nct_id":"NCT########"}; optional "max_reported_outcomes":1-50 (primary outcomes are retained first)'
    tool_function = staticmethod(get_trial_info)

    def build_kwargs(self, params: dict[str, Any]) -> dict[str, Any]:
        return {
            "nct_id": self.require_str(params, "nct_id"),
            "max_reported_outcomes": self.bounded_int(
                params,
                "max_reported_outcomes",
                default=20,
                minimum=1,
                maximum=50,
            ),
        }
