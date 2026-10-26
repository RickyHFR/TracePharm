from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
from typing import Any

import requests
import yaml

from agent.telemetry import record_tool_substep
from tool_support.benchmark_mirror_filter import (
    VisibleExampleReference,
    add_audit_decision,
    build_visible_example_reference,
    inspect_candidate,
    merge_filter_audits,
    new_filter_audit,
)
from .base_tool import FunctionTool


def detect_language(text: str) -> str:
    chinese_chars = sum(1 for char in text if "\u4e00" <= char <= "\u9fff")
    total_chars = len(text)
    if total_chars == 0:
        return "en"
    return "zh" if (chinese_chars / total_chars) > 0.3 else "en"


def _runtime_config() -> dict[str, Any]:
    path = Path(__file__).resolve().parents[1] / "config" / "agent.yaml"
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return data if isinstance(data, dict) else {}


def _serper_api_key() -> str:
    config = _runtime_config()
    direct_value = str(config.get("serper_api_key", "")).strip()
    if direct_value:
        return direct_value
    env_name = str(config.get("serper_api_key_env", "SERPER_API_KEY") or "").strip()
    return str(os.environ.get(env_name, "") or "").strip()


@lru_cache(maxsize=100)
def _serper_search(query: str) -> dict[str, Any]:
    headers = {
        "X-API-KEY": _serper_api_key(),
        "Content-Type": "application/json",
    }
    payload = {"q": query, "num": 10}
    response = requests.post(
        "https://google.serper.dev/search",
        headers=headers,
        data=json.dumps(payload),
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def _format_results(
    query: str,
    payload: dict[str, Any],
    max_results: int,
    *,
    reference: VisibleExampleReference | None = None,
) -> dict[str, Any]:
    organic = payload.get("organic", []) or []
    results: list[dict[str, Any]] = []
    rendered_blocks: list[str] = []
    audit = new_filter_audit(reference)
    limit = max(1, int(max_results))
    for index, page in enumerate(organic, start=1):
        if len(results) >= limit:
            break
        item = {
            "rank": index,
            "title": str(page.get("title", "") or "").strip(),
            "link": str(page.get("link", "") or "").strip(),
            "snippet": str(page.get("snippet", "") or "").strip(),
            "source": str(page.get("source", "") or "").strip(),
            "date": str(page.get("date", "") or "").strip(),
        }
        inspected_content = "\n".join(
            value for value in (item["title"], item["snippet"], item["source"], item["link"]) if value
        )
        decision = inspect_candidate(
            inspected_content,
            reference,
            url=item["link"],
            title=item["title"],
            candidate_type="web_search_result",
        )
        add_audit_decision(audit, decision)
        if decision["blocked"]:
            continue
        results.append(item)
        rendered_blocks.append(
            f"{item['rank']}. [{item['title']}]({item['link']})"
            + (f"\nDate: {item['date']}" if item["date"] else "")
            + (f"\nSource: {item['source']}" if item["source"] else "")
            + (f"\n{item['snippet']}" if item["snippet"] else "")
        )
    return {
        "query": query,
        "language": detect_language(query),
        "results": results,
        "benchmark_mirror_filter": audit,
        "rendered_context": (
            f"A Google search for '{query}' returned {len(results)} result(s):\n\n" + "\n\n".join(rendered_blocks)
            if results
            else f"No search results found for '{query}'."
        ),
    }


def web_search(query: str | list[str], max_results: int = 10, context: Any = None) -> dict[str, Any]:
    try:
        reference = build_visible_example_reference(context)
        if not _serper_api_key():
            return {"success": False, "error": "SERPER_API_KEY is not configured.", "query": query}
        if isinstance(query, str):
            request_started = time.perf_counter()
            formatted = _format_results(
                query,
                _serper_search(query),
                max_results=max_results,
                reference=reference,
            )
            record_tool_substep(
                "web_search",
                "serper_request",
                time.perf_counter() - request_started,
                {
                    "query_count": 1,
                    "result_count": len(formatted["results"]),
                    "benchmark_mirror_filtered_count": int(
                        formatted["benchmark_mirror_filter"].get("filtered_count", 0) or 0
                    ),
                },
            )
            return {"success": bool(formatted["results"]), "data": formatted}

        queries = [str(item).strip() for item in query if str(item).strip()][:3]
        batch_started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=max(1, min(len(queries), 8))) as executor:
            payloads = list(executor.map(_serper_search, queries))
        batches = [
            _format_results(item_query, payload, max_results=max_results, reference=reference)
            for item_query, payload in zip(queries, payloads, strict=False)
        ]
        record_tool_substep(
            "web_search",
            "serper_batch_request",
            time.perf_counter() - batch_started,
            {
                "query_count": len(queries),
                "nonempty_batch_count": sum(1 for item in batches if item["results"]),
                "benchmark_mirror_filtered_count": sum(
                    int(item["benchmark_mirror_filter"].get("filtered_count", 0) or 0)
                    for item in batches
                ),
            },
        )
        rendered = "\n=======\n".join(item["rendered_context"] for item in batches)
        aggregate_audit = merge_filter_audits(
            reference,
            [item.get("benchmark_mirror_filter", {}) for item in batches],
        )
        return {
            "success": any(item["results"] for item in batches),
            "data": {
                "queries": queries,
                "batches": batches,
                "benchmark_mirror_filter": aggregate_audit,
                "rendered_context": rendered,
            },
        }
    except Exception as exc:
        return {"success": False, "error": f"web_search failed: {exc}", "query": query}


class WebSearchTool(FunctionTool):
    name = "web_search"
    description = "Run a Serper-backed web search and return ranked search results with snippets."
    input_hint = 'requires {"query":"<string>" or ["<string>", "..."],"max_results":10}'
    tool_function = staticmethod(web_search)
    passes_context = True

    def unavailable_reason(self) -> str:
        return "" if _serper_api_key() else "missing_serper_api_key"

    def build_kwargs(self, params: dict[str, Any]) -> dict[str, Any]:
        query = params.get("query")
        if isinstance(query, list):
            cleaned = [str(item).strip() for item in query if str(item).strip()]
            if not cleaned:
                raise ValueError("Missing required parameter: query")
            query_value: str | list[str] = cleaned
        else:
            query_value = self.require_str(params, "query")
        return {
            "query": query_value,
            "max_results": self.bounded_int(params, "max_results", default=10, minimum=1, maximum=10),
        }
