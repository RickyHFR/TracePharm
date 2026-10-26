from __future__ import annotations

import importlib
import inspect
import pkgutil
from pathlib import Path
from typing import Any, Dict, Iterable, List

from .base_tool import BaseTool


_SKIP_MODULES = {
    "__init__",
    "base_tool",
    "tool_registry",
    "trial_tools",
}
_TOOLS_CACHE: Dict[str, BaseTool] | None = None
_TOOL_ALIASES = {
    # The legacy HLE adapter used ``retrieve_*`` while the unified runtime
    # exposes the same evidence lane as ``get_*``.  Keep the public legacy
    # name usable without duplicating the implementation.
    "retrieve_hle_drug_external_evidence": "get_hle_drug_external_evidence",
}


def _normalize_tool_names(values: Iterable[str] | None) -> list[str]:
    if not values:
        return []
    names: list[str] = []
    for value in values:
        text = str(value).strip()
        if text and text not in names:
            names.append(text)
    return names


def _discover_tools() -> Dict[str, BaseTool]:
    global _TOOLS_CACHE
    if _TOOLS_CACHE is not None:
        return _TOOLS_CACHE

    discovered: Dict[str, BaseTool] = {}
    package_name = __name__.rsplit(".", 1)[0]
    package_dir = Path(__file__).resolve().parent

    for module_info in pkgutil.iter_modules([str(package_dir)]):
        if module_info.name in _SKIP_MODULES or module_info.ispkg:
            continue
        module = importlib.import_module(f"{package_name}.{module_info.name}")
        for _, obj in inspect.getmembers(module, inspect.isclass):
            if obj is BaseTool or not issubclass(obj, BaseTool):
                continue
            if obj.__module__ != module.__name__:
                continue
            tool = obj()
            if not tool.name:
                continue
            discovered[tool.name] = tool

    _TOOLS_CACHE = dict(sorted(discovered.items()))
    return _TOOLS_CACHE


def _select_tools(
    enabled_tools: Iterable[str] | None = None,
    disabled_tools: Iterable[str] | None = None,
    *,
    strict: bool = False,
    check_availability: bool = True,
) -> Dict[str, BaseTool]:
    available = _discover_tools()
    enabled = _normalize_tool_names(enabled_tools)
    disabled = set(_normalize_tool_names(disabled_tools))

    if enabled:
        missing = [
            name
            for name in enabled
            if name not in available and _TOOL_ALIASES.get(name) not in available
        ]
        if strict and missing:
            raise ValueError(
                "Unknown enabled tool name(s): "
                + ", ".join(missing)
                + ". Available tools: "
                + ", ".join(sorted(available))
            )
        selected: Dict[str, BaseTool] = {}
        for name in enabled:
            target_name = name if name in available else _TOOL_ALIASES.get(name, "")
            if target_name in available:
                selected[name] = available[target_name]
    else:
        selected = dict(sorted(available.items()))

    for name in list(selected):
        if name in disabled or (check_availability and selected[name].unavailable_reason()):
            selected.pop(name, None)
    return selected


def get_tool_availability(enabled_tools=None, disabled_tools=None) -> Dict[str, Dict[str, Any]]:
    configured = _select_tools(enabled_tools, disabled_tools, check_availability=False)
    return {
        name: {"available": name in configured and not tool.unavailable_reason(),
               "reason": tool.unavailable_reason() if name in configured else "not_enabled_by_config"}
        for name, tool in _discover_tools().items()
    }


def execute_tool(
    tool_name: str,
    params: Dict[str, Any],
    context: Any = None,
    enabled_tools: Iterable[str] | None = None,
    disabled_tools: Iterable[str] | None = None,
    *,
    strict: bool = False,
) -> Dict[str, Any]:
    selected = _select_tools(
        enabled_tools=enabled_tools,
        disabled_tools=disabled_tools,
        strict=strict,
    )
    if tool_name not in selected:
        info = get_tool_availability(enabled_tools, disabled_tools).get(tool_name, {})
        return {"success": False, "error": f"Unavailable tool: {tool_name}",
                "reason": info.get("reason", "unregistered_tool"), "tool": tool_name}
    return selected[tool_name].execute(params, context=context)


def get_allowed_tools(
    enabled_tools: Iterable[str] | None = None,
    disabled_tools: Iterable[str] | None = None,
    *,
    strict: bool = False,
) -> List[str]:
    return list(
        _select_tools(
            enabled_tools=enabled_tools,
            disabled_tools=disabled_tools,
            strict=strict,
        )
    )


def get_tool_specs(
    enabled_tools: Iterable[str] | None = None,
    disabled_tools: Iterable[str] | None = None,
    *,
    strict: bool = False,
) -> Dict[str, Dict[str, str]]:
    return {
        name: tool.spec()
        for name, tool in _select_tools(
            enabled_tools=enabled_tools,
            disabled_tools=disabled_tools,
            strict=strict,
        ).items()
    }
