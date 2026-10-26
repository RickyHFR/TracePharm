from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .deepseek_client import DeepSeekClient
from .openai_responses_client import OpenAIResponsesClient


def _resolve_backend(config: dict[str, Any]) -> str:
    explicit = str(config.get("llm_backend", "") or config.get("backend_type", "")).strip().lower()
    if explicit:
        return explicit
    model_name = str(config.get("model", "") or "").strip().lower()
    if model_name.startswith("gpt-5") or model_name.startswith("o1") or model_name.startswith("o3") or model_name.startswith("o4"):
        return "openai_responses"
    return "deepseek"


def build_llm_client(config_path: str | None = None) -> Any:
    resolved_path = str(Path(config_path).resolve()) if config_path else None
    config: dict[str, Any] = {}
    if resolved_path:
        with open(resolved_path, "r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle) or {}
        if not isinstance(config, dict):
            raise ValueError(f"Expected mapping at LLM config path: {resolved_path}")
    backend = _resolve_backend(config)
    if backend == "openai_responses":
        return OpenAIResponsesClient(config_path=resolved_path)
    return DeepSeekClient(config_path=resolved_path)
