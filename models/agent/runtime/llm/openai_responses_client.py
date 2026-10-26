from __future__ import annotations

import os
from pathlib import Path
import time
from threading import Lock, local
from typing import Any, Dict, List, Optional

import yaml
from openai import OpenAI

from agent.telemetry import get_active_timing_recorder


_ALLOWED_ROLES = {"system", "user", "assistant"}
_USAGE_FIELDS = (
    "input_tokens",
    "input_cached_tokens",
    "output_tokens",
    "output_reasoning_tokens",
    "total_tokens",
)


def _default_config_path() -> str:
    return str((Path(__file__).resolve().parents[2] / "config" / "deepseek.yaml").resolve())


def _empty_usage() -> Dict[str, int]:
    return {field: 0 for field in _USAGE_FIELDS}


def _to_int(value: Any) -> int:
    try:
        return int(value or 0)
    except Exception:
        return 0


def _merge_usage(target: Dict[str, int], delta: Dict[str, Any]) -> Dict[str, int]:
    for field in _USAGE_FIELDS:
        target[field] = _to_int(target.get(field)) + _to_int(delta.get(field))
    return target


def _usage_to_dict(response_usage: Any) -> Dict[str, int]:
    if response_usage is None:
        return _empty_usage()

    input_tokens = getattr(response_usage, "input_tokens", 0)
    output_tokens = getattr(response_usage, "output_tokens", 0)
    total_tokens = getattr(response_usage, "total_tokens", None)
    if total_tokens is None:
        total_tokens = _to_int(input_tokens) + _to_int(output_tokens)

    input_details = getattr(response_usage, "input_tokens_details", None)
    output_details = getattr(response_usage, "output_tokens_details", None)
    cached_tokens = getattr(input_details, "cached_tokens", 0) if input_details is not None else 0
    reasoning_tokens = getattr(output_details, "reasoning_tokens", 0) if output_details is not None else 0

    return {
        "input_tokens": _to_int(input_tokens),
        "input_cached_tokens": _to_int(cached_tokens),
        "output_tokens": _to_int(output_tokens),
        "output_reasoning_tokens": _to_int(reasoning_tokens),
        "total_tokens": _to_int(total_tokens),
    }


def _normalize_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    normalized: List[Dict[str, str]] = []
    for message in messages:
        role = str(message.get("role", "")).strip().lower()
        content = str(message.get("content", ""))
        if role == "tool":
            role = "user"
        elif role not in _ALLOWED_ROLES:
            role = "assistant"
        normalized.append({"role": role, "content": content})
    return normalized


def _extract_output_text(response: Any) -> str:
    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str) and output_text:
        return output_text
    chunks: list[str] = []
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            text = getattr(content, "text", None)
            if text:
                chunks.append(str(text))
    return "".join(chunks)


class OpenAIResponsesClient:
    _usage_lock = Lock()
    _global_usage: Dict[str, int] = _empty_usage()
    _thread_local = local()

    def __init__(self, config_path: Optional[str] = None):
        self.config_path = str(Path(config_path or _default_config_path()).resolve())
        with open(self.config_path, "r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle) or {}
        if not isinstance(config, dict):
            raise ValueError(f"Expected mapping in OpenAI responses config: {self.config_path}")
        self.api_key = str(config.get("api_key", "")).strip()
        if not self.api_key:
            env_name = str(config.get("api_key_env", "OPENAI_API_KEY") or "OPENAI_API_KEY").strip()
            self.api_key = str(os.environ.get(env_name, "") or "").strip()
        self.model = str(config.get("model", "gpt-5.4")).strip() or "gpt-5.4"
        self.reasoning_effort = str(config.get("reasoning_effort", "high")).strip() or "high"
        self.text_verbosity = str(config.get("text_verbosity", "low")).strip() or "low"
        self.timeout_seconds = float(config.get("request_timeout_seconds", 180) or 180)
        if not self.api_key:
            raise RuntimeError(f"Missing OpenAI API key in {self.config_path}.")
        self.client = OpenAI(api_key=self.api_key, timeout=self.timeout_seconds)
        self.last_usage: Dict[str, int] = _empty_usage()
        self.total_usage: Dict[str, int] = _empty_usage()

    @classmethod
    def snapshot_global_usage(cls, reset: bool = False) -> Dict[str, int]:
        with cls._usage_lock:
            snapshot = dict(cls._global_usage)
            if reset:
                cls._global_usage = _empty_usage()
        return snapshot

    @classmethod
    def snapshot_thread_usage(cls, reset: bool = False) -> Dict[str, int]:
        snapshot = dict(getattr(cls._thread_local, "usage", _empty_usage()))
        if reset:
            cls._thread_local.usage = _empty_usage()
        return snapshot

    def _record_usage(self, response_usage: Any) -> None:
        usage_dict = _usage_to_dict(response_usage)
        self.last_usage = usage_dict
        _merge_usage(self.total_usage, usage_dict)
        with self._usage_lock:
            _merge_usage(self._global_usage, usage_dict)
        thread_usage = dict(getattr(self._thread_local, "usage", _empty_usage()))
        _merge_usage(thread_usage, usage_dict)
        self._thread_local.usage = thread_usage
        recorder = get_active_timing_recorder()
        if recorder is not None:
            recorder.record_usage(usage_dict)

    def _create_response(
        self,
        messages: List[Dict[str, Any]],
        *,
        model_name: Optional[str] = None,
        max_tokens: Optional[int] = None,
        json_mode: bool = False,
    ) -> Any:
        request_kwargs: Dict[str, Any] = {
            "model": model_name or self.model,
            "input": _normalize_messages(messages),
            "reasoning": {"effort": self.reasoning_effort},
            "text": {"verbosity": self.text_verbosity},
            "timeout": self.timeout_seconds,
        }
        if json_mode:
            request_kwargs["text"] = {
                "verbosity": self.text_verbosity,
                "format": {"type": "json_object"},
            }
        if max_tokens is not None:
            request_kwargs["max_output_tokens"] = int(max_tokens)

        started = time.perf_counter()
        response = self.client.responses.create(**request_kwargs)
        elapsed_seconds = time.perf_counter() - started
        self._record_usage(getattr(response, "usage", None))
        recorder = get_active_timing_recorder()
        if recorder is not None:
            recorder.record_llm_call(
                kind="responses_json" if json_mode else "responses_text",
                model_name=str(request_kwargs["model"]),
                elapsed_seconds=elapsed_seconds,
                metadata={
                    "message_count": len(request_kwargs["input"]),
                    "max_tokens": max_tokens,
                },
            )
        return response

    def chat(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: Optional[int] = None,
        model_name: Optional[str] = None,
    ) -> str:
        response = self._create_response(
            messages,
            model_name=model_name,
            max_tokens=max_tokens,
            json_mode=False,
        )
        return _extract_output_text(response)

    def chat_json(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: Optional[int] = None,
        model_name: Optional[str] = None,
    ) -> str:
        response = self._create_response(
            messages,
            model_name=model_name,
            max_tokens=max_tokens,
            json_mode=True,
        )
        return _extract_output_text(response)

    def close(self) -> None:
        close_fn = getattr(self.client, "close", None)
        if callable(close_fn):
            close_fn()
