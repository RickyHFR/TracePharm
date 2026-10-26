import os
import time
from threading import Lock, local
from typing import Any, Dict, List, Optional, Tuple

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


def _project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _default_config_path() -> str:
    return os.path.join(_project_root(), "config", "deepseek.yaml")


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


def _parse_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _usage_to_dict(response_usage: Any) -> Dict[str, int]:
    if response_usage is None:
        return _empty_usage()

    prompt_tokens = getattr(response_usage, "prompt_tokens", None)
    if prompt_tokens is None:
        prompt_tokens = getattr(response_usage, "input_tokens", 0)

    completion_tokens = getattr(response_usage, "completion_tokens", None)
    if completion_tokens is None:
        completion_tokens = getattr(response_usage, "output_tokens", 0)

    total_tokens = getattr(response_usage, "total_tokens", None)
    if total_tokens is None:
        total_tokens = _to_int(prompt_tokens) + _to_int(completion_tokens)

    prompt_details = getattr(response_usage, "prompt_tokens_details", None)
    if prompt_details is None:
        prompt_details = getattr(response_usage, "input_tokens_details", None)
    completion_details = getattr(response_usage, "completion_tokens_details", None)
    if completion_details is None:
        completion_details = getattr(response_usage, "output_tokens_details", None)

    cached_tokens = 0
    if prompt_details is not None:
        cached_tokens = getattr(prompt_details, "cached_tokens", 0)
        if not cached_tokens and isinstance(prompt_details, dict):
            cached_tokens = prompt_details.get("cached_tokens", 0)

    reasoning_tokens = 0
    if completion_details is not None:
        reasoning_tokens = getattr(completion_details, "reasoning_tokens", 0)
        if not reasoning_tokens and isinstance(completion_details, dict):
            reasoning_tokens = completion_details.get("reasoning_tokens", 0)

    return {
        "input_tokens": _to_int(prompt_tokens),
        "input_cached_tokens": _to_int(cached_tokens),
        "output_tokens": _to_int(completion_tokens),
        "output_reasoning_tokens": _to_int(reasoning_tokens),
        "total_tokens": _to_int(total_tokens),
    }


def _normalize_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """
    Ensure roles are acceptable to the OpenAI SDK.
    Map tool outputs to 'user' so the backend sees a valid assistant -> user -> assistant turn pattern.
    Map any other unknown roles to 'assistant' while keeping content intact.
    """
    norm: List[Dict[str, str]] = []
    for m in messages:
        role = str(m.get("role", "")).strip().lower()
        content = str(m.get("content", ""))
        if role == "tool":
            role = "user"
        elif role not in _ALLOWED_ROLES:
            role = "assistant"
        norm.append({"role": role, "content": content})
    return norm


def _coerce_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: List[str] = []
        for item in value:
            if isinstance(item, dict):
                text_value = item.get("text")
                if text_value:
                    parts.append(str(text_value))
                elif item.get("type") == "output_text" and item.get("text"):
                    parts.append(str(item.get("text")))
                else:
                    parts.append(str(item))
            else:
                parts.append(str(item))
        return "".join(parts)
    return str(value)


def _extract_message_text(message: Any) -> str:
    content = _coerce_text(getattr(message, "content", None))
    reasoning = _coerce_text(getattr(message, "reasoning_content", None))
    if content.strip() and reasoning.strip():
        return content
    if content.strip():
        return content
    if reasoning.strip():
        return reasoning
    return ""


def load_deepseek_config(config_path: Optional[str] = None) -> Tuple[str, str, str, float, str, bool]:
    path = config_path or _default_config_path()
    api_key = ""
    base_url = "https://api.deepseek.com"
    model = "deepseek-chat"
    timeout_seconds = 120.0
    reasoning_effort = ""
    thinking_enabled = False
    api_key_env = "DEEPSEEK_API_KEY"

    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        api_key = str(data.get("api_key", "")).strip()
        api_key_env = str(data.get("api_key_env", api_key_env)).strip() or api_key_env
        base_url = str(data.get("base_url", base_url)).strip() or base_url
        model = str(data.get("model", model)).strip() or model
        reasoning_effort = str(data.get("reasoning_effort", "") or "").strip().lower()
        thinking_enabled = _parse_bool(data.get("thinking_enabled"), default=False)
        try:
            timeout_seconds = float(data.get("request_timeout_seconds", timeout_seconds) or timeout_seconds)
        except Exception:
            timeout_seconds = 120.0

    env_key = os.environ.get(api_key_env)
    if env_key:
        api_key = env_key

    return api_key, base_url, model, timeout_seconds, reasoning_effort, thinking_enabled


def _load_retry_settings(config_path: str) -> tuple[int, float, float]:
    max_retries = 2
    initial_seconds = 1.0
    max_seconds = 8.0
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as handle:
                data = yaml.safe_load(handle) or {}
            max_retries = max(0, int(data.get("max_retries", max_retries) or 0))
            initial_seconds = max(0.0, float(data.get("retry_initial_seconds", initial_seconds) or 0.0))
            max_seconds = max(initial_seconds, float(data.get("retry_max_seconds", max_seconds) or max_seconds))
        except Exception:
            pass
    return max_retries, initial_seconds, max_seconds


def _load_sampling_temperature(config_path: str) -> float:
    """Load the explicitly frozen sampling temperature for agent calls."""

    temperature = 0.0
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        raw_value = data.get("temperature", temperature)
        try:
            temperature = float(raw_value)
        except Exception as exc:
            raise ValueError(f"Invalid DeepSeek temperature in {config_path}: {raw_value!r}") from exc
    if not 0.0 <= temperature <= 2.0:
        raise ValueError(f"DeepSeek temperature must be between 0 and 2: {temperature}")
    return temperature


def _is_retryable_exception(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    try:
        if int(status_code) == 429 or int(status_code) >= 500:
            return True
    except Exception:
        pass
    message = str(exc).lower()
    return any(
        token in message
        for token in (
            "timeout",
            "timed out",
            "connection reset",
            "connection aborted",
            "temporarily unavailable",
            "rate limit",
            "http 429",
            "http 500",
            "http 502",
            "http 503",
            "http 504",
        )
    )


class DeepSeekClient:
    _usage_lock = Lock()
    _global_usage: Dict[str, int] = _empty_usage()
    _thread_local = local()

    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or _default_config_path()
        config = {}
        if os.path.exists(self.config_path):
            with open(self.config_path, "r", encoding="utf-8") as handle:
                config = yaml.safe_load(handle) or {}
        self.send_thinking_parameter = _parse_bool(config.get("send_thinking_parameter"), default=True)
        (
            self.api_key,
            self.base_url,
            self.model,
            self.timeout_seconds,
            self.reasoning_effort,
            self.thinking_enabled,
        ) = load_deepseek_config(self.config_path)

        if not self.api_key:
            key_env = config.get("api_key_env", "DEEPSEEK_API_KEY")
            raise RuntimeError(f"Missing API key. Set {key_env} or provide api_key in the model configuration.")

        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url, max_retries=0)
        self.max_retries, self.retry_initial_seconds, self.retry_max_seconds = _load_retry_settings(self.config_path)
        self.temperature = _load_sampling_temperature(self.config_path)
        self.last_usage: Dict[str, int] = _empty_usage()
        self.total_usage: Dict[str, int] = _empty_usage()

    def _load_config(self) -> Tuple[str, str, str, float, str, bool]:
        return load_deepseek_config(self.config_path)

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

    def get_usage_snapshot(self, reset: bool = False) -> Dict[str, int]:
        snapshot = dict(self.total_usage)
        if reset:
            self.total_usage = _empty_usage()
            self.last_usage = _empty_usage()
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

    def _request_once(self, request_kwargs: Dict[str, Any]) -> Any:
        return self.client.chat.completions.create(**request_kwargs)

    def _create_completion(self, request_kwargs: Dict[str, Any]) -> Any:
        attempts = self.max_retries + 1
        for attempt in range(attempts):
            try:
                return self._request_once(request_kwargs)
            except Exception as exc:
                if attempt >= attempts - 1 or not _is_retryable_exception(exc):
                    raise
                delay = min(self.retry_max_seconds, self.retry_initial_seconds * (2 ** attempt))
                if delay > 0:
                    time.sleep(delay)
        raise RuntimeError("unreachable completion retry state")

    def chat(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: Optional[int] = None,
        model_name: Optional[str] = None,
    ) -> str:
        """
        Send Chat Completions messages and return assistant text.
        Expects a list of dicts with 'role' and 'content'.
        """
        norm_messages = _normalize_messages(messages)

        request_kwargs = {
            "model": model_name or self.model,
            "messages": norm_messages,
            "stream": False,
            "timeout": self.timeout_seconds,
            "temperature": self.temperature,
        }
        if self.reasoning_effort and (not self.send_thinking_parameter or self.thinking_enabled):
            request_kwargs["reasoning_effort"] = self.reasoning_effort
        if self.send_thinking_parameter:
            request_kwargs["extra_body"] = {
                "thinking": {"type": "enabled" if self.thinking_enabled else "disabled"}
            }
        if max_tokens is not None:
            request_kwargs["max_tokens"] = int(max_tokens)

        started = time.perf_counter()
        resp = self._create_completion(request_kwargs)
        elapsed_seconds = time.perf_counter() - started
        self._record_usage(getattr(resp, "usage", None))
        recorder = get_active_timing_recorder()
        if recorder is not None:
            recorder.record_llm_call(
                kind="chat",
                model_name=str(request_kwargs["model"]),
                elapsed_seconds=elapsed_seconds,
                metadata={
                    "message_count": len(norm_messages),
                    "max_tokens": max_tokens,
                    "reasoning_effort": self.reasoning_effort or None,
                    "thinking_enabled": self.thinking_enabled,
                    "temperature": self.temperature,
                },
            )
        return _extract_message_text(resp.choices[0].message)

    def chat_json(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: Optional[int] = None,
        model_name: Optional[str] = None,
        thinking_enabled: Optional[bool] = None,
    ) -> str:
        """
        Send Chat Completions messages and return assistant text.
        Expects a list of dicts with 'role' and 'content'.
        """
        norm_messages = _normalize_messages(messages)

        request_kwargs = {
            "model": model_name or self.model,
            "messages": norm_messages,
            "stream": False,
            "timeout": self.timeout_seconds,
            "temperature": self.temperature,
            "response_format": {
                'type': 'json_object'
            },
        }
        effective_thinking = (
            self.thinking_enabled if thinking_enabled is None else bool(thinking_enabled)
        )
        if self.reasoning_effort and (not self.send_thinking_parameter or effective_thinking):
            request_kwargs["reasoning_effort"] = self.reasoning_effort
        if self.send_thinking_parameter:
            request_kwargs["extra_body"] = {
                "thinking": {"type": "enabled" if effective_thinking else "disabled"}
            }
        if max_tokens is not None:
            request_kwargs["max_tokens"] = int(max_tokens)

        started = time.perf_counter()
        resp = self._create_completion(request_kwargs)
        elapsed_seconds = time.perf_counter() - started
        self._record_usage(getattr(resp, "usage", None))
        recorder = get_active_timing_recorder()
        if recorder is not None:
            recorder.record_llm_call(
                kind="chat_json",
                model_name=str(request_kwargs["model"]),
                elapsed_seconds=elapsed_seconds,
                metadata={
                    "message_count": len(norm_messages),
                    "max_tokens": max_tokens,
                    "reasoning_effort": self.reasoning_effort or None,
                    "thinking_enabled": effective_thinking,
                    "temperature": self.temperature,
                },
            )
        return _extract_message_text(resp.choices[0].message)

    def close(self) -> None:
        close_fn = getattr(self.client, "close", None)
        if callable(close_fn):
            close_fn()
