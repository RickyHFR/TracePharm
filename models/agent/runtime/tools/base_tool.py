from __future__ import annotations

from abc import ABC, abstractmethod
import time
from typing import Any, Callable

from agent.telemetry import get_active_timing_recorder


ToolFunction = Callable[..., dict[str, Any]]


class BaseTool(ABC):
    name: str = ""
    description: str = ""
    input_hint: str = ""

    def unavailable_reason(self) -> str:
        """Check local prerequisites only; never probe remote services here."""
        return ""

    def spec(self) -> dict[str, str]:
        return {
            "description": self.description,
            "input_hint": self.input_hint,
        }

    def execute(self, params: dict[str, Any], context: Any = None) -> dict[str, Any]:
        started = time.perf_counter()
        success: bool | None = None
        try:
            if not self.name:
                result = self._error("Unnamed tool.")
                success = bool(result.get("success", False))
                return result
            if not isinstance(params, dict):
                result = self._error("Tool parameters must be a JSON object.")
                success = bool(result.get("success", False))
                return result
            result = self.run(params, context=context)
            success = bool(result.get("success", False)) if isinstance(result, dict) else None
            return result
        except ValueError as exc:
            result = self._error(str(exc))
            success = False
            return result
        except Exception as exc:
            result = self._error(f"Tool execution failed: {exc}")
            success = False
            return result
        finally:
            recorder = get_active_timing_recorder()
            if recorder is not None:
                recorder.record_tool_call(
                    tool_name=self.name or "<unnamed>",
                    elapsed_seconds=time.perf_counter() - started,
                    success=success,
                    metadata={"has_context": context is not None},
                )

    @abstractmethod
    def run(self, params: dict[str, Any], context: Any = None) -> dict[str, Any]:
        raise NotImplementedError

    def _error(self, message: str) -> dict[str, Any]:
        return {"success": False, "error": message, "tool": self.name}

    def require_str(self, params: dict[str, Any], key: str) -> str:
        value = str(params.get(key, "")).strip()
        if not value:
            raise ValueError(f"Missing required parameter: {key}")
        return value

    def optional_str(self, params: dict[str, Any], key: str) -> str:
        return str(params.get(key, "")).strip()

    def clean_string_list(self, value: Any) -> list[str]:
        if not isinstance(value, list):
            return []
        cleaned: list[str] = []
        for item in value:
            text = str(item).strip()
            if text:
                cleaned.append(text)
        return cleaned

    def bounded_int(self, params: dict[str, Any], key: str, default: int, minimum: int = 1, maximum: int = 10) -> int:
        try:
            value = int(params.get(key, default))
        except Exception:
            value = default
        return max(minimum, min(maximum, value))


class FunctionTool(BaseTool):
    tool_function: ToolFunction | None = None
    passes_context: bool = False

    def run(self, params: dict[str, Any], context: Any = None) -> dict[str, Any]:
        if self.tool_function is None:
            raise ValueError("No tool function configured.")
        kwargs = self.build_kwargs(params)
        if self.passes_context:
            kwargs["context"] = context
        return self.tool_function(**kwargs)

    @abstractmethod
    def build_kwargs(self, params: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError
