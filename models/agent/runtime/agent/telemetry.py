from __future__ import annotations

import time
from contextlib import contextmanager
from threading import Lock, local
from typing import Any, Iterator


_TLS = local()


def _round_seconds(value: float) -> float:
    return round(float(value), 6)


def get_active_timing_recorder() -> "TimingRecorder | None":
    recorder = getattr(_TLS, "timing_recorder", None)
    return recorder if isinstance(recorder, TimingRecorder) else None


@contextmanager
def bind_timing_recorder(recorder: "TimingRecorder") -> Iterator["TimingRecorder"]:
    previous = getattr(_TLS, "timing_recorder", None)
    _TLS.timing_recorder = recorder
    try:
        yield recorder
    finally:
        _TLS.timing_recorder = previous


class TimingRecorder:
    def __init__(self) -> None:
        self._lock = Lock()
        self._turn_started_at: float | None = None
        self._turn_finished_at: float | None = None
        self.usage: dict[str, int] = {
            "input_tokens": 0,
            "input_cached_tokens": 0,
            "output_tokens": 0,
            "output_reasoning_tokens": 0,
            "total_tokens": 0,
        }
        self.llm_calls: list[dict[str, Any]] = []
        self.tool_calls: list[dict[str, Any]] = []
        self.tool_substeps: list[dict[str, Any]] = []
        self.agent_loops: list[dict[str, Any]] = []
        self.system_events: list[dict[str, Any]] = []

    def start_turn(self) -> None:
        self._turn_started_at = time.perf_counter()
        self._turn_finished_at = None

    def finish_turn(self) -> None:
        self._turn_finished_at = time.perf_counter()

    def record_llm_call(
        self,
        *,
        kind: str,
        model_name: str,
        elapsed_seconds: float,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        with self._lock:
            self.llm_calls.append(
                {
                    "call_index": len(self.llm_calls) + 1,
                    "kind": str(kind),
                    "model_name": str(model_name),
                    "elapsed_seconds": _round_seconds(elapsed_seconds),
                    "metadata": dict(metadata or {}),
                }
            )

    def record_usage(self, usage: dict[str, Any] | None) -> None:
        if not isinstance(usage, dict):
            return
        with self._lock:
            for key in self.usage:
                try:
                    self.usage[key] += int(usage.get(key, 0) or 0)
                except Exception:
                    continue

    def record_tool_call(
        self,
        *,
        tool_name: str,
        elapsed_seconds: float,
        success: bool | None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        with self._lock:
            self.tool_calls.append(
                {
                    "call_index": len(self.tool_calls) + 1,
                    "tool_name": str(tool_name),
                    "success": success,
                    "elapsed_seconds": _round_seconds(elapsed_seconds),
                    "metadata": dict(metadata or {}),
                }
            )

    def record_tool_substep(
        self,
        *,
        tool_name: str,
        step_name: str,
        elapsed_seconds: float,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        with self._lock:
            self.tool_substeps.append(
                {
                    "substep_index": len(self.tool_substeps) + 1,
                    "tool_name": str(tool_name),
                    "step_name": str(step_name),
                    "elapsed_seconds": _round_seconds(elapsed_seconds),
                    "metadata": dict(metadata or {}),
                }
            )

    def record_agent_loop(
        self,
        *,
        loop_index: int,
        elapsed_seconds: float,
        used_tools: bool,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        with self._lock:
            self.agent_loops.append(
                {
                    "loop_index": int(loop_index),
                    "used_tools": bool(used_tools),
                    "elapsed_seconds": _round_seconds(elapsed_seconds),
                    "metadata": dict(metadata or {}),
                }
            )

    def record_system_event(
        self,
        *,
        name: str,
        elapsed_seconds: float,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        with self._lock:
            self.system_events.append(
                {
                    "event_index": len(self.system_events) + 1,
                    "name": str(name),
                    "elapsed_seconds": _round_seconds(elapsed_seconds),
                    "metadata": dict(metadata or {}),
                }
            )

    @contextmanager
    def time_system_event(self, name: str, metadata: dict[str, Any] | None = None) -> Iterator[None]:
        started = time.perf_counter()
        try:
            yield
        finally:
            self.record_system_event(
                name=name,
                elapsed_seconds=time.perf_counter() - started,
                metadata=metadata,
            )

    def _total_seconds(self) -> float:
        if self._turn_started_at is None:
            return 0.0
        finished = self._turn_finished_at if self._turn_finished_at is not None else time.perf_counter()
        return max(0.0, finished - self._turn_started_at)

    def build_summary(self) -> dict[str, float | int]:
        total_seconds = self._total_seconds()
        llm_seconds = sum(float(item.get("elapsed_seconds", 0.0) or 0.0) for item in self.llm_calls)
        tool_seconds = sum(float(item.get("elapsed_seconds", 0.0) or 0.0) for item in self.tool_calls)
        measured_system_seconds = sum(float(item.get("elapsed_seconds", 0.0) or 0.0) for item in self.system_events)
        residual_seconds = max(0.0, total_seconds - llm_seconds - tool_seconds - measured_system_seconds)
        system_processing_seconds = measured_system_seconds + residual_seconds
        waiting_seconds = 0.0
        other_seconds = 0.0
        return {
            "llm_backend_seconds": _round_seconds(llm_seconds),
            "tool_seconds": _round_seconds(tool_seconds),
            "system_processing_seconds": _round_seconds(system_processing_seconds),
            "waiting_seconds": _round_seconds(waiting_seconds),
            "other_seconds": _round_seconds(other_seconds),
            "total_seconds": _round_seconds(total_seconds),
            "llm_call_count": len(self.llm_calls),
            "tool_call_count": len(self.tool_calls),
            "tool_substep_count": len(self.tool_substeps),
            "agent_loop_count": len(self.agent_loops),
        }

    def export(self) -> dict[str, Any]:
        return {
            "summary": self.build_summary(),
            "usage": dict(self.usage),
            "llm_calls": list(self.llm_calls),
            "tool_calls": list(self.tool_calls),
            "tool_substeps": list(self.tool_substeps),
            "agent_loops": list(self.agent_loops),
            "system_events": list(self.system_events),
        }


def record_tool_substep(
    tool_name: str,
    step_name: str,
    elapsed_seconds: float,
    metadata: dict[str, Any] | None = None,
) -> None:
    recorder = get_active_timing_recorder()
    if recorder is None:
        return
    recorder.record_tool_substep(
        tool_name=tool_name,
        step_name=step_name,
        elapsed_seconds=elapsed_seconds,
        metadata=metadata,
    )
