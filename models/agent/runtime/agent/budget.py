from __future__ import annotations

from dataclasses import dataclass, field
import time


ONLINE_EVIDENCE_TOOLS = {
    "web_search",
    "web_visit",
    "get_all_info_by_drug",
    "get_all_info_by_disease",
    "get_drug_patents",
    "get_drug_exlusivities",
    "get_drug_approvals",
    "get_trial_info",
}
LOCAL_EVIDENCE_TOOLS = {"search_local"}


def _limit(value: object, default: int) -> int:
    try:
        parsed = int(value)
    except Exception:
        parsed = int(default)
    return parsed


@dataclass(slots=True)
class ExecutionBudget:
    max_total_tool_calls: int = 6
    max_online_tool_calls: int = 4
    max_local_tool_calls: int = 2
    max_tool_turns: int = 3
    max_llm_calls: int = 12
    max_total_tokens: int = 120_000
    deadline_seconds: float = 300.0
    started_at: float = field(default_factory=time.perf_counter)
    total_tool_calls: int = 0
    online_tool_calls: int = 0
    local_tool_calls: int = 0
    tool_turns: int = 0
    llm_calls: int = 0
    stop_reason: str = ""

    @classmethod
    def from_config(cls, config: dict[str, object]) -> "ExecutionBudget":
        return cls(
            max_total_tool_calls=_limit(config.get("max_total_tool_calls"), 6),
            max_online_tool_calls=_limit(config.get("max_online_tool_calls"), 4),
            max_local_tool_calls=_limit(config.get("max_local_tool_calls"), 2),
            max_tool_turns=_limit(config.get("max_tool_turns"), 3),
            max_llm_calls=_limit(config.get("max_llm_calls"), 12),
            max_total_tokens=_limit(config.get("max_total_tokens"), 120_000),
            deadline_seconds=float(config.get("deadline_seconds", 300.0) or 300.0),
        )

    @staticmethod
    def _remaining(limit: int, used: int) -> int:
        return -1 if limit < 0 else max(0, limit - used)

    @property
    def remaining_tool_turns(self) -> int:
        return self._remaining(self.max_tool_turns, self.tool_turns)

    @property
    def elapsed_seconds(self) -> float:
        return max(0.0, time.perf_counter() - self.started_at)

    def wall_time_ok(self) -> bool:
        if self.deadline_seconds < 0:
            return True
        if self.elapsed_seconds < self.deadline_seconds:
            return True
        self.stop_reason = self.stop_reason or "deadline_seconds"
        return False

    def can_call_llm(self, current_total_tokens: int = 0) -> tuple[bool, str]:
        if not self.wall_time_ok():
            return False, self.stop_reason
        if self.max_llm_calls >= 0 and self.llm_calls >= self.max_llm_calls:
            self.stop_reason = self.stop_reason or "max_llm_calls"
            return False, self.stop_reason
        if self.max_total_tokens >= 0 and int(current_total_tokens) >= self.max_total_tokens:
            self.stop_reason = self.stop_reason or "max_total_tokens"
            return False, self.stop_reason
        return True, ""

    def record_llm_call(self) -> None:
        self.llm_calls += 1

    def can_start_tool_turn(self) -> tuple[bool, str]:
        if not self.wall_time_ok():
            return False, self.stop_reason
        if self.max_tool_turns >= 0 and self.tool_turns >= self.max_tool_turns:
            self.stop_reason = self.stop_reason or "max_tool_turns"
            return False, self.stop_reason
        return True, ""

    def record_tool_turn(self) -> None:
        self.tool_turns += 1

    def can_call_tool(self, tool_name: str) -> tuple[bool, str]:
        if not self.wall_time_ok():
            return False, self.stop_reason
        if self.max_total_tool_calls >= 0 and self.total_tool_calls >= self.max_total_tool_calls:
            self.stop_reason = self.stop_reason or "max_total_tool_calls"
            return False, self.stop_reason
        if tool_name in ONLINE_EVIDENCE_TOOLS:
            if self.max_online_tool_calls >= 0 and self.online_tool_calls >= self.max_online_tool_calls:
                return False, "max_online_tool_calls"
        if tool_name in LOCAL_EVIDENCE_TOOLS:
            if self.max_local_tool_calls >= 0 and self.local_tool_calls >= self.max_local_tool_calls:
                return False, "max_local_tool_calls"
        return True, ""

    def record_tool_call(self, tool_name: str) -> None:
        self.total_tool_calls += 1
        if tool_name in ONLINE_EVIDENCE_TOOLS:
            self.online_tool_calls += 1
        if tool_name in LOCAL_EVIDENCE_TOOLS:
            self.local_tool_calls += 1

    def export(self) -> dict[str, object]:
        return {
            "limits": {
                "max_total_tool_calls": self.max_total_tool_calls,
                "max_online_tool_calls": self.max_online_tool_calls,
                "max_local_tool_calls": self.max_local_tool_calls,
                "max_tool_turns": self.max_tool_turns,
                "max_llm_calls": self.max_llm_calls,
                "max_total_tokens": self.max_total_tokens,
                "deadline_seconds": self.deadline_seconds,
            },
            "used": {
                "total_tool_calls": self.total_tool_calls,
                "online_tool_calls": self.online_tool_calls,
                "local_tool_calls": self.local_tool_calls,
                "tool_turns": self.tool_turns,
                "llm_calls": self.llm_calls,
                "elapsed_seconds": round(self.elapsed_seconds, 6),
            },
            "stop_reason": self.stop_reason,
        }
