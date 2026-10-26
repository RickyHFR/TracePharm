from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from typing import Any, Dict, List

from agent.tool_format import bound_model_visible_text, render_tool_result_message


class Memory:
    def __init__(
        self,
        system_prompt: str,
        *,
        max_model_visible_tool_result_chars: int = 12000,
    ):
        self.messages: List[Dict[str, Any]] = [{"role": "system", "content": system_prompt}]
        self._tool_payloads_by_index: dict[int, dict[str, Any]] = {}
        self._tool_result_audit_history: list[dict[str, Any]] = []
        self.set_max_model_visible_tool_result_chars(max_model_visible_tool_result_chars)

    def set_max_model_visible_tool_result_chars(self, value: int) -> None:
        limit = int(value)
        if limit < 160:
            raise ValueError("max_model_visible_tool_result_chars must be at least 160")
        self.max_model_visible_tool_result_chars = limit

    def set_system_prompt(self, text: str) -> None:
        if self.messages and self.messages[0].get("role") == "system":
            self.messages[0] = {"role": "system", "content": text}
            return
        self.messages.insert(0, {"role": "system", "content": text})

    def add_user(self, text: str) -> None:
        self.messages.append({"role": "user", "content": text})

    def add_assistant(self, text: str) -> None:
        self.messages.append({"role": "assistant", "content": text})

    def add_tool_result(self, tool_name: str, result_dict: Dict[str, Any]) -> None:
        raw_payload = deepcopy(dict(result_dict))
        canonical_payload = json.dumps(
            raw_payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        full_line = render_tool_result_message(tool_name, raw_payload)
        line, visibility_audit = bound_model_visible_text(
            full_line,
            max_chars=self.max_model_visible_tool_result_chars,
        )
        audit = {
            **visibility_audit,
            "tool_name": str(tool_name),
            "raw_payload_sha256": hashlib.sha256(canonical_payload.encode("utf-8")).hexdigest(),
            "raw_payload_chars": len(canonical_payload),
            "raw_payload_utf8_bytes": len(canonical_payload.encode("utf-8")),
        }
        self.messages.append(
            {
                "role": "tool",
                "content": line,
                "_tool_result_audit": deepcopy(audit),
            }
        )
        self._tool_payloads_by_index[len(self.messages) - 1] = {
            "tool_name": tool_name,
            "payload": raw_payload,
            "audit": deepcopy(audit),
        }
        self._tool_result_audit_history.append(
            {
                **deepcopy(audit),
                # The full payload is audit-only.  LLM clients normalize each
                # message down to role/content and therefore never see it.
                "raw_payload": deepcopy(raw_payload),
            }
        )

    def get_tool_result_audit_records(self) -> List[Dict[str, Any]]:
        return deepcopy(self._tool_result_audit_history)

    def get_messages(self) -> List[Dict[str, Any]]:
        return self.messages

    def get_messages_context(self, include_tool_payload: bool = False) -> List[Dict[str, Any]]:
        if not include_tool_payload:
            return self.messages[1:]

        context: list[dict[str, Any]] = []
        for index, message in enumerate(self.messages[1:], start=1):
            copied = dict(message)
            metadata = self._tool_payloads_by_index.get(index)
            if metadata is not None:
                copied["_tool_name"] = str(metadata.get("tool_name", "")).strip()
                payload = metadata.get("payload")
                if isinstance(payload, dict):
                    copied["_tool_payload"] = dict(payload)
            context.append(copied)
        return context

    def compact(self, retain_tool_pairs_k: int = 0) -> None:
        if retain_tool_pairs_k is None or int(retain_tool_pairs_k) < 0:
            return
        msgs = self.messages
        if not msgs:
            return

        keep_indices = set()

        if msgs[0].get("role") == "system":
            keep_indices.add(0)

        last_user_idx = None
        for i in range(len(msgs) - 1, -1, -1):
            if msgs[i].get("role") == "user":
                last_user_idx = i
                break
        if last_user_idx is not None:
            keep_indices.add(last_user_idx)

        tool_related_indices: List[int] = []
        for i, msg in enumerate(msgs):
            content = str(msg.get("content", ""))
            role = msg.get("role")
            if role == "tool" and content.startswith("TOOL_RESULT:"):
                tool_related_indices.append(i)
            elif role == "assistant" and "TOOL_CALL:" in content:
                tool_related_indices.append(i)

        if retain_tool_pairs_k == 0:
            tool_related_indices = []
        elif retain_tool_pairs_k > 0:
            tool_related_indices = tool_related_indices[-(retain_tool_pairs_k * 2):]

        keep_indices.update(tool_related_indices)

        kept_messages = [msgs[i] for i in sorted(keep_indices)]
        remapped_payloads: dict[int, dict[str, Any]] = {}
        for new_index, old_index in enumerate(sorted(keep_indices)):
            payload = self._tool_payloads_by_index.get(old_index)
            if payload is not None:
                remapped_payloads[new_index] = dict(payload)

        self.messages = kept_messages
        self._tool_payloads_by_index = remapped_payloads
