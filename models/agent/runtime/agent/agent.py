import json
import os
from pathlib import Path
import re
import sys
import time
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional, Tuple

import yaml

from agent.prompt import (
    build_answer_only_prompt,
    build_main_system_prompt,
    build_main_system_prompt_task_form_routing,
    build_planner_prompt,
    build_reflection_system_prompt,
)
from agent.budget import ExecutionBudget, LOCAL_EVIDENCE_TOOLS, ONLINE_EVIDENCE_TOOLS
from agent.debate import DEBATE_SYSTEM_PROMPT, EvidenceDebate
from agent.evidence import normalize_tool_evidence
from agent.planning import (
    ENTITY_PLAN_SYSTEM_PROMPT,
    RELATION_SCREENING_SYSTEM_PROMPT,
    disabled_plan,
    extract_json_object,
    fallback_plan,
    normalize_plan,
    normalize_relation_screening,
    plan_entity_names,
    plan_goal_context,
    render_plan_for_model,
    screening_candidates,
    screening_failure_plan,
    visible_fallback_plan,
    visible_wording_constraints,
)
from agent.tool_format import (
    bound_model_visible_text,
    render_tool_call_block,
    render_tool_result_body,
)
from agent.memory import Memory
from agent.telemetry import get_active_timing_recorder
from llm.deepseek_client import DeepSeekClient
from tools.tool_registry import execute_tool, get_allowed_tools, get_tool_specs


TOOL_CALL_PREFIX = "TOOL_CALL:"
_TOOL_CALL_RE = re.compile(r"^\s*TOOL_CALL:\s+([A-Za-z_]\w*)\s+(.*)\s*$")
_REFLECTION_REQUIRED_KEYS = (
    "evidence_assessments",
    "top_two_candidates",
    "preferred_candidate",
    "runner_up",
    "why_preferred_over_runner_up",
    "confidence",
    "missing_fact",
    "retry",
    "needs_revision",
    "revised_answer",
)
_CRITICAL_EVIDENCE_RE = re.compile(
    r"\b("
    r"contraindicat(?:ed|ion)|unacceptable health risk|not to be used|category\s*4|"
    r"boxed warning|black box|avoid|do not use|must not|stroke risk|fatal|"
    r"first[- ]line|preferred|recommended|guideline|diagnostic criteria|medical eligibility|"
    r"fluid restriction|decrease further|worsen(?:s|ed|ing)?|symptomatic|"
    r"prolonged treatment|periodically|expected to need prolonged"
    r")\b",
    flags=re.IGNORECASE,
)
_FINAL_ANSWER_LINE_RE = re.compile(r"^\s*FINAL_ANSWER:\s*(.+?)\s*$", flags=re.IGNORECASE)


def _project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _agent_config_path() -> str:
    return os.path.join(_project_root(), "config", "agent.yaml")


def _repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.dirname(_project_root())))


def _load_agent_config() -> Dict[str, Any]:
    cfg = {
        "max_rounds": 20,
        "max_tools_per_turn": 3,
        "planner_max_tools_per_turn": 3,
        "enable_tools": True,
        "enable_planner": False,
        "enable_visible_planner_fallback": True,
        "enable_goal_guidance": True,
        "max_tool_turns": 3,
        "retain_tool_pairs_k": -1,
        "enable_reflection": False,
        "enable_rec_debate": True,
        "debate_max_rounds": 2,
        "debate_max_retrievals": 2,
        "debate_max_tokens": 1800,
        "debate_max_retries": 1,
        "structured_json_thinking_enabled": False,
        "reflection_max_passes": 1,
        "reflection_retry_confidence_threshold": 0.58,
        "reflection_retry_enabled": True,
        "reflection_generic_mcq_evidence_gap_local_retry": True,
        "reflection_generic_clinical_context_local_retry": True,
        "reflection_prefer_unused_local_retry": True,
        "reflection_missing_fact_local_retry": False,
        "reflection_weak_mcq_local_retry": False,
        "reflection_weak_mcq_local_retry_min_assessments": 1,
        "reflection_local_retry_missing_fact_confidence_threshold": 0.75,
        "reflection_followup_missing_fact_local_retry": False,
        "reflection_followup_missing_fact_confidence_threshold": 0.86,
        "reflection_direct_local_retry": False,
        "reflection_direct_retry_answer_only": True,
        "reflection_extra_parameter_local_retry": True,
        "reflection_retry_multiquery_enabled": True,
        "reflection_retry_max_queries": 3,
        "reflection_retry_dense_k": 16,
        "reflection_retry_bm25_k": 20,
        "reflection_retry_max_evidence": 12,
        "enable_main_evidence_brief": True,
        "main_evidence_brief_max_records": 8,
        "main_evidence_brief_max_chars": 6000,
        "prefetch_local_before_main": False,
        "prefetch_local_requires_reflection": False,
        "prefetch_local_for_mcq": True,
        "prefetch_local_for_clinical_context": True,
        "prefetch_local_skip_label_comparison": False,
        "prefetch_local_use_planner_search_tasks": True,
        "prefetch_local_goal_mode": "anchors",
        "prefetch_local_entity_mode": "visible_primary",
        "prefetch_local_include_choices": True,
        "prefetch_local_max_queries": 2,
        "prefetch_local_max_evidence": 12,
        "prefetch_local_dense_k": 12,
        "prefetch_local_bm25_k": 12,
        "unplanned_search_local_dense_k": 2,
        "unplanned_search_local_bm25_k": 2,
        "unplanned_search_local_max_evidence": 1,
        "max_total_tool_calls": 6,
        "max_online_tool_calls": 4,
        "max_local_tool_calls": 2,
        "max_llm_calls": 24,
        "max_total_tokens": 120000,
        "deadline_seconds": 300,
        "enforce_goal_propagation": True,
        "retrieval_max_rounds": 5,
        "compression_chunk_tokens": 20000,
        "max_model_visible_tool_result_chars": 12000,
        "planner_max_retries": 1,
        "planner_retry_max_tokens": 7000,
        "screening_max_tokens": 2400,
        "screening_max_retries": 1,
        "qwen_tokenizer_name": "Qwen/Qwen2.5-7B-Instruct",
        "use_primekg": True,
        "isolate_get_all_info_by_drug_process": sys.platform == "darwin",
        "get_all_info_by_drug_subprocess_timeout_seconds": 240,
        "enabled_tools": [],
        "disabled_tools": [],
        "strict_tool_registry": False,
        "main_prompt_variant": "default",
        "dump_prompts": False,
        "prompt_dump_dir": "output/prompt_dumps/agent_runtime",
    }

    def _parse_tool_list(value: Any) -> List[str]:
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        return []

    path = _agent_config_path()
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        cfg.update({
            "max_rounds": int(data.get("max_rounds", cfg["max_rounds"])),
            "max_tools_per_turn": int(data.get("max_tools_per_turn", cfg["max_tools_per_turn"])),
            "planner_max_tools_per_turn": int(data.get("planner_max_tools_per_turn", data.get("max_tools_per_turn", cfg["planner_max_tools_per_turn"]))),
            "enable_tools": bool(data.get("enable_tools", cfg["enable_tools"])),
            "enable_planner": bool(data.get("enable_planner", cfg["enable_planner"])),
            "enable_visible_planner_fallback": bool(
                data.get(
                    "enable_visible_planner_fallback",
                    cfg["enable_visible_planner_fallback"],
                )
            ),
            "enable_goal_guidance": bool(data.get("enable_goal_guidance", cfg["enable_goal_guidance"])),
            "max_tool_turns": int(data.get("max_tool_turns", data.get("max_tool_steps", cfg["max_tool_turns"]))),
            "retain_tool_pairs_k": int(data.get("retain_tool_pairs_k", cfg["retain_tool_pairs_k"])),
            "enable_reflection": bool(data.get("enable_reflection", cfg["enable_reflection"])),
            "enable_rec_debate": bool(data.get("enable_rec_debate", cfg["enable_rec_debate"])),
            "debate_max_rounds": int(data.get("debate_max_rounds", cfg["debate_max_rounds"])),
            "debate_max_retrievals": int(data.get("debate_max_retrievals", cfg["debate_max_retrievals"])),
            "debate_max_tokens": int(data.get("debate_max_tokens", cfg["debate_max_tokens"])),
            "debate_max_retries": int(data.get("debate_max_retries", cfg["debate_max_retries"])),
            "structured_json_thinking_enabled": bool(
                data.get(
                    "structured_json_thinking_enabled",
                    cfg["structured_json_thinking_enabled"],
                )
            ),
            "reflection_max_passes": int(data.get("reflection_max_passes", cfg["reflection_max_passes"])),
            "reflection_retry_confidence_threshold": float(data.get("reflection_retry_confidence_threshold", cfg["reflection_retry_confidence_threshold"])),
            "reflection_retry_enabled": bool(
                data.get("reflection_retry_enabled", cfg["reflection_retry_enabled"])
            ),
            "reflection_generic_mcq_evidence_gap_local_retry": bool(
                data.get(
                    "reflection_generic_mcq_evidence_gap_local_retry",
                    cfg["reflection_generic_mcq_evidence_gap_local_retry"],
                )
            ),
            "reflection_generic_clinical_context_local_retry": bool(
                data.get(
                    "reflection_generic_clinical_context_local_retry",
                    cfg["reflection_generic_clinical_context_local_retry"],
                )
            ),
            "reflection_prefer_unused_local_retry": bool(
                data.get("reflection_prefer_unused_local_retry", cfg["reflection_prefer_unused_local_retry"])
            ),
            "reflection_missing_fact_local_retry": bool(
                data.get("reflection_missing_fact_local_retry", cfg["reflection_missing_fact_local_retry"])
            ),
            "reflection_weak_mcq_local_retry": bool(
                data.get("reflection_weak_mcq_local_retry", cfg["reflection_weak_mcq_local_retry"])
            ),
            "reflection_weak_mcq_local_retry_min_assessments": int(
                data.get(
                    "reflection_weak_mcq_local_retry_min_assessments",
                    cfg["reflection_weak_mcq_local_retry_min_assessments"],
                )
            ),
            "reflection_local_retry_missing_fact_confidence_threshold": float(
                data.get(
                    "reflection_local_retry_missing_fact_confidence_threshold",
                    cfg["reflection_local_retry_missing_fact_confidence_threshold"],
                )
            ),
            "reflection_followup_missing_fact_local_retry": bool(
                data.get(
                    "reflection_followup_missing_fact_local_retry",
                    cfg["reflection_followup_missing_fact_local_retry"],
                )
            ),
            "reflection_followup_missing_fact_confidence_threshold": float(
                data.get(
                    "reflection_followup_missing_fact_confidence_threshold",
                    cfg["reflection_followup_missing_fact_confidence_threshold"],
                )
            ),
            "reflection_direct_local_retry": bool(
                data.get("reflection_direct_local_retry", cfg["reflection_direct_local_retry"])
            ),
            "reflection_direct_retry_answer_only": bool(
                data.get(
                    "reflection_direct_retry_answer_only",
                    cfg["reflection_direct_retry_answer_only"],
                )
            ),
            "reflection_extra_parameter_local_retry": bool(
                data.get(
                    "reflection_extra_parameter_local_retry",
                    cfg["reflection_extra_parameter_local_retry"],
                )
            ),
            "reflection_retry_multiquery_enabled": bool(
                data.get(
                    "reflection_retry_multiquery_enabled",
                    cfg["reflection_retry_multiquery_enabled"],
                )
            ),
            "reflection_retry_max_queries": int(
                data.get("reflection_retry_max_queries", cfg["reflection_retry_max_queries"])
            ),
            "reflection_retry_dense_k": int(
                data.get("reflection_retry_dense_k", cfg["reflection_retry_dense_k"])
            ),
            "reflection_retry_bm25_k": int(
                data.get("reflection_retry_bm25_k", cfg["reflection_retry_bm25_k"])
            ),
            "reflection_retry_max_evidence": int(
                data.get("reflection_retry_max_evidence", cfg["reflection_retry_max_evidence"])
            ),
            "enable_main_evidence_brief": bool(
                data.get("enable_main_evidence_brief", cfg["enable_main_evidence_brief"])
            ),
            "main_evidence_brief_max_records": int(
                data.get("main_evidence_brief_max_records", cfg["main_evidence_brief_max_records"])
            ),
            "main_evidence_brief_max_chars": int(
                data.get("main_evidence_brief_max_chars", cfg["main_evidence_brief_max_chars"])
            ),
            "prefetch_local_before_main": bool(
                data.get("prefetch_local_before_main", cfg["prefetch_local_before_main"])
            ),
            "prefetch_local_requires_reflection": bool(
                data.get(
                    "prefetch_local_requires_reflection",
                    cfg["prefetch_local_requires_reflection"],
                )
            ),
            "prefetch_local_for_mcq": bool(
                data.get("prefetch_local_for_mcq", cfg["prefetch_local_for_mcq"])
            ),
            "prefetch_local_for_clinical_context": bool(
                data.get(
                    "prefetch_local_for_clinical_context",
                    cfg["prefetch_local_for_clinical_context"],
                )
            ),
            "prefetch_local_goal_mode": str(
                data.get("prefetch_local_goal_mode", cfg["prefetch_local_goal_mode"]) or ""
            ).strip().lower(),
            "prefetch_local_entity_mode": str(
                data.get("prefetch_local_entity_mode", cfg["prefetch_local_entity_mode"]) or ""
            ).strip().lower(),
            "prefetch_local_include_choices": bool(
                data.get("prefetch_local_include_choices", cfg["prefetch_local_include_choices"])
            ),
            "prefetch_local_max_evidence": int(
                data.get("prefetch_local_max_evidence", cfg["prefetch_local_max_evidence"])
            ),
            "prefetch_local_dense_k": int(
                data.get("prefetch_local_dense_k", cfg["prefetch_local_dense_k"])
            ),
            "prefetch_local_bm25_k": int(
                data.get("prefetch_local_bm25_k", cfg["prefetch_local_bm25_k"])
            ),
            "unplanned_search_local_dense_k": int(
                data.get("unplanned_search_local_dense_k", cfg["unplanned_search_local_dense_k"])
            ),
            "unplanned_search_local_bm25_k": int(
                data.get("unplanned_search_local_bm25_k", cfg["unplanned_search_local_bm25_k"])
            ),
            "unplanned_search_local_max_evidence": int(
                data.get("unplanned_search_local_max_evidence", cfg["unplanned_search_local_max_evidence"])
            ),
            "max_total_tool_calls": int(data.get("max_total_tool_calls", cfg["max_total_tool_calls"])),
            "max_online_tool_calls": int(data.get("max_online_tool_calls", cfg["max_online_tool_calls"])),
            "max_local_tool_calls": int(data.get("max_local_tool_calls", cfg["max_local_tool_calls"])),
            "max_llm_calls": int(data.get("max_llm_calls", cfg["max_llm_calls"])),
            "max_total_tokens": int(data.get("max_total_tokens", cfg["max_total_tokens"])),
            "deadline_seconds": float(data.get("deadline_seconds", cfg["deadline_seconds"])),
            "enforce_goal_propagation": bool(data.get("enforce_goal_propagation", cfg["enforce_goal_propagation"])),
            "retrieval_max_rounds": int(data.get("retrieval_max_rounds", cfg["retrieval_max_rounds"])),
            "compression_chunk_tokens": int(data.get("compression_chunk_tokens", cfg["compression_chunk_tokens"])),
            "max_model_visible_tool_result_chars": int(
                data.get(
                    "max_model_visible_tool_result_chars",
                    cfg["max_model_visible_tool_result_chars"],
                )
            ),
            "planner_max_retries": int(data.get("planner_max_retries", cfg["planner_max_retries"])),
            "planner_retry_max_tokens": int(data.get("planner_retry_max_tokens", cfg["planner_retry_max_tokens"])),
            "screening_max_tokens": int(data.get("screening_max_tokens", cfg["screening_max_tokens"])),
            "screening_max_retries": int(data.get("screening_max_retries", cfg["screening_max_retries"])),
            "qwen_tokenizer_name": str(data.get("qwen_tokenizer_name", cfg["qwen_tokenizer_name"])).strip() or cfg["qwen_tokenizer_name"],
            "use_primekg": bool(data.get("use_primekg", cfg["use_primekg"])),
            "isolate_get_all_info_by_drug_process": bool(
                data.get(
                    "isolate_get_all_info_by_drug_process",
                    cfg["isolate_get_all_info_by_drug_process"],
                )
            ),
            "get_all_info_by_drug_subprocess_timeout_seconds": int(
                data.get(
                    "get_all_info_by_drug_subprocess_timeout_seconds",
                    cfg["get_all_info_by_drug_subprocess_timeout_seconds"],
                )
            ),
            "enabled_tools": _parse_tool_list(data.get("enabled_tools")),
            "disabled_tools": _parse_tool_list(data.get("disabled_tools")),
            "strict_tool_registry": bool(data.get("strict_tool_registry", cfg["strict_tool_registry"])),
            "main_prompt_variant": str(data.get("main_prompt_variant", cfg["main_prompt_variant"])).strip() or cfg["main_prompt_variant"],
            "dump_prompts": bool(data.get("dump_prompts", cfg["dump_prompts"])),
            "prompt_dump_dir": str(data.get("prompt_dump_dir", cfg["prompt_dump_dir"])).strip() or cfg["prompt_dump_dir"],
        })
    return cfg


def _is_unbounded_limit(value: int) -> bool:
    return int(value) < 0


class Agent:
    def __init__(
        self,
        memory: Optional[Memory] = None,
        llm: Optional[DeepSeekClient] = None,
        config: Optional[Dict[str, Any]] = None,
        output_sink: Optional[Any] = None,
    ):
        self.config = config or _load_agent_config()
        os.environ["AGENT_TOOLS_USE_PRIMEKG"] = "1" if bool(self.config.get("use_primekg", True)) else "0"
        os.environ["AGENT_GET_ALL_INFO_BY_DRUG_ISOLATE"] = (
            "1" if bool(self.config.get("isolate_get_all_info_by_drug_process", sys.platform == "darwin")) else "0"
        )
        os.environ["AGENT_GET_ALL_INFO_BY_DRUG_SUBPROCESS_TIMEOUT_SECONDS"] = str(
            max(1, int(self.config.get("get_all_info_by_drug_subprocess_timeout_seconds", 240) or 240))
        )
        os.environ["AGENT_TOOL_DEEPSEEK_CHAT_MAX_WORKERS"] = str(
            max(1, int(self.config.get("tool_deepseek_chat_max_workers", 16) or 16))
        )
        self.enabled_tools = list(self.config.get("enabled_tools", []) or [])
        self.disabled_tools = list(self.config.get("disabled_tools", []) or [])
        self.current_example_payload = (
            dict(self.config.get("_current_example_payload"))
            if isinstance(self.config.get("_current_example_payload"), dict)
            else {}
        )
        self.strict_tool_registry = bool(self.config.get("strict_tool_registry", False))
        self.allowed_tools = list(
            get_allowed_tools(
                enabled_tools=self.enabled_tools,
                disabled_tools=self.disabled_tools,
                strict=self.strict_tool_registry,
            )
        )
        self.allowed_tool_set = set(self.allowed_tools)
        self.current_plan: Dict[str, Any] = (
            fallback_plan("")
            if bool(self.config.get("enable_planner", False))
            else disabled_plan()
        )
        self.current_user_text: str = ""
        self.evidence_records: List[Dict[str, Any]] = []
        self.reflection_history: List[Dict[str, Any]] = []
        self.debate_history: List[Dict[str, Any]] = []
        self.debate_state: Dict[str, Any] = {}
        self.planner_raw: str = ""
        self.candidate_plan: Dict[str, Any] = {}
        self.relation_screening_raw: str = ""
        self.execution_budget = ExecutionBudget.from_config(self.config)
        self.goal_guidance_enabled = bool(
            self.config.get(
                "enable_goal_guidance",
                self.config.get("enable_planner", False),
            )
        )
        self.main_prompt_variant = str(self.config.get("main_prompt_variant", "default")).strip() or "default"
        self.main_system_prompt = self._build_main_prompt(
            enabled_tools=self.allowed_tools,
            disabled_tools=self.disabled_tools,
            remaining_tool_turns=int(self.config.get("max_tool_turns", 3)),
            max_tools_per_turn=int(self.config.get("max_tools_per_turn", 3)),
            plan_context="",
        )
        self.answer_only_system_prompt = build_answer_only_prompt(
            enabled_tools=self.allowed_tools,
            disabled_tools=self.disabled_tools,
            goal_guidance_enabled=self.goal_guidance_enabled,
        )
        self.reflection_system_prompt = build_reflection_system_prompt(
            goal_guidance_enabled=self.goal_guidance_enabled,
            online_evidence_enabled=bool(self.allowed_tool_set & ONLINE_EVIDENCE_TOOLS),
            local_evidence_enabled=bool(self.allowed_tool_set & LOCAL_EVIDENCE_TOOLS),
            retry_enabled=bool(self.config.get("reflection_retry_enabled", True)),
        )
        self.planner_system_prompt = ""
        if bool(self.config.get("enable_planner", False)):
            self.planner_system_prompt = ENTITY_PLAN_SYSTEM_PROMPT
        self._maybe_dump_prompts()
        visible_tool_result_limit = int(
            self.config.get("max_model_visible_tool_result_chars", 12000) or 12000
        )
        self.memory = memory or Memory(
            self.main_system_prompt,
            max_model_visible_tool_result_chars=visible_tool_result_limit,
        )
        # Tests and integrations may inject a pre-existing Memory instance.
        # Apply the frozen runtime limit to that instance as well.
        set_visible_limit = getattr(
            self.memory,
            "set_max_model_visible_tool_result_chars",
            None,
        )
        if callable(set_visible_limit):
            set_visible_limit(visible_tool_result_limit)
        self.llm = llm or DeepSeekClient()
        self._sink = output_sink or sys.__stdout__

    def _refresh_available_tools(self) -> None:
        self.allowed_tools = get_allowed_tools(enabled_tools=self.enabled_tools,
                                               disabled_tools=self.disabled_tools,
                                               strict=self.strict_tool_registry)
        self.allowed_tool_set = set(self.allowed_tools)

    def _build_main_prompt(
        self,
        *,
        enabled_tools: List[str],
        disabled_tools: List[str],
        remaining_tool_turns: int,
        max_tools_per_turn: int,
        plan_context: str = "",
    ) -> str:
        if self.main_prompt_variant == "task_form_routing":
            prompt = build_main_system_prompt_task_form_routing(
                enabled_tools=enabled_tools,
                disabled_tools=disabled_tools,
                remaining_tool_turns=remaining_tool_turns,
                max_tools_per_turn=max_tools_per_turn,
                max_search_local_calls=self._remaining_search_local_calls_for_prompt(enabled_tools),
                plan_context=plan_context,
                goal_guidance_enabled=self.goal_guidance_enabled,
            )
        else:
            prompt = build_main_system_prompt(
                enabled_tools=enabled_tools,
                disabled_tools=disabled_tools,
                remaining_tool_turns=remaining_tool_turns,
                max_tools_per_turn=max_tools_per_turn,
                max_search_local_calls=self._remaining_search_local_calls_for_prompt(enabled_tools),
                plan_context=plan_context,
                goal_guidance_enabled=self.goal_guidance_enabled,
            )
        evidence_context = self._main_evidence_brief_context(self.current_user_text)
        if evidence_context:
            prompt += "\n\n" + evidence_context
        return prompt

    def _remaining_search_local_calls_for_prompt(self, enabled_tools: List[str]) -> Optional[int]:
        allowed = {str(item).strip() for item in (enabled_tools or []) if str(item).strip()}
        if allowed != {"search_local"}:
            return None
        limit = int(self.execution_budget.max_local_tool_calls)
        used = int(self.execution_budget.local_tool_calls)
        if limit < 0:
            return None
        return max(0, limit - used)

    def _active_plan_context(self) -> str:
        if not bool(self.config.get("enable_planner", False)):
            return ""
        return render_plan_for_model(self.current_plan)

    def _refresh_answer_prompt(self) -> None:
        self.answer_only_system_prompt = build_answer_only_prompt(
            enabled_tools=self.allowed_tools,
            disabled_tools=self.disabled_tools,
            plan_context=self._active_plan_context(),
            goal_guidance_enabled=self.goal_guidance_enabled,
        )
        evidence_context = self._main_evidence_brief_context(self.current_user_text)
        if evidence_context:
            self.answer_only_system_prompt += "\n\n" + evidence_context

    def _current_total_tokens(self) -> int:
        try:
            usage = self.llm.get_usage_snapshot(reset=False)
        except Exception:
            return 0
        if not isinstance(usage, dict):
            return 0
        try:
            return int(usage.get("total_tokens", 0) or 0)
        except Exception:
            return 0

    def _chat(
        self,
        messages: List[Dict[str, Any]],
        *,
        json_mode: bool = False,
        max_tokens: Optional[int] = None,
        json_thinking_enabled: Optional[bool] = None,
    ) -> str:
        allowed, reason = self.execution_budget.can_call_llm(self._current_total_tokens())
        if not allowed:
            raise RuntimeError(f"Execution budget blocked LLM call: {reason}")
        self.execution_budget.record_llm_call()
        if json_mode and hasattr(self.llm, "chat_json"):
            if json_thinking_enabled is not None:
                try:
                    return self.llm.chat_json(
                        messages,
                        max_tokens=max_tokens,
                        thinking_enabled=bool(json_thinking_enabled),
                    )
                except TypeError:
                    # Test doubles and older compatible clients may not expose
                    # the optional per-call thinking override.
                    pass
            if max_tokens is None:
                return self.llm.chat_json(messages)
            try:
                return self.llm.chat_json(messages, max_tokens=max_tokens)
            except TypeError:
                return self.llm.chat_json(messages)
        if max_tokens is None:
            return self.llm.chat(messages)
        try:
            return self.llm.chat(messages, max_tokens=max_tokens)
        except TypeError:
            return self.llm.chat(messages)

    def _resolve_prompt_dump_dir(self) -> Path:
        raw_path = str(self.config.get("prompt_dump_dir", "") or "").strip()
        if not raw_path:
            raw_path = "output/prompt_dumps/agent_runtime"
        dump_dir = Path(raw_path)
        if not dump_dir.is_absolute():
            dump_dir = Path(_repo_root()) / dump_dir
        return dump_dir

    def _write_prompt_file(self, path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    def _maybe_dump_prompts(self) -> None:
        if not bool(self.config.get("dump_prompts", False)):
            return

        dump_dir = self._resolve_prompt_dump_dir()
        self._write_prompt_file(dump_dir / "main_system_prompt.txt", self.main_system_prompt)
        self._write_prompt_file(dump_dir / "answer_only_system_prompt.txt", self.answer_only_system_prompt)
        self._write_prompt_file(
            dump_dir / "reflection_system_prompt.txt",
            self.reflection_system_prompt.strip() + "\n",
        )
        self._write_prompt_file(dump_dir / "debate_system_prompt.txt", DEBATE_SYSTEM_PROMPT)

        planner_path = dump_dir / "planner_system_prompt.txt"
        screening_path = dump_dir / "relation_screening_system_prompt.txt"
        if self.planner_system_prompt.strip():
            self._write_prompt_file(planner_path, self.planner_system_prompt)
            self._write_prompt_file(screening_path, RELATION_SCREENING_SYSTEM_PROMPT)
        else:
            for path in (planner_path, screening_path):
                if path.exists():
                    path.unlink()

        manifest = {
            "main_prompt_variant": self.main_prompt_variant,
            "enable_planner": bool(self.config.get("enable_planner", False)),
            "enable_reflection": bool(self.config.get("enable_reflection", False)),
            "enabled_tools": list(self.allowed_tools),
            "files": {
                "main_system_prompt": str(dump_dir / "main_system_prompt.txt"),
                "answer_only_system_prompt": str(dump_dir / "answer_only_system_prompt.txt"),
                "reflection_system_prompt": str(dump_dir / "reflection_system_prompt.txt"),
                "debate_system_prompt": str(dump_dir / "debate_system_prompt.txt"),
                "planner_system_prompt": str(planner_path) if self.planner_system_prompt.strip() else "",
                "relation_screening_system_prompt": str(screening_path) if self.planner_system_prompt.strip() else "",
            },
        }
        self._write_prompt_file(
            dump_dir / "prompt_manifest.json",
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        )

    def _log(self, text: str) -> None:
        try:
            if text and not text.endswith("\n"):
                text = text + "\n"
            self._sink.write(text)
            self._sink.flush()
        except Exception:
            pass

    def _is_tool_call(self, text: str) -> bool:
        if text is None:
            return False
        return (
            TOOL_CALL_PREFIX in str(text)
            or bool(re.search(r"<(?:｜｜|\|\|)DSML(?:｜｜|\|\|)\s+(?:calls|invoke)\b", str(text)))
            or len(self._extract_all_tool_calls(text)) > 0
        )

    def _truncate_to_tool_call_head(self, text: str) -> str:
        lines = text.splitlines()
        last_idx = -1
        for i, line in enumerate(lines):
            s = line.strip()
            if s.startswith(TOOL_CALL_PREFIX):
                last_idx = i
            else:
                for name in self.allowed_tools:
                    if name in s and "{" in s:
                        last_idx = i
        if last_idx == -1:
            return text
        return "\n".join(lines[: last_idx + 1])

    def _extract_braced_json(self, text: str, start_idx: int) -> Tuple[Optional[str], int]:
        depth = 0
        i = start_idx
        in_str = False
        esc = False
        quote = ""
        while i < len(text):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == quote:
                    in_str = False
            else:
                if ch in ['"', "'"]:
                    in_str = True
                    quote = ch
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        return text[start_idx:i + 1], i
            i += 1
        return None, start_idx

    def _extract_dsml_tool_calls(self, text: str) -> List[Tuple[str, Dict[str, Any]]]:
        """Decode complete DSML calls; execution still uses the registry and budgets."""
        normalized = re.sub(r"<(/?)(?:｜｜|\|\|)DSML(?:｜｜|\|\|)\s+", r"<\1", text)
        blocks = list(re.finditer(r"<calls>.*?</calls>", normalized, flags=re.DOTALL))
        if not blocks or len(blocks) != normalized.count("<calls>"):
            return []
        remainder = re.sub(r"<calls>.*?</calls>", "", normalized, flags=re.DOTALL)
        if re.search(r"</?(?:calls|invoke|parameter)\b", remainder):
            return []
        results: List[Tuple[str, Dict[str, Any]]] = []
        try:
            for block in blocks:
                root = ET.fromstring(block.group())
                if not len(root) or (root.text or "").strip():
                    return []
                for invoke in root:
                    name = invoke.attrib.get("name", "")
                    if invoke.tag != "invoke" or set(invoke.attrib) != {"name"} or name not in self.allowed_tool_set:
                        return []
                    if (invoke.text or "").strip() or (invoke.tail or "").strip():
                        return []
                    args: Dict[str, Any] = {}
                    for parameter in invoke:
                        key = parameter.attrib.get("name", "")
                        mode = parameter.attrib.get("string", "true")
                        if (
                            parameter.tag != "parameter" or not key or key in args or len(parameter)
                            or set(parameter.attrib) - {"name", "string"}
                            or mode not in {"true", "false"} or (parameter.tail or "").strip()
                        ):
                            return []
                        value = parameter.text or ""
                        args[key] = value if mode == "true" else json.loads(value)
                    results.append((name, args))
        except (ET.ParseError, ValueError, TypeError):
            return []
        return results

    def _extract_all_tool_calls(self, text: str) -> List[Tuple[str, Dict[str, Any]]]:
        if re.search(r"<(?:｜｜|\|\|)DSML(?:｜｜|\|\|)\s+(?:calls|invoke)\b", text):
            return self._extract_dsml_tool_calls(text)
        results: List[Tuple[str, Dict[str, Any]]] = []

        for raw_line in text.splitlines():
            line = raw_line.strip()
            if line.startswith(TOOL_CALL_PREFIX):
                line = line[len(TOOL_CALL_PREFIX):].strip()

            cursor = 0
            while cursor < len(line):
                while cursor < len(line) and line[cursor] in {" ", ","}:
                    cursor += 1
                if cursor >= len(line):
                    break

                if line[cursor] == "{":
                    json_str, end_pos = self._extract_braced_json(line, cursor)
                    if json_str:
                        try:
                            explicit_payload = json.loads(json_str)
                        except Exception:
                            explicit_payload = None
                        if isinstance(explicit_payload, dict):
                            explicit_name = str(explicit_payload.get("name", "")).strip()
                            explicit_args = explicit_payload.get("arguments")
                            if explicit_name in self.allowed_tool_set:
                                parsed_args: Dict[str, Any] | None = None
                                if isinstance(explicit_args, dict):
                                    parsed_args = explicit_args
                                else:
                                    flat_args = {
                                        key: value
                                        for key, value in explicit_payload.items()
                                        if key not in {"name", "arguments"}
                                    }
                                    if flat_args:
                                        parsed_args = flat_args
                                if parsed_args is not None:
                                    results.append((explicit_name, parsed_args))
                                    cursor = end_pos + 1
                                    continue

                match = re.search(r"([A-Za-z_]\w*)\s+\{", line[cursor:])
                if not match:
                    break

                tool_name = match.group(1)
                if tool_name not in self.allowed_tool_set:
                    cursor += match.end(1)
                    continue

                brace_start = cursor + match.end(0) - 1
                json_str, end_pos = self._extract_braced_json(line, brace_start)
                if not json_str:
                    break

                try:
                    args = json.loads(json_str)
                    if isinstance(args, dict):
                        results.append((tool_name, args))
                except Exception:
                    pass

                cursor = end_pos + 1
        if results:
            return results

        informal_matches: List[Tuple[int, str, Dict[str, Any]]] = []
        lower_text = text.lower()
        for tool_name in self.allowed_tools:
            patterns = [
                f"`{tool_name}`",
                f" {tool_name} ",
                f'"{tool_name}"',
            ]
            positions = [lower_text.find(pattern.lower()) for pattern in patterns]
            positions = [position for position in positions if position >= 0]
            if not positions:
                continue
            start_idx = min(positions)
            brace_start = text.find("{", start_idx)
            if brace_start == -1:
                continue
            json_str, _ = self._extract_braced_json(text, brace_start)
            if not json_str:
                continue
            try:
                args = json.loads(json_str)
            except Exception:
                continue
            if isinstance(args, dict):
                informal_matches.append((start_idx, tool_name, args))

        informal_matches.sort(key=lambda item: item[0])
        deduped: List[Tuple[str, Dict[str, Any]]] = []
        seen: set[Tuple[str, str]] = set()
        for _, tool_name, args in informal_matches:
            signature = (tool_name, json.dumps(args, ensure_ascii=False, sort_keys=True))
            if signature in seen:
                continue
            seen.add(signature)
            deduped.append((tool_name, args))
        return deduped

    def _normalize_tool_call_message(self, raw_text: str, tool_calls: List[Tuple[str, Dict[str, Any]]]) -> str:
        normalized_block = render_tool_call_block(tool_calls)
        dsml_start = re.search(r"<(?:｜｜|\|\|)DSML(?:｜｜|\|\|)\s+calls\b", raw_text)
        if dsml_start:
            prefix = raw_text[:dsml_start.start()].strip()
            return (prefix + "\n" if prefix else "") + normalized_block
        if "TOOL_CALL:" in raw_text:
            prefix = raw_text.split("TOOL_CALL:", 1)[0].strip()
            if not prefix:
                return normalized_block
            return prefix + "\n" + normalized_block
        cleaned = raw_text.strip()
        if not cleaned:
            return normalized_block
        if cleaned.endswith(normalized_block):
            return cleaned
        return cleaned + "\n" + normalized_block

    def _error_result(self, tool_name: str, msg: str) -> Dict[str, Any]:
        return {"success": False, "error": msg, "tool": tool_name}

    def _record_rejected_tool_call(self, tool_name: str, reason: str) -> None:
        recorder = get_active_timing_recorder()
        if recorder is not None:
            recorder.record_tool_call(
                tool_name=tool_name,
                elapsed_seconds=0.0,
                success=False,
                metadata={"rejected_before_execution": True, "reason": reason},
            )

    def _parse_choice_label(self, text: Optional[str]) -> str:
        if not text:
            return "None"
        s = str(text).strip().upper()

        valid_labels = set("ABCDEFGHIJ")
        if s and s[0] in valid_labels:
            return s[0]

        patterns = [
            r"(?:answer is|answer:|is)\s*([A-J])",
            r"([A-J])\)",
            r"\b([A-J])\b",
        ]
        for pat in patterns:
            match = re.search(pat, s)
            if match:
                return match.group(1)

        if "NONE" in s:
            return "None"
        return "None"

    def judge(self, prompt: str) -> str:
        try:
            messages = [{"role": "user", "content": str(prompt)}]
            raw = self._chat(messages, max_tokens=200)
            return self._parse_choice_label(raw)
        except Exception:
            return "None"

    def _tool_context(self) -> Dict[str, Any]:
        current_example = self.current_example_payload
        if isinstance(current_example, dict):
            current_example = {
                key: current_example.get(key)
                for key in (
                    "question",
                    "choices",
                    "answer_type",
                )
                if key in current_example
            }
        return {
            "message_context": self._memory_context(include_tool_payload=True),
            "plan": dict(self.current_plan),
            "goal": plan_goal_context(self.current_plan) if bool(self.config.get("enable_planner", False)) else "",
            "current_example": current_example,
        }

    def _current_example_answer_type(self) -> str:
        current_example = self.current_example_payload
        if not isinstance(current_example, dict):
            return ""
        return str(current_example.get("answer_type", "") or "").strip()

    def _looks_like_multiple_choice_task(self, user_text: str) -> bool:
        if getattr(self, "config", {}).get("answer_type") == "open":
            return False
        answer_type = self._current_example_answer_type()
        if answer_type in {"multipleChoice", "multi_choice", "open_ended_multi_choice"}:
            return True
        text = str(user_text or "")
        if re.search(r"\b(?:Answer\s+Choices?|Choices)\s*:", text, flags=re.IGNORECASE):
            return True
        return re.search(r"(?m)^\s*[A-Z][\.\)]\s+\S+", text) is not None

    def _looks_like_clinical_context_task(self, user_text: str) -> bool:
        text = re.sub(r"\s+", " ", str(user_text or "")).strip()
        if not text:
            return False
        patient_context = bool(
            re.search(
                r"\b(?:patient|man|woman|male|female|child|infant|adolescent|"
                r"history|presents?|presented|reports?|complains?|symptoms?|signs?|"
                r"examination|physical exam|vital signs?|blood pressure|heart rate|"
                r"laborator(?:y|ies)|labs?|serum|urine|platelet|creatinine|imaging|"
                r"diagnos(?:is|ed)|medication list|taking|receiving|currently|"
                r"pregnan(?:t|cy)|postpartum|contraindication|comorbid)\b",
                text,
                flags=re.IGNORECASE,
            )
        )
        decision_context = bool(
            re.search(
                r"\b(?:management|treatment|therapy|next step|recommend(?:ation|ed)?|"
                r"counsel(?:ing)?|select|choose|most appropriate|least appropriate|"
                r"best|safest|avoid|switch|stop|hold|discontinue|administer|dose|"
                r"route|monitor(?:ing)?|interpret|based on this information)\b",
                text,
                flags=re.IGNORECASE,
            )
        )
        return patient_context and (decision_context or self._looks_like_multiple_choice_task(user_text))

    def _answer_choices_fragment(self, user_text: str, *, limit: int = 900) -> str:
        text = str(user_text or "")
        match = re.search(r"(?is)(Answer Choices:\s*.+)$", text)
        if match:
            return re.sub(r"\s+", " ", match.group(1)).strip()[:limit]
        choices: list[str] = []
        for line in text.splitlines():
            if re.match(r"\s*[A-Z][\.\)]\s+", line):
                choices.append(line.strip())
        return " ".join(choices)[:limit]

    def _current_example_choices(self) -> Any:
        current_example = self.current_example_payload if isinstance(self.current_example_payload, dict) else {}
        return current_example.get("choices")

    def _choice_texts_for_routing(self, user_text: str) -> List[str]:
        choices = self._current_example_choices()
        texts: List[str] = []
        if isinstance(choices, dict):
            texts = [str(value or "").strip() for _, value in sorted(choices.items())]
        elif isinstance(choices, list):
            for item in choices:
                if isinstance(item, dict):
                    value = item.get("text") or item.get("value") or item.get("answer")
                else:
                    value = item
                if str(value or "").strip():
                    texts.append(str(value or "").strip())
        if not texts:
            texts = list(self._choice_map_for_reflection(user_text).values())
        return [re.sub(r"\s+", " ", item).strip()[:500] for item in texts if str(item).strip()]

    def _looks_like_drug_or_product_text(self, text: str) -> bool:
        normalized = re.sub(r"\s+", " ", str(text or "")).strip()
        if not normalized:
            return False
        lower = normalized.casefold()
        known_terms = {
            "acetaminophen", "acyclovir", "albuterol", "amoxicillin", "ampicillin",
            "aspirin", "azithromycin", "budesonide", "ceftriaxone", "ciprofloxacin",
            "clindamycin", "clopidogrel", "daptomycin", "dextrose", "famotidine",
            "fluconazole", "formoterol", "gabapentin", "glargine", "glycopyrrolate",
            "heparin", "ibuprofen", "insulin", "levofloxacin", "meropenem",
            "metformin", "methotrexate", "pantoprazole", "phytonadione",
            "prednisone", "regular insulin", "rivaroxaban", "simvastatin",
            "sucralfate", "tiotropium", "vancomycin", "warfarin",
        }
        if any(term in lower for term in known_terms):
            return True
        for token in re.findall(r"\b[A-Za-z][A-Za-z0-9-]{3,42}\b", normalized):
            token_l = token.casefold().strip("-")
            if any(
                token_l.endswith(suffix)
                for suffix in (
                    "ab", "afil", "azole", "caine", "cillin", "cycline", "dipine",
                    "epine", "floxacin", "gliflozin", "gliptin", "mab", "mycin",
                    "nib", "olol", "oxetine", "penem", "prazole", "pril",
                    "sartan", "setron", "statin", "terol", "tidine", "vir",
                    "xaban", "xone", "zepam",
                )
            ):
                return True
        return bool(re.search(r"\b(?:mg|mcg|units?|iu)\b.*\b(?:po|iv|im|sc|subcutaneously|q\d+h|bid|tid|daily)\b", lower))

    def _looks_like_named_drug_label_comparison_task(self, user_text: str) -> bool:
        if not self._looks_like_multiple_choice_task(user_text):
            return False
        choice_texts = self._choice_texts_for_routing(user_text)
        if len(choice_texts) < 4:
            return False
        drugish = sum(1 for item in choice_texts if self._looks_like_drug_or_product_text(item))
        if drugish < max(3, int(len(choice_texts) * 0.55)):
            return False
        text = re.sub(r"\s+", " ", str(user_text or "")).casefold()
        label_cues = bool(
            re.search(
                r"\b(?:safest|contraindicat|avoid|switch|continue|discontinue|hold|"
                r"dose|dosing|route|renal|hepatic|pregnan|lactat|interaction|"
                r"adverse|toxicity|monitor|infusion|administration|therapeutic duplication|"
                r"trough|q\d+h|mg|mcg|units?)\b",
                text,
            )
        )
        named_drug_cues = bool(re.search(r"\b(?:drug|medication|antibiotic|insulin|inhaler|therapy|administration)\b", text))
        return bool(label_cues or named_drug_cues)

    def _should_prefetch_local(self, user_text: str) -> bool:
        if not bool(self.config.get("prefetch_local_before_main", False)):
            return False
        if bool(self.config.get("prefetch_local_requires_reflection", False)) and not bool(
            self.config.get("enable_reflection", False)
        ):
            self._log(
                "[LOCAL PREFETCH | SKIPPED]\n"
                + json.dumps(
                    {
                        "reason": "reflection_coupled_prefetch_disabled",
                        "note": "This local evidence priming path is reserved for reflection-led evidence audit.",
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return False
        if not bool(self.config.get("enable_tools", True)):
            return False
        if "search_local" not in self.allowed_tool_set:
            return False
        if int(self.execution_budget.local_tool_calls) > 0:
            return False
        max_local = int(self.execution_budget.max_local_tool_calls)
        if max_local == 0:
            return False
        can_start, _ = self.execution_budget.can_start_tool_turn()
        if not can_start:
            return False

        if (
            bool(self.config.get("prefetch_local_skip_label_comparison", False))
            and self._looks_like_named_drug_label_comparison_task(user_text)
        ):
            self._log(
                "[LOCAL PREFETCH | SKIPPED]\n"
                + json.dumps(
                    {
                        "reason": "named_drug_label_comparison",
                        "note": "Preserve the structured-label route first; local evidence remains available to reflection if needed.",
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return False

        mcq_ok = bool(self.config.get("prefetch_local_for_mcq", True)) and self._looks_like_multiple_choice_task(user_text)
        clinical_ok = bool(self.config.get("prefetch_local_for_clinical_context", True)) and self._looks_like_clinical_context_task(user_text)
        return bool(mcq_ok or clinical_ok)

    def _planner_local_prefetch_queries(self, user_text: str) -> List[str]:
        queries: List[str] = []
        if bool(self.config.get("enable_planner", False)) and isinstance(self.current_plan, dict):
            entity_by_id = {
                str(item.get("id") or "").strip(): str(item.get("name") or "").strip()
                for item in list(self.current_plan.get("entities") or [])
                if isinstance(item, dict) and str(item.get("id") or "").strip()
            }
            for task in list(self.current_plan.get("search_tasks") or []):
                if not isinstance(task, dict):
                    continue
                channel = str(task.get("channel") or "").strip().lower()
                if channel not in {"local", "either", ""}:
                    continue
                query = str(task.get("query") or "").strip()
                task_goal = str(task.get("goal") or "").strip()
                entity_names = [
                    entity_by_id.get(str(entity_id or "").strip(), "")
                    for entity_id in list(task.get("entity_ids") or [])[:8]
                ]
                parts = [
                    query,
                    "Search goal: " + task_goal if task_goal else "",
                    "Task entities: " + ", ".join(item for item in entity_names if item) if entity_names else "",
                ]
                rendered = self._compact_retry_query_piece(" | ".join(part for part in parts if part), limit=900)
                if rendered:
                    queries.append(rendered)
                if len(queries) >= max(1, int(self.config.get("prefetch_local_max_queries", 2) or 2)):
                    break

        # An empty successful screen means no planned local evidence is needed.
        # Only a failed/legacy plan may fall back to a visible-case query.
        if not queries and self.current_plan.get("screening_status") != "ok":
            generic_query = self._build_local_prefetch_query(user_text)
            if generic_query:
                queries.append(generic_query)
        seen: set[str] = set()
        deduped: List[str] = []
        for query in queries:
            cleaned = self._compact_retry_query_piece(query, limit=1100)
            key = re.sub(r"\W+", " ", cleaned.casefold()).strip()
            if not cleaned or key in seen:
                continue
            seen.add(key)
            deduped.append(cleaned)
        max_queries = max(1, int(self.config.get("prefetch_local_max_queries", 2) or 2))
        return deduped[:max_queries]

    def _planner_local_goal_context(self) -> str:
        if not bool(self.config.get("enable_planner", False)) or not isinstance(self.current_plan, dict):
            return ""
        mode = str(self.config.get("prefetch_local_goal_mode", "anchors") or "anchors").strip().lower()
        if mode in {"", "none", "off", "false"}:
            return ""
        if mode in {"full", "all"}:
            return plan_goal_context(self.current_plan)

        constraints = [
            str(item or "").strip()
            for item in list(self.current_plan.get("constraints") or [])[:4]
            if str(item or "").strip()
        ]
        anchors: List[str] = []
        for item in list(self.current_plan.get("patient_anchors") or [])[:8]:
            if not isinstance(item, dict):
                continue
            fact = str(item.get("fact") or "").strip()
            role = str(item.get("decision_role") or "").strip()
            if fact and role:
                anchors.append(f"{fact} ({role})")
            elif fact:
                anchors.append(fact)
        parts = [
            "Resolve the exact visible medical decision; use retrieved sources only as evidence, not as answer keys.",
            "Decision boundaries: " + "; ".join(constraints) if constraints else "",
            "Patient anchors: " + "; ".join(anchors) if anchors else "",
        ]
        return " | ".join(part for part in parts if part)[:1400]

    def _planner_local_entity_names(self, max_items: int = 12) -> List[str]:
        if not bool(self.config.get("enable_planner", False)) or not isinstance(self.current_plan, dict):
            return []
        if "screening_status" in self.current_plan:
            return plan_entity_names(self.current_plan, max_items=max_items)
        mode = str(self.config.get("prefetch_local_entity_mode", "visible_primary") or "visible_primary").strip().lower()
        if mode in {"", "none", "off", "false"}:
            return []
        if mode in {"full", "all"}:
            return plan_entity_names(self.current_plan, max_items=max_items)

        names: List[str] = []
        for item in list(self.current_plan.get("entities") or []):
            if not isinstance(item, dict):
                continue
            relation = str(item.get("relation") or "").strip().lower()
            provenance = str(item.get("provenance_status") or "").strip().lower()
            importance = str(item.get("importance") or "").strip().lower()
            hop = int(item.get("hop", 0) or 0)
            if "candidate" in relation or "option" in relation:
                continue
            if mode == "visible_primary" and importance != "primary":
                continue
            if mode in {"visible", "visible_primary"} and hop > 0:
                continue
            if mode in {"visible", "visible_primary"} and not provenance.startswith("verified_visible"):
                continue
            name = str(item.get("name") or "").strip()
            if name and name not in names:
                names.append(name)
            if len(names) >= max(1, int(max_items)):
                break
        return names

    def _build_local_prefetch_query(self, user_text: str) -> str:
        visible_question = re.sub(r"\s+", " ", str(user_text or "")).strip()
        choices_text = self._answer_choices_fragment(user_text, limit=1200)
        goal = self._planner_local_goal_context()
        entities = self._planner_local_entity_names(max_items=16)
        parts = [
            "Retrieve source-backed local evidence for the exact visible medical decision.",
            "Focus on patient-specific anchors, contraindications, diagnosis/management boundaries, dose/route/timing/monitoring differences, and option disambiguation.",
            "Visible question: " + visible_question[:1700],
            "Answer choices: " + choices_text if choices_text else "",
            "Decision goal: " + goal if goal else "",
            "Plan entities: " + ", ".join(entities) if entities else "",
        ]
        return "\n".join(part for part in parts if str(part).strip())[:2600]

    def _prefetch_local_evidence(self, user_text: str) -> bool:
        if not self._should_prefetch_local(user_text):
            return False

        queries = (
            self._planner_local_prefetch_queries(user_text)
            if "screening_status" in self.current_plan
            or bool(self.config.get("prefetch_local_use_planner_search_tasks", True))
            else [self._build_local_prefetch_query(user_text)]
        )
        queries = [item for item in queries if str(item).strip()]
        if not queries:
            return False
        tool_args: Dict[str, Any] = {
            "query": queries if len(queries) > 1 else queries[0],
            "choices": (
                self._current_example_choices()
                if bool(self.config.get("prefetch_local_include_choices", True))
                else {}
            ),
            "goal": self._planner_local_goal_context(),
            "entities": self._planner_local_entity_names(max_items=20),
            "require_rag": True,
            "dense_k": max(1, int(self.config.get("prefetch_local_dense_k", 12) or 12)),
            "bm25_k": max(1, int(self.config.get("prefetch_local_bm25_k", 12) or 12)),
            "max_evidence": max(1, int(self.config.get("prefetch_local_max_evidence", 12) or 12)),
        }
        self.memory.add_assistant(render_tool_call_block([("search_local", tool_args)]))
        executed = self._execute_tool_calls(
            [("search_local", tool_args)],
            max_tools_this_turn=1,
            phase_label="LOCAL PREFETCH",
        )
        if executed:
            self.execution_budget.record_tool_turn()
            self._log(
                "[LOCAL PREFETCH | SUMMARY]\n"
                + json.dumps(
                    {
                        "enabled": True,
                        "query_count": len(queries),
                        "query_chars": sum(len(item) for item in queries),
                        "dense_k": tool_args["dense_k"],
                        "bm25_k": tool_args["bm25_k"],
                        "max_evidence": tool_args["max_evidence"],
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
        return bool(executed)

    def _choice_map_for_reflection(self, user_text: str) -> Dict[str, str]:
        if getattr(self, "config", {}).get("answer_type") == "open":
            return {}
        current_example = self.current_example_payload if isinstance(self.current_example_payload, dict) else {}
        raw_choices = current_example.get("choices")
        if isinstance(raw_choices, dict) and raw_choices:
            return {
                str(label).strip().upper(): re.sub(r"\s+", " ", str(text or "")).strip()
                for label, text in raw_choices.items()
                if str(label).strip() and str(text or "").strip()
            }
        text = str(user_text or "")
        marker = re.search(r"\b(?:Answer\s+Choices?|Choices)\s*:\s*", text, flags=re.IGNORECASE)
        if marker:
            text = text[marker.end():]
        starts = list(re.finditer(r"(?m)(?:^|\n)\s*([A-Z])[\.\)]\s+", text))
        choices: Dict[str, str] = {}
        for index, match in enumerate(starts):
            label = match.group(1).strip().upper()
            end = starts[index + 1].start() if index + 1 < len(starts) else len(text)
            option = text[match.end():end]
            # Separate trailing answer instructions before flattening multiline choices.
            option = re.split(
                r"(?im)^\s*(?:Use source-backed evidence\b|End with exactly\s*:|"
                r"(?:Answer|Response|Output) format\s*:|Instructions\s*:|FINAL_ANSWER\s*:)",
                option,
                maxsplit=1,
            )[0]
            option = re.sub(r"\s+", " ", option).strip()
            option = re.sub(r"\s*FINAL_ANSWER\s*:.*$", "", option, flags=re.IGNORECASE).strip()
            if label and option:
                choices[label] = option[:500]
        return choices

    def _reflection_choice_table(self, user_text: str, draft_answer: str) -> str:
        choices = self._choice_map_for_reflection(user_text)
        if not choices:
            return ""
        draft_choice = ""
        final_line = self._extract_final_answer_line(draft_answer)
        match = re.search(r"\b([A-Z])\b", final_line)
        if match and match.group(1).upper() in choices:
            label = match.group(1).upper()
            draft_choice = f"\nDraft selected option: {label}. {choices[label]}"
        lines = ["Multiple-choice option table:"]
        for label in sorted(choices):
            lines.append(f"- {label}. {choices[label]}")
        lines.append(
            "When reflecting, choose the top two from this full option table, not only from the draft rationale. "
            "If a source-backed option outside the draft's comparison better matches the exact stem, include it in top_two. "
            "If one option adds extra monitoring, treatment intensity, dose, route, timing, or combination therapy, "
            "require direct evidence for that extra parameter. Set needs_revision=true only when cited evidence directly "
            "supports replacing the draft, then put the full corrected FINAL_ANSWER in revised_answer."
        )
        return "\n".join(lines) + draft_choice + "\n\n"

    def _reflection_option_evidence_hints(
        self,
        user_text: str,
        *,
        compact: bool = False,
    ) -> str:
        choices = self._choice_map_for_reflection(user_text)
        if not choices:
            return ""
        evidence = self._reflection_evidence_ledger(compact=compact)
        if not evidence:
            return ""

        def tokens(text: str) -> set[str]:
            raw = re.findall(r"[A-Za-z][A-Za-z0-9-]{2,}|\d+(?:\.\d+)?%?", str(text or "").casefold())
            stop = {
                "and", "are", "for", "from", "has", "have", "into", "not", "the", "then", "this",
                "that", "with", "without", "would", "should", "could", "most", "least", "patient",
                "patients", "treatment", "therapy", "drug", "dose", "use", "used", "using",
            }
            return {item.strip("-") for item in raw if item.strip("-") and item.strip("-") not in stop}

        choice_tokens = {label: tokens(text) for label, text in choices.items()}
        rows: list[dict[str, Any]] = []
        for label, option_text in sorted(choices.items()):
            option_tokens = choice_tokens.get(label) or set()
            if not option_tokens:
                continue
            matches: list[dict[str, Any]] = []
            for record in evidence:
                record_text = " ".join(
                    str(record.get(key) or "")
                    for key in ("title", "entity", "source_type", "content")
                )
                overlap = sorted(option_tokens & tokens(record_text))
                if not overlap:
                    continue
                matches.append(
                    {
                        "record_id": str(record.get("record_id") or ""),
                        "tool": str(record.get("tool") or ""),
                        "authority_tier": record.get("authority_tier"),
                        "overlap": overlap[:8],
                    }
                )
            if matches:
                matches.sort(
                    key=lambda item: (
                        len(item.get("overlap") or []),
                        1 if item.get("tool") == "search_local" else 0,
                        -int(item.get("authority_tier") or 4),
                    ),
                    reverse=True,
                )
                rows.append(
                    {
                        "label": label,
                        "text": option_text[:180],
                        "evidence_matches": matches[: 2 if compact else 4],
                    }
                )
        if not rows:
            return ""
        max_rows = 6 if compact else 10
        payload = json.dumps(rows[:max_rows], ensure_ascii=False, indent=2)
        return (
            "Option-evidence coverage hints (retrieval aid, not an answer key):\n"
            "Use this only to ensure source-supported options are not omitted from top_two_candidates; "
            "overlap alone is not sufficient to select an answer.\n"
            f"{payload}\n\n"
        )

    def _main_evidence_brief_context(self, user_text: str) -> str:
        """Expose retrieved evidence structure to the main answer stage.

        This is an evidence-module artifact rather than reflection feedback: it
        contains only records already retrieved by tools plus visible
        option-overlap hints. It must not score options or infer correctness.
        """
        if not bool(self.config.get("enable_main_evidence_brief", True)):
            return ""
        if not self.evidence_records:
            return ""
        max_records = max(1, int(self.config.get("main_evidence_brief_max_records", 8) or 8))
        records = self._reflection_evidence_ledger(compact=True)[:max_records]
        if not records:
            return ""
        rows: list[dict[str, Any]] = []
        for record in records:
            rows.append(
                {
                    "record_id": str(record.get("record_id") or ""),
                    "tool": str(record.get("tool") or ""),
                    "title": str(record.get("title") or "")[:160],
                    "entity": str(record.get("entity") or "")[:120],
                    "source_type": str(record.get("source_type") or "")[:80],
                    "authority_tier": record.get("authority_tier"),
                    "content_kind": str(record.get("content_kind") or "")[:80],
                    "content": str(record.get("content") or "")[:420],
                }
            )
        evidence_text = json.dumps(rows, ensure_ascii=False, indent=2)
        option_hints = self._reflection_option_evidence_hints(user_text, compact=True)
        brief = (
            "Evidence brief for answer synthesis (retrieved evidence only; not an answer key):\n"
            "- Use these source-preserved records to answer directly when they distinguish the visible candidates.\n"
            "- Treat option-evidence overlap as a retrieval/coverage hint only; do not select an answer from overlap alone.\n"
            "- If a record is a local support/differentiator atom, use it as a pointer to the cited source fact, not as hidden benchmark metadata.\n"
            f"{evidence_text}\n\n"
            f"{option_hints}"
        )
        bounded, _ = bound_model_visible_text(
            brief,
            max_chars=max(1200, int(self.config.get("main_evidence_brief_max_chars", 6000) or 6000)),
        )
        return bounded

    def _has_inspected_or_structured_evidence(self) -> bool:
        inspected_tools = {
            "search_local",
            "web_visit",
            "get_all_info_by_drug",
            "get_drug_patents",
            "get_drug_exlusivities",
            "get_drug_approvals",
            "get_trial_info",
        }
        for record in self.evidence_records:
            if not isinstance(record, dict):
                continue
            if str(record.get("tool") or "").strip() in inspected_tools:
                return True
        return False

    def _can_retry_local_evidence(self) -> bool:
        if not bool(self.allowed_tool_set & LOCAL_EVIDENCE_TOOLS):
            return False
        max_local = int(self.execution_budget.max_local_tool_calls)
        if max_local >= 0 and int(self.execution_budget.local_tool_calls) >= max_local:
            return False
        retry_possible, _ = self.execution_budget.can_start_tool_turn()
        return bool(retry_possible)

    def _weak_mcq_local_evidence_gap(
        self,
        *,
        assessments: List[Dict[str, Any]],
        top_two: List[Dict[str, Any]],
        user_text: str,
    ) -> bool:
        """Return true when MCQ reflection lacks usable local support.

        This is deliberately benchmark-agnostic: it only checks whether the
        reflection compared leading candidates using relevant local evidence.
        """
        if not self._looks_like_multiple_choice_task(user_text):
            return False
        if int(self.execution_budget.local_tool_calls) <= 0:
            return False
        local_record_ids = {
            str(record.get("record_id", "") or "")
            for record in self.evidence_records
            if isinstance(record, dict) and str(record.get("tool", "") or "") == "search_local"
        }
        local_record_ids.discard("")
        if not local_record_ids:
            return False

        min_assessments = max(
            0,
            int(self.config.get("reflection_weak_mcq_local_retry_min_assessments", 1) or 1),
        )
        useful_assessments = 0
        for item in assessments:
            if not isinstance(item, dict):
                continue
            if str(item.get("record_id", "") or "") not in local_record_ids:
                continue
            if int(item.get("relevance", 0) or 0) >= 3 and int(item.get("patient_applicability", 0) or 0) >= 2:
                useful_assessments += 1
        if useful_assessments >= min_assessments:
            return False

        supported_local_ids: set[str] = set()
        for candidate in top_two:
            if not isinstance(candidate, dict):
                continue
            for value in list(candidate.get("supporting_evidence_ids") or []):
                text = str(value or "")
                if text in local_record_ids:
                    supported_local_ids.add(text)
        if supported_local_ids:
            return False
        return True

    def _generic_mcq_local_retry_query(
        self,
        user_text: str,
        *,
        top_two: List[Dict[str, Any]],
        missing_fact: str,
        retry_query: str,
    ) -> str:
        candidate_texts: List[str] = []
        for candidate in top_two[:2]:
            if not isinstance(candidate, dict):
                continue
            label = str(candidate.get("label") or "").strip()
            text = str(candidate.get("text") or "").strip()
            summary = str(candidate.get("case_summary") or "").strip()
            rendered = " | ".join(part for part in (label, text, summary) if part)
            if rendered:
                candidate_texts.append(rendered)
        choices_text = self._answer_choices_fragment(user_text)
        parts = [
            retry_query,
            missing_fact,
            "Visible question: " + re.sub(r"\s+", " ", str(user_text or "")).strip()[:1100],
            "Answer choices: " + choices_text if choices_text else "",
            plan_goal_context(self.current_plan) if bool(self.config.get("enable_planner", False)) else "",
            "Top candidates: " + " || ".join(candidate_texts) if candidate_texts else "",
            "Plan entities: " + ", ".join(plan_entity_names(self.current_plan)[:12]) if bool(self.config.get("enable_planner", False)) else "",
        ]
        return " | ".join(part for part in parts if str(part).strip())[:2200]

    def _generic_clinical_context_local_retry_query(
        self,
        user_text: str,
        *,
        top_two: List[Dict[str, Any]],
        missing_fact: str,
        retry_query: str,
    ) -> str:
        candidate_texts: List[str] = []
        for candidate in top_two[:2]:
            if not isinstance(candidate, dict):
                continue
            rendered = " | ".join(
                part
                for part in (
                    str(candidate.get("label") or "").strip(),
                    str(candidate.get("text") or "").strip(),
                    str(candidate.get("case_summary") or "").strip(),
                )
                if part
            )
            if rendered:
                candidate_texts.append(rendered)
        parts = [
            retry_query,
            missing_fact,
            "Visible clinical context: " + re.sub(r"\s+", " ", str(user_text or "")).strip()[:1300],
            "Answer choices: " + self._answer_choices_fragment(user_text) if self._answer_choices_fragment(user_text) else "",
            plan_goal_context(self.current_plan) if bool(self.config.get("enable_planner", False)) else "",
            "Top candidates: " + " || ".join(candidate_texts) if candidate_texts else "",
            "Plan entities: " + ", ".join(plan_entity_names(self.current_plan)[:12]) if bool(self.config.get("enable_planner", False)) else "",
        ]
        return " | ".join(part for part in parts if str(part).strip())[:2200]

    def _compact_retry_query_piece(self, text: str, *, limit: int = 760) -> str:
        compact = re.sub(r"\s+", " ", str(text or "")).strip(" |;\t\r\n")
        if len(compact) <= limit:
            return compact
        return compact[:limit].rsplit(" ", 1)[0].rstrip(" |;,.")

    def _reflection_retry_local_queries(
        self,
        user_text: str,
        reflection: Dict[str, Any],
    ) -> List[str]:
        retry = reflection.get("retry") if isinstance(reflection.get("retry"), dict) else {}
        retry_query = self._compact_retry_query_piece(str(retry.get("query", "") or ""))
        missing_fact = self._compact_retry_query_piece(str(reflection.get("missing_fact", "") or ""))
        generic_gap = bool(
            re.search(
                r"\bSource-backed (?:local )?evidence is needed\b",
                missing_fact,
                flags=re.IGNORECASE,
            )
        )
        goal = self._compact_retry_query_piece(
            plan_goal_context(self.current_plan) if bool(self.config.get("enable_planner", False)) else "",
            limit=520,
        )
        entities = plan_entity_names(self.current_plan)[:12] if bool(self.config.get("enable_planner", False)) else []
        top_two = [
            item for item in list(reflection.get("top_two_candidates") or [])[:2]
            if isinstance(item, dict)
        ]

        queries: List[str] = []
        if retry_query:
            queries.append(retry_query)
        elif missing_fact:
            queries.append(missing_fact)

        boundary_parts = []
        if missing_fact:
            boundary_parts.append("Missing decision fact: " + missing_fact)
        if goal:
            boundary_parts.append("Patient/decision goal: " + goal)
        if entities:
            boundary_parts.append("Key visible entities: " + ", ".join(entities))
        boundary_query = self._compact_retry_query_piece(" | ".join(boundary_parts))
        if boundary_query:
            queries.append(boundary_query)

        candidate_parts: List[str] = []
        for candidate in top_two:
            rendered = " | ".join(
                part
                for part in (
                    str(candidate.get("label") or "").strip(),
                    str(candidate.get("text") or "").strip(),
                    str(candidate.get("case_summary") or "").strip(),
                )
                if part
            )
            if rendered:
                candidate_parts.append(rendered)
        if candidate_parts and not generic_gap:
            queries.append(
                self._compact_retry_query_piece(
                    "Compare leading candidates and verify exact differentiating parameters: "
                    + " || ".join(candidate_parts),
                    limit=820,
                )
            )

        choices_text = self._answer_choices_fragment(user_text, limit=760)
        if choices_text and self._looks_like_multiple_choice_task(user_text):
            queries.append(
                self._compact_retry_query_piece(
                    "Answer choice disambiguation with source-backed boundaries: " + choices_text,
                    limit=820,
                )
            )

        case_boundary_query = self._compact_retry_query_piece(
            self._build_local_prefetch_query(user_text),
            limit=1000,
        )
        if case_boundary_query:
            queries.append(case_boundary_query)

        if not queries:
            queries.append(self._compact_retry_query_piece(user_text, limit=1000))

        seen: set[str] = set()
        deduped: List[str] = []
        for query in queries:
            cleaned = self._compact_retry_query_piece(query)
            key = re.sub(r"\W+", " ", cleaned.casefold()).strip()
            if not cleaned or key in seen:
                continue
            seen.add(key)
            deduped.append(cleaned)

        max_queries = max(1, int(self.config.get("reflection_retry_max_queries", 3) or 3))
        if not bool(self.config.get("reflection_retry_multiquery_enabled", True)):
            return deduped[:1]
        return deduped[:max_queries]

    def _execute_direct_local_retry(self, user_text: str, reflection: Dict[str, Any]) -> bool:
        if not bool(self.config.get("reflection_direct_local_retry", False)):
            return False
        retry = reflection.get("retry") if isinstance(reflection.get("retry"), dict) else {}
        retry_channel = str(retry.get("channel", "none") or "none").strip().lower()
        if retry_channel not in {"local", "either"}:
            return False
        if "search_local" not in self.allowed_tool_set:
            return False
        if not self._can_retry_local_evidence():
            return False

        queries = self._reflection_retry_local_queries(user_text, reflection)
        if not queries:
            return False
        tool_args: Dict[str, Any] = {
            "query": queries if len(queries) > 1 else queries[0],
            "choices": self._current_example_choices(),
            "goal": plan_goal_context(self.current_plan) if bool(self.config.get("enable_planner", False)) else "",
            "entities": plan_entity_names(self.current_plan)[:20] if bool(self.config.get("enable_planner", False)) else [],
            "require_rag": True,
            "dense_k": max(1, int(self.config.get("reflection_retry_dense_k", 16) or 16)),
            "bm25_k": max(1, int(self.config.get("reflection_retry_bm25_k", 20) or 20)),
            "max_evidence": max(1, int(self.config.get("reflection_retry_max_evidence", 12) or 12)),
        }
        self.memory.add_assistant(render_tool_call_block([("search_local", tool_args)]))
        executed = self._execute_tool_calls(
            [("search_local", tool_args)],
            max_tools_this_turn=1,
            phase_label="REFLECTION DIRECT LOCAL RETRY",
        )
        if executed:
            self.execution_budget.record_tool_turn()
            self._log(
                "[REFLECTION DIRECT LOCAL RETRY | SUMMARY]\n"
                + json.dumps(
                    {
                        "query_count": len(queries),
                        "dense_k": tool_args["dense_k"],
                        "bm25_k": tool_args["bm25_k"],
                        "max_evidence": tool_args["max_evidence"],
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
        return bool(executed)

    def _answer_after_direct_retry(self) -> str:
        self.memory.set_system_prompt(self.answer_only_system_prompt)
        final_text = self._chat(
            self.memory.get_messages(),
            max_tokens=max(300, int(self.config.get("final_max_tokens", 3000) or 3000)),
        )
        self._log("[REFLECTION DIRECT LOCAL RETRY | ANSWER]\n" + (final_text or ""))
        return final_text

    def _expects_short_final_answer(self, user_text: str) -> bool:
        if getattr(self, "config", {}).get("answer_type") == "open":
            return "FINAL_ANSWER" in user_text
        answer_type = self._current_example_answer_type()
        if answer_type in {"multipleChoice", "multi_choice", "open_ended_multi_choice"}:
            return False
        text = str(user_text or "")
        if "FINAL_ANSWER" not in text:
            return False
        if re.search(r"\b(?:Answer\s+Choices?|Choices)\s*:", text, flags=re.IGNORECASE):
            return False
        if re.search(r"(?m)^\s*[A-Z][\.\)]\s+\S+", text):
            return False
        return True

    def _extract_final_answer_line(self, text: str) -> str:
        extracted = ""
        for line in str(text or "").splitlines():
            match = _FINAL_ANSWER_LINE_RE.match(line)
            if match:
                extracted = match.group(1).strip()
        return extracted or str(text or "").strip()

    def _replace_final_answer_line(self, text: str, final_answer: str) -> str:
        answer = re.sub(r"\s+", " ", str(final_answer or "")).strip(" \t\r\n.;")
        if not answer:
            return str(text or "")
        lines = str(text or "").splitlines()
        for index in range(len(lines) - 1, -1, -1):
            if _FINAL_ANSWER_LINE_RE.match(lines[index]):
                lines[index] = f"FINAL_ANSWER: {answer}"
                return "\n".join(lines)
        base = str(text or "").rstrip()
        if not base:
            return f"FINAL_ANSWER: {answer}"
        return base + f"\nFINAL_ANSWER: {answer}"

    def _extract_mcq_final_label(self, user_text: str, answer_text: str) -> str:
        choices = self._choice_map_for_reflection(user_text)
        if not choices:
            return ""
        final_line = self._extract_final_answer_line(answer_text)
        match = re.search(r"\b([A-Z])\b", final_line)
        label = match.group(1).upper() if match else ""
        return label if label in choices else ""

    def _strong_revision_support_ids(self, evidence_ids: List[str]) -> List[str]:
        by_id = {
            str(record.get("record_id") or ""): record
            for record in self.evidence_records
            if isinstance(record, dict)
        }
        strong: List[str] = []
        for record_id in evidence_ids:
            record = by_id.get(str(record_id or ""))
            if not record:
                continue
            if str(record.get("tool") or "") == "web_search" and int(record.get("authority_tier", 4) or 4) >= 3:
                continue
            if self._record_is_local_atom(record):
                continue
            strong.append(str(record_id))
        return strong

    def _auto_revision_from_reflection(
        self,
        *,
        user_text: str,
        draft_answer: str,
        top_two: List[Dict[str, Any]],
        preferred_candidate: str,
        confidence: float,
    ) -> str:
        if confidence < float(self.config.get("reflection_auto_revision_confidence_threshold", 0.86) or 0.86):
            return ""
        draft_label = self._extract_mcq_final_label(user_text, draft_answer)
        preferred_label = str(preferred_candidate or "").strip().upper()
        choices = self._choice_map_for_reflection(user_text)
        if not draft_label or preferred_label not in choices or preferred_label == draft_label:
            return ""
        preferred = None
        draft_candidate = None
        for candidate in top_two:
            label = str(candidate.get("label") or "").strip().upper()
            if label == preferred_label:
                preferred = candidate
            if label == draft_label:
                draft_candidate = candidate
        if not preferred or not draft_candidate:
            return ""
        strong_support = self._strong_revision_support_ids(list(preferred.get("supporting_evidence_ids") or []))
        draft_contradictions = [
            str(item)
            for item in list(draft_candidate.get("contradicting_evidence_ids") or [])
            if str(item)
        ]
        if not strong_support or not draft_contradictions:
            return ""
        revised = self._replace_final_answer_line(draft_answer, preferred_label)
        return revised if revised != draft_answer else ""

    _EXTRA_PARAMETER_RE = re.compile(
        r"\b("
        r"add|additional|adjunct|combination|combined|dual|concomitant|"
        r"monitor|monitoring|recheck|repeat|level|laboratory|lab|"
        r"increase|decrease|reduce|escalate|hold|stop|discontinue|restart|"
        r"delay|defer|until|before|after|daily|twice|q\d+h|dose|route|iv|po"
        r")\b",
        flags=re.IGNORECASE,
    )
    _NEGATED_EXTRA_PARAMETER_RE = re.compile(
        r"\b(?:no|without|not|avoid|avoiding)\s+(?:additional\s+)?[^.;,\n]{0,90}"
        r"\b(?:monitor|monitoring|recheck|repeat|level|laboratory|lab|"
        r"treatment|therapy|dose|route|iv|po)\b",
        flags=re.IGNORECASE,
    )

    @classmethod
    def _extra_parameter_terms(cls, text: str) -> set[str]:
        normalized = cls._NEGATED_EXTRA_PARAMETER_RE.sub(" ", str(text or ""))
        terms = {match.group(1).lower() for match in cls._EXTRA_PARAMETER_RE.finditer(normalized)}
        aliases = {
            "monitoring": "monitor",
            "laboratory": "lab",
            "additional": "add",
            "combined": "combination",
            "dual": "combination",
            "concomitant": "combination",
            "decrease": "reduce",
            "defer": "delay",
        }
        return {aliases.get(term, term) for term in terms}

    def _extra_parameter_gap_for_reflection(
        self,
        *,
        top_two: List[Dict[str, Any]],
        preferred_candidate: str,
    ) -> bool:
        if len(top_two) < 2:
            return False

        preferred_label = str(preferred_candidate or "").strip().upper()
        preferred = None
        runner = None
        for item in top_two:
            label = str(item.get("label") or "").strip().upper()
            if preferred_label and label == preferred_label:
                preferred = item
            elif runner is None:
                runner = item
        if preferred is None:
            preferred = top_two[0]
            runner = top_two[1]
        elif runner is None:
            runner = next((item for item in top_two if item is not preferred), None)
        if runner is None:
            return False

        preferred_text = str(preferred.get("text") or "")
        runner_text = str(runner.get("text") or "")
        preferred_terms = self._extra_parameter_terms(preferred_text)
        runner_terms = self._extra_parameter_terms(runner_text)
        if not (preferred_terms - runner_terms):
            return False

        preferred_support = {
            str(value)
            for value in list(preferred.get("supporting_evidence_ids") or [])
            if str(value).strip()
        }
        runner_support = {
            str(value)
            for value in list(runner.get("supporting_evidence_ids") or [])
            if str(value).strip()
        }
        # If the more intensive candidate has no unique cited support, let a
        # bounded evidence retry decide the extra parameter instead of allowing
        # generic "more monitoring is safer" reasoning to settle the answer.
        return not bool(preferred_support - runner_support)

    def _short_answer_needs_compaction(self, final_answer: str) -> bool:
        value = str(final_answer or "").strip()
        if not value:
            return False
        if self._short_answer_has_regimen_parameters(value):
            return False
        if len(value) > int(self.config.get("short_answer_compaction_min_chars", 120) or 120):
            return True
        return bool(
            re.search(
                r"\b(?:also|for example|e\.g\.|alternatives?|secondary)\b|[;(]|\s+or\s+",
                value,
                flags=re.IGNORECASE,
            )
        )

    def _short_answer_has_regimen_parameters(self, value: str) -> bool:
        text = str(value or "")
        return bool(
            re.search(
                r"\b(?:mg|mcg|g|kg|mL|IU|units?|q\d+h|q\s*\d+\s*h|every\s+\d+\s+(?:hours?|days?)|"
                r"once\s+daily|twice\s+daily|IV|PO|oral(?:ly)?|intravenous(?:ly)?|infus(?:e|ion)|"
                r"hold|restart|resume|recheck|trough|target|threshold|CrCl|renal(?:ly)?\s+dosed|"
                r"without\s+(?:a\s+)?loading|no\s+loading|monotherapy)\b",
                text,
                flags=re.IGNORECASE,
            )
        )

    def _clean_short_answer_phrase(self, phrase: str) -> str:
        cleaned = str(phrase or "").strip()
        cleaned = re.sub(r"^[\s:,-]*(?:a|an|the)\s+", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s+", " ", cleaned)
        cleaned = cleaned.strip(" \t\r\n.;,()[]")
        return cleaned

    def _looks_like_compact_answer(self, phrase: str) -> bool:
        cleaned = self._clean_short_answer_phrase(phrase)
        if not cleaned:
            return False
        words = re.findall(r"[A-Za-z0-9][A-Za-z0-9+/-]*", cleaned)
        return 1 <= len(words) <= 10 and len(cleaned) <= 80

    def _action_prefix_for_short_answer(self, context: str, phrase: str) -> str:
        phrase = self._clean_short_answer_phrase(phrase)
        if not phrase:
            return ""
        phrase_l = phrase.lower()
        if phrase_l.startswith(
            (
                "switch ",
                "change ",
                "avoid ",
                "stop ",
                "hold ",
                "discontinue ",
                "administer ",
                "give ",
                "use ",
                "select ",
                "choose ",
            )
        ):
            return phrase
        context_l = context.lower()
        if any(
            term in context_l
            for term in (
                "switch",
                "alternative",
                "replace",
                "contraindicat",
                "stop taking",
            )
        ):
            return f"Switch to {phrase}"
        if any(term in context_l for term in ("avoid", "do not use", "must not")):
            return f"Avoid {phrase}"
        if any(term in context_l for term in ("administer", "give", "dose")):
            return f"Administer {phrase}"
        return phrase

    def _deterministic_short_answer(self, user_text: str, response_text: str) -> str:
        final_answer = self._extract_final_answer_line(response_text)
        if not self._short_answer_needs_compaction(final_answer):
            return ""

        text = re.sub(r"\s+", " ", str(final_answer or "")).strip()
        first_clause = re.split(
            r"\bAlso\b|;\s*also\b|\. Also\b",
            text,
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0].strip()

        example_match = re.search(
            r"(?:e\.g\.|for example),?\s*([^).;]+)",
            first_clause,
            flags=re.IGNORECASE,
        )
        if example_match:
            first_option = re.split(
                r"\s+or\s+|,|/|;|\band\b",
                example_match.group(1).strip(),
                maxsplit=1,
                flags=re.IGNORECASE,
            )[0]
            first_option = self._clean_short_answer_phrase(first_option)
            if self._looks_like_compact_answer(first_option):
                return self._action_prefix_for_short_answer(first_clause, first_option)

        action_match = re.search(
            r"\b(?:should\s+)?(switch to|change to|avoid|stop|hold|discontinue|administer|give|use|recommend)\s+([^.;()]+)",
            first_clause,
            flags=re.IGNORECASE,
        )
        if action_match:
            verb = action_match.group(1).strip()
            obj = self._clean_short_answer_phrase(action_match.group(2))
            obj = re.split(r"\s+or\s+|, and |,|\band also\b", obj, maxsplit=1, flags=re.IGNORECASE)[0]
            obj = self._clean_short_answer_phrase(obj)
            if self._looks_like_compact_answer(obj):
                return self._clean_short_answer_phrase(f"{verb} {obj}")

        numeric_match = re.search(
            r"\b(\d+(?:\.\d+)?)\s*(hours?|minutes?|days?|weeks?|months?|years?)\b",
            first_clause,
            flags=re.IGNORECASE,
        )
        question_l = str(user_text or "").lower()
        if numeric_match and any(term in question_l for term in ("how long", "time", "duration", "beyond-use", "within")):
            return self._clean_short_answer_phrase(" ".join(numeric_match.groups()))

        return ""

    def _maybe_compact_final_short_answer(self, user_text: str, final_answer: str) -> str:
        if not bool(
            self.config.get(
                "compact_short_final_answer",
                True,
            )
        ):
            return final_answer
        if not self._expects_short_final_answer(user_text):
            return final_answer
        compact = self._deterministic_short_answer(user_text, final_answer)
        if not compact:
            return final_answer
        rewritten = self._replace_final_answer_line(final_answer, compact)
        if rewritten != final_answer:
            self._log(
                "[FINAL FORMAT | SHORT ANSWER COMPACTION]\n"
                + json.dumps(
                    {
                        "original_final_answer": self._extract_final_answer_line(final_answer)[:500],
                        "compacted_final_answer": compact,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
        return rewritten

    def _memory_context(self, *, include_tool_payload: bool = False) -> List[Dict[str, Any]]:
        """Read context from both the unified and pre-v2 Memory contracts."""
        getter = self.memory.get_messages_context
        if include_tool_payload:
            try:
                return getter(include_tool_payload=True)
            except TypeError:
                # Older runtimes expose get_messages_context() without the
                # metadata flag.  Tool execution still only needs the visible
                # conversation, so falling back is lossless for that runtime.
                return getter()
        return getter()

    def _enrich_tool_args(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        enriched = dict(args)
        if not bool(self.config.get("enable_planner", False)):
            if tool_name == "search_local":
                dense_k = max(1, int(self.config.get("unplanned_search_local_dense_k", 4) or 4))
                bm25_k = max(1, int(self.config.get("unplanned_search_local_bm25_k", 4) or 4))
                max_evidence = max(1, int(self.config.get("unplanned_search_local_max_evidence", 4) or 4))
                enriched["dense_k"] = min(max(1, int(enriched.get("dense_k", dense_k) or dense_k)), dense_k)
                enriched["bm25_k"] = min(max(1, int(enriched.get("bm25_k", bm25_k) or bm25_k)), bm25_k)
                enriched["max_evidence"] = min(
                    max(1, int(enriched.get("max_evidence", max_evidence) or max_evidence)),
                    max_evidence,
                )
            return enriched
        goal = plan_goal_context(self.current_plan)
        if not goal:
            return enriched

        if tool_name == "web_visit":
            existing_goal = str(enriched.get("goal", "") or "").strip()
            if existing_goal and goal not in existing_goal:
                enriched["goal"] = f"{existing_goal} | Patient/decision goal: {goal}"
            elif not existing_goal:
                enriched["goal"] = goal
        elif tool_name == "search_local":
            enriched.setdefault("goal", self._planner_local_goal_context() or goal)
            if not enriched.get("entities"):
                enriched["entities"] = self._planner_local_entity_names(max_items=20)
        elif tool_name == "web_search" and bool(self.config.get("enforce_goal_propagation", True)):
            suffix = " patient goal " + goal[:320]
            query = enriched.get("query")
            if isinstance(query, list):
                enriched["query"] = [
                    item if goal[:80].lower() in str(item).lower() else f"{str(item).strip()}{suffix}"
                    for item in query[:3]
                    if str(item).strip()
                ]
            elif query and goal[:80].lower() not in str(query).lower():
                enriched["query"] = f"{str(query).strip()}{suffix}"
        return enriched

    def _execute_tool_calls(
        self,
        tool_calls: List[Tuple[str, Dict[str, Any]]],
        *,
        max_tools_this_turn: int,
        phase_label: str,
    ) -> int:
        if not bool(self.config.get("enable_tools", True)):
            err = self._error_result("TOOLS_DISABLED", "Tools are disabled by config.")
            self._record_rejected_tool_call("TOOLS_DISABLED", "tools_disabled")
            self._log(f"[{phase_label} | TOOL | RESULT | TOOLS_DISABLED]\n" + json.dumps(err, ensure_ascii=False, indent=2))
            self.memory.add_tool_result("TOOLS_DISABLED", err)
            return 0

        if _is_unbounded_limit(int(max_tools_this_turn)):
            exec_calls = list(tool_calls)
        else:
            exec_calls = tool_calls[: max(0, int(max_tools_this_turn))]
        overflow = len(tool_calls) - len(exec_calls)

        for tool_name, args in exec_calls:
            allowed, budget_reason = self.execution_budget.can_call_tool(tool_name)
            if not allowed:
                err = self._error_result(tool_name, f"Execution budget blocked tool call: {budget_reason}")
                self._record_rejected_tool_call(tool_name, budget_reason or "budget_blocked")
                self._log(f"[{phase_label} | TOOL | RESULT | {tool_name}]\n" + json.dumps(err, ensure_ascii=False, indent=2))
                self.memory.add_tool_result(tool_name, err)
                continue
            args = self._enrich_tool_args(tool_name, args)
            self.execution_budget.record_tool_call(tool_name)
            self._log(f"[{phase_label} | TOOL | CALL]")
            self._log(f"name: {tool_name}")
            self._log("args:")
            self._log(json.dumps(args, ensure_ascii=False, indent=2))

            if tool_name not in self.allowed_tool_set:
                err = self._error_result(tool_name, f"Illegal tool name: {tool_name}")
                self._log(f"[{phase_label} | TOOL | RESULT | {tool_name}]\n" + json.dumps(err, ensure_ascii=False, indent=2))
                self.memory.add_tool_result(tool_name, err)
                continue

            try:
                context = self._tool_context()
                execute_kwargs = {
                    "enabled_tools": self.enabled_tools,
                    "disabled_tools": self.disabled_tools,
                }
                if self.strict_tool_registry:
                    execute_kwargs["strict"] = True
                tool_result = execute_tool(tool_name, args, context, **execute_kwargs)
            except Exception as exc:
                tool_result = self._error_result(tool_name, f"Tool execution raised: {exc}")

            evidence_records = normalize_tool_evidence(
                tool_name,
                tool_result,
                goal=plan_goal_context(self.current_plan) if bool(self.config.get("enable_planner", False)) else "",
            )
            if evidence_records:
                known_ids = {str(item.get("record_id", "")) for item in self.evidence_records}
                for record in evidence_records:
                    if str(record.get("record_id", "")) not in known_ids:
                        self.evidence_records.append(record)
                        known_ids.add(str(record.get("record_id", "")))
                tool_result = dict(tool_result)
                tool_result["evidence_records"] = evidence_records

            self._log(f"[{phase_label} | TOOL | RESULT | {tool_name}]\n" + json.dumps(tool_result, ensure_ascii=False, indent=2))
            self.memory.add_tool_result(tool_name, tool_result)

        if overflow > 0 and not _is_unbounded_limit(int(max_tools_this_turn)):
            err = self._error_result(
                "MAX_TOOLS_PER_TURN",
                f"Requested {len(tool_calls)} tools in one turn; executed only the first {len(exec_calls)}.",
            )
            self._log(f"[{phase_label} | TOOL | RESULT | MAX_TOOLS_PER_TURN]\n" + json.dumps(err, ensure_ascii=False, indent=2))
            self.memory.add_tool_result("MAX_TOOLS_PER_TURN", err)
            for _ in range(overflow):
                self._record_rejected_tool_call(
                    "MAX_TOOLS_PER_TURN",
                    "per_turn_tool_limit_exceeded",
                )

        retain_k = int(self.config.get("retain_tool_pairs_k", -1))
        self.memory.compact(retain_tool_pairs_k=retain_k)
        return len(exec_calls)

    def _screen_plan_relations(self, user_text: str) -> Dict[str, Any]:
        raw_attempts: List[str] = []
        error = ""
        try:
            request = {
                "visible_case": user_text,
                "decision_goal": self.candidate_plan.get("decision_goal", ""),
                "patient_goal": self.candidate_plan.get("patient_goal", ""),
                "patient_anchors": self.candidate_plan.get("patient_anchors", []),
                "constraints": self.candidate_plan.get("constraints", []),
                "require_evidence": bool(self.config.get("screening_require_evidence", False)),
                **screening_candidates(self.candidate_plan),
            }
            result = None
            retries = max(0, int(self.config.get("screening_max_retries", 1)))
            for attempt in range(retries + 1):
                try:
                    feedback = f"\nPrevious response was invalid: {error}" if error else ""
                    raw = self._chat(
                        [
                            {"role": "system", "content": RELATION_SCREENING_SYSTEM_PROMPT},
                            {"role": "user", "content": json.dumps(request, ensure_ascii=False) + feedback},
                        ],
                        json_mode=True,
                        max_tokens=max(400, int(self.config.get("screening_max_tokens", 2400))),
                        json_thinking_enabled=bool(self.config.get("structured_json_thinking_enabled", False)),
                    )
                    raw_attempts.append(f"[ATTEMPT {attempt + 1}]\n{raw}")
                    result = normalize_relation_screening(
                        extract_json_object(raw, required_keys=("selected_relations", "evidence_requirements")),
                        self.candidate_plan,
                    )
                    if request["require_evidence"] and not result["evidence_requirements"]:
                        result = None
                        raise ValueError("Evidence verification requires at least one requirement; a visible root may use relation_ids=[].")
                    break
                except Exception as exc:
                    error = str(exc)
                    raw_attempts.append(f"[ATTEMPT {attempt + 1} ERROR] {error}")
                    allowed, _ = self.execution_budget.can_call_llm(self._current_total_tokens())
                    if not allowed:
                        break
            if result is None:
                result = screening_failure_plan(self.candidate_plan, error)
        except Exception as exc:
            result = screening_failure_plan(self.candidate_plan, str(exc))
        self.relation_screening_raw = "\n\n".join(raw_attempts)
        self._log("[RELATION SCREENING | RAW]\n" + self.relation_screening_raw)
        self._log("[RELATION SCREENING | RESULT]\n" + json.dumps(result, ensure_ascii=False, indent=2))
        return result

    def _run_planner(self, user_text: str) -> Dict[str, Any]:
        self.candidate_plan = {}
        self.relation_screening_raw = ""
        if not bool(self.config.get("enable_planner", True)):
            self.current_plan = disabled_plan()
            return self.current_plan

        planner_messages = [
            {"role": "system", "content": self.planner_system_prompt},
            {"role": "user", "content": user_text},
        ]
        raw_attempts: List[str] = []
        base_max_tokens = max(400, int(self.config.get("planner_max_tokens", 1800) or 1800))
        retry_max_tokens = max(
            base_max_tokens,
            int(self.config.get("planner_retry_max_tokens", max(base_max_tokens * 2, 7000)) or 7000),
        )
        max_retries = max(0, int(self.config.get("planner_max_retries", 1) or 0))
        token_schedule = [base_max_tokens] + [retry_max_tokens] * max_retries
        self.current_plan = fallback_plan(user_text, "")
        for attempt_index, max_tokens in enumerate(token_schedule, start=1):
            try:
                messages = list(planner_messages)
                if attempt_index > 1:
                    messages = [
                        {
                            "role": "system",
                            "content": (
                                self.planner_system_prompt
                                + "\n\nRetry instruction: return one complete compact JSON object only. "
                                "Do not include explanations, markdown, or partial nested objects."
                            ),
                        },
                        {"role": "user", "content": user_text},
                    ]
                planner_raw = self._chat(
                    messages,
                    json_mode=True,
                    max_tokens=max_tokens,
                    json_thinking_enabled=bool(
                        self.config.get("structured_json_thinking_enabled", False)
                    ),
                )
                raw_attempts.append(planner_raw or "")
                candidate_plan = normalize_plan(
                    extract_json_object(
                        planner_raw,
                        required_keys=("decision_goal", "patient_goal"),
                    ),
                    user_text=user_text,
                )
                if str(candidate_plan.get("status", "") or "").strip().lower() != "fallback":
                    self.current_plan = candidate_plan
                    break
                self.current_plan = candidate_plan
            except Exception as exc:
                raw_attempts.append(f"[planner_attempt_{attempt_index}_error] {exc}")
                self.current_plan = fallback_plan(user_text, str(exc))
        self.planner_raw = "\n\n".join(
            f"[ATTEMPT {index}]\n{raw}" for index, raw in enumerate(raw_attempts, start=1)
        )
        if (
            str(self.current_plan.get("status", "") or "").strip().lower() == "fallback"
            and bool(self.config.get("enable_visible_planner_fallback", True))
        ):
            self.current_plan = visible_fallback_plan(
                user_text,
                error=str(self.current_plan.get("planner_error", "") or ""),
            )
        self._log("[PLANNER | RAW]\n" + (self.planner_raw or ""))
        self.candidate_plan = json.loads(json.dumps(self.current_plan, ensure_ascii=False))
        self._log("[PLANNER | CANDIDATE TREE]\n" + json.dumps(self.candidate_plan, ensure_ascii=False, indent=2))
        self.current_plan = self._screen_plan_relations(user_text)
        self._log("[PLANNER | NORMALIZED PLAN]\n" + json.dumps(self.current_plan, ensure_ascii=False, indent=2))
        self._refresh_answer_prompt()
        return self.current_plan

    def _run_main_agent(self) -> str:
        max_rounds = max(1, int(self.config.get("max_rounds", 20) or 20))
        raw_max_tools_per_turn = int(self.config.get("max_tools_per_turn", 3))
        max_tools_per_turn = -1 if raw_max_tools_per_turn < 0 else max(1, raw_max_tools_per_turn)
        rounds = 0

        while rounds < max_rounds:
            self._refresh_available_tools()
            round_started = time.perf_counter()
            rounds += 1
            if rounds > 1:
                self._log("")
            remaining_tool_turns = self.execution_budget.remaining_tool_turns
            if remaining_tool_turns == 0:
                self.memory.set_system_prompt(self.answer_only_system_prompt)
            else:
                self.memory.set_system_prompt(
                    self._build_main_prompt(
                        enabled_tools=self.allowed_tools,
                        disabled_tools=self.disabled_tools,
                        remaining_tool_turns=remaining_tool_turns,
                        max_tools_per_turn=max_tools_per_turn,
                        plan_context=self._active_plan_context(),
                    )
                )
            assistant_raw = self._chat(
                self.memory.get_messages(),
                max_tokens=max(400, int(self.config.get("main_max_tokens", 5000) or 5000)),
            )
            self._log(f"[MAIN | ROUND {rounds} | RAW]\n" + (assistant_raw or ""))

            if self._is_tool_call(assistant_raw):
                can_start_turn, budget_reason = self.execution_budget.can_start_tool_turn()
                if not can_start_turn:
                    self.memory.set_system_prompt(self.answer_only_system_prompt)
                    err = self._error_result("MAX_TOOL_TURNS", f"Tool budget exhausted ({budget_reason}). According to the available context, provide the final answer now without calling tools.")
                    self._record_rejected_tool_call(
                        "MAX_TOOL_TURNS",
                        budget_reason or "tool_turn_limit_exceeded",
                    )
                    self._log("[MAIN | TOOL | RESULT | MAX_TOOL_TURNS]\n" + json.dumps(err, ensure_ascii=False, indent=2))
                    self.memory.add_tool_result("MAX_TOOL_TURNS", err)
                    final_text = self._chat(
                        self.memory.get_messages(),
                        max_tokens=max(300, int(self.config.get("final_max_tokens", 3000) or 3000)),
                    )
                    self._log("[MAIN | FINAL AFTER LIMIT]\n" + (final_text or ""))
                    recorder = get_active_timing_recorder()
                    if recorder is not None:
                        recorder.record_agent_loop(
                            loop_index=rounds,
                            elapsed_seconds=time.perf_counter() - round_started,
                            used_tools=False,
                            metadata={"forced_final_after_tool_turn_limit": True},
                        )
                    return final_text

                tool_calls = self._extract_all_tool_calls(assistant_raw)
                if not tool_calls:
                    err = self._error_result("UNKNOWN_TOOL", "No valid tool call found.")
                    self.memory.add_assistant(str(assistant_raw or "").strip())
                    self.execution_budget.record_tool_call("UNKNOWN_TOOL")
                    self.execution_budget.record_tool_turn()
                    self._record_rejected_tool_call(
                        "UNKNOWN_TOOL",
                        "unregistered_disabled_or_malformed_tool_request",
                    )
                    self._log("[MAIN | TOOL | RESULT | UNKNOWN_TOOL]\n" + json.dumps(err, ensure_ascii=False, indent=2))
                    self.memory.add_tool_result("UNKNOWN_TOOL", err)
                    recorder = get_active_timing_recorder()
                    if recorder is not None:
                        recorder.record_agent_loop(
                            loop_index=rounds,
                            elapsed_seconds=time.perf_counter() - round_started,
                            used_tools=True,
                            metadata={"invalid_tool_call": True},
                        )
                    continue
                self.memory.add_assistant(self._normalize_tool_call_message(assistant_raw, tool_calls))

                self._execute_tool_calls(
                    tool_calls,
                    max_tools_this_turn=max_tools_per_turn,
                    phase_label=f"MAIN TURN {self.execution_budget.tool_turns + 1}",
                )
                self.execution_budget.record_tool_turn()
                recorder = get_active_timing_recorder()
                if recorder is not None:
                    recorder.record_agent_loop(
                        loop_index=rounds,
                        elapsed_seconds=time.perf_counter() - round_started,
                        used_tools=True,
                        metadata={
                            "executed_tool_count": len(tool_calls) if max_tools_per_turn < 0 else min(len(tool_calls), max_tools_per_turn),
                            "tool_turn_index": self.execution_budget.tool_turns,
                        },
                    )
                if self.execution_budget.remaining_tool_turns == 0:
                    self.memory.set_system_prompt(self.answer_only_system_prompt)
                continue

            recorder = get_active_timing_recorder()
            if recorder is not None:
                recorder.record_agent_loop(
                    loop_index=rounds,
                    elapsed_seconds=time.perf_counter() - round_started,
                    used_tools=False,
                    metadata={"completed_with_final_answer": True},
                )
            return assistant_raw

        self.memory.set_system_prompt(self.answer_only_system_prompt)
        fallback = self._chat(
            self.memory.get_messages(),
            max_tokens=max(300, int(self.config.get("final_max_tokens", 3000) or 3000)),
        )
        self._log("[MAIN | FINAL AFTER MAX_ROUNDS]\n" + (fallback or ""))
        return fallback

    def _render_tool_payload(self, tool_name: str, payload: Dict[str, Any]) -> str:
        rendered, _ = bound_model_visible_text(
            render_tool_result_body(tool_name, payload),
            max_chars=int(
                self.config.get("max_model_visible_tool_result_chars", 12000) or 12000
            ),
        )
        return rendered

    def _evidence_content_for_reflection(self, record: Dict[str, Any]) -> str:
        # Bound the number of records, not the contents of a selected source unit.
        return str(record.get("quote") or record.get("fact") or "").strip()

    def _record_is_local_atom(self, record: Dict[str, Any]) -> bool:
        haystack = " ".join(
            str(record.get(key) or "")
            for key in ("url", "title", "quote", "fact", "entity", "source_type")
        ).casefold()
        return (
            "local://support-atom/" in haystack
            or "support atom" in haystack
            or "decision-support atom" in haystack
            or "differentiator atom" in haystack
            or "parameter atom" in haystack
        )

    def _record_has_visible_anchor_overlap(self, record: Dict[str, Any]) -> bool:
        current_example = self.current_example_payload if isinstance(self.current_example_payload, dict) else {}
        visible_parts = [
            str(current_example.get("question") or ""),
            str(current_example.get("context") or ""),
            " ".join(str(value) for value in (current_example.get("choices") or {}).values())
            if isinstance(current_example.get("choices"), dict)
            else str(current_example.get("choices") or ""),
            plan_goal_context(self.current_plan) if bool(self.config.get("enable_planner", False)) else "",
            " ".join(plan_entity_names(self.current_plan)[:20]) if bool(self.config.get("enable_planner", False)) else "",
        ]
        visible = re.sub(r"\s+", " ", " ".join(part for part in visible_parts if part)).casefold()
        if not visible:
            return False
        content = " ".join(
            str(record.get(key) or "")
            for key in ("quote", "fact", "title", "entity")
        ).casefold()
        tokens = [
            token
            for token in re.findall(r"[a-z0-9]+", content)
            if len(token) >= 4
            and not token.isdigit()
            and token
            not in {
                "answer",
                "atom",
                "case",
                "choice",
                "clinical",
                "context",
                "decision",
                "disease",
                "dose",
                "drug",
                "evidence",
                "final",
                "local",
                "management",
                "medical",
                "patient",
                "question",
                "record",
                "source",
                "support",
                "therapy",
                "treatment",
            }
        ]
        overlap = {token for token in tokens if token in visible}
        return len(overlap) >= 2 or any(len(token) >= 8 for token in overlap)

    def _reflection_record_priority(self, record: Dict[str, Any], recency_rank: int) -> tuple[int, int]:
        content = " ".join(
            str(record.get(key) or "")
            for key in ("quote", "fact", "title", "entity", "goal", "source_type")
        )
        content_l = content.lower()
        score = max(0, min(60, int(recency_rank)))
        try:
            authority_tier = int(record.get("authority_tier", 4) or 4)
        except Exception:
            authority_tier = 4
        if authority_tier <= 1:
            score += 45
        elif authority_tier == 2:
            score += 28
        elif authority_tier == 3:
            score += 8
        is_local = str(record.get("tool") or "") == "search_local"
        has_visible_overlap = self._record_has_visible_anchor_overlap(record)
        is_local_atom = self._record_is_local_atom(record)
        if is_local:
            score += 28 if has_visible_overlap else 8
        elif str(record.get("tool") or "") == "web_visit":
            score += 20
        url_l = str(record.get("url") or "").lower()
        if is_local_atom and not has_visible_overlap:
            score -= 30
        elif is_local_atom and has_visible_overlap:
            score += 4
        elif "local://support-atom/" in url_l:
            score += 2
        if _CRITICAL_EVIDENCE_RE.search(content):
            score += 40
        source_type = str(record.get("source_type") or "").lower()
        if any(term in source_type for term in ("guideline", "regulator", "official", "label", "public_health")):
            score += 20
        for entity in plan_entity_names(self.current_plan)[:12]:
            entity_l = str(entity or "").strip().lower()
            if entity_l and entity_l in content_l:
                score += 4
        return score, recency_rank

    def _reflection_evidence_ledger(self, *, compact: bool = False) -> List[Dict[str, Any]]:
        records = list(self.evidence_records[-60:])
        if not records:
            return []
        limit = int(
            self.config.get(
                "reflection_max_evidence_records_compact" if compact else "reflection_max_evidence_records",
                10 if compact else 16,
            )
            or (10 if compact else 16)
        )
        scored: List[tuple[tuple[int, int], int, Dict[str, Any]]] = []
        for index, record in enumerate(records):
            if not isinstance(record, dict):
                continue
            recency_rank = index + 1
            scored.append((self._reflection_record_priority(record, recency_rank), index, record))
        scored.sort(key=lambda item: item[0], reverse=True)
        selected_records = [record for _, _, record in scored[: max(1, limit)]]

        compact_records: List[Dict[str, Any]] = []
        for record in selected_records:
            compact_records.append(
                {
                    "record_id": str(record.get("record_id", "") or ""),
                    "tool": str(record.get("tool", "") or ""),
                    "title": str(record.get("title", "") or "")[:180],
                    "url": str(record.get("url", "") or "")[:500],
                    "source_type": str(record.get("source_type", "") or "")[:80],
                    "authority_tier": record.get("authority_tier"),
                    "content_kind": str(record.get("content_kind", "") or ""),
                    "entity": str(record.get("entity", "") or "")[:120],
                    "goal": str(record.get("goal", "") or "")[:220],
                    "content": self._evidence_content_for_reflection(record),
                }
            )
        return compact_records

    def _reflection_safety_flags_text(self, *, compact: bool = False) -> str:
        flagged: List[Dict[str, Any]] = []
        for record in self._reflection_evidence_ledger(compact=compact):
            content = " ".join(
                str(record.get(key) or "")
                for key in ("content", "title", "entity", "source_type")
            )
            content_l = content.lower()
            if _CRITICAL_EVIDENCE_RE.search(content) and "support atom" not in content_l:
                flagged.append(record)
            if len(flagged) >= (4 if compact else 6):
                break
        if not flagged:
            return ""
        lines = [
            "Mandatory patient-specific safety flags:",
            "You must explicitly assess these record_id values and compare them against the draft answer's main rationale.",
        ]
        for record in flagged:
            lines.append(
                "- "
                + str(record.get("record_id", ""))
                + " | "
                + str(record.get("title", ""))[:120]
                + " | "
                + str(record.get("content", ""))[:360]
            )
        return "\n".join(lines)

    def _build_reflection_prompt(self, user_text: str, draft_answer: str, *, compact: bool = False) -> str:
        reasoning_steps: List[str] = []
        evidence_blocks: List[str] = []
        evidence_ledger = self._reflection_evidence_ledger(compact=compact)

        for msg in self._memory_context(include_tool_payload=True):
            role = msg.get("role")
            content = str(msg.get("content", ""))
            if role == "assistant" and "TOOL_CALL:" in content:
                reasoning_steps.append(content.strip())
            elif role == "tool":
                payload = msg.get("_tool_payload")
                if isinstance(payload, dict) and not evidence_ledger:
                    evidence_blocks.append(self._render_tool_payload(str(msg.get("_tool_name", "")).strip(), payload))

        reasoning_trace = "\n\n".join(reasoning_steps[-5 if compact else -8:]) or "(no reasoning trace captured)"
        visible_tool_limit = min(
            int(self.config.get("max_model_visible_tool_result_chars", 12000) or 12000),
            4500 if compact else 8000,
        )
        retrieved_context, _ = bound_model_visible_text(
            "\n\n".join(evidence_blocks[-3 if compact else -6:]) or (
                "(see the complete selected evidence records in the ledger below)"
                if evidence_ledger else "(no retrieved context captured)"
            ),
            max_chars=visible_tool_limit,
        )
        evidence_ledger_text = json.dumps(evidence_ledger, ensure_ascii=False, indent=2)
        safety_flags = self._reflection_safety_flags_text(compact=compact)
        wording_constraints = visible_wording_constraints(user_text)
        wording_section = ""
        if wording_constraints:
            wording_section = (
                "Stem wording constraints:\n"
                + "\n".join(f"- {item}" for item in wording_constraints)
                + "\n\n"
            )
        plan_section = ""
        if bool(self.config.get("enable_planner", False)):
            plan_section = (
                "Structured plan (guidance, not evidence):\n"
                + json.dumps(self.current_plan, ensure_ascii=False, indent=2)
                + "\n\n"
            )
        choice_table_section = self._reflection_choice_table(user_text, draft_answer)
        option_evidence_hints = self._reflection_option_evidence_hints(user_text, compact=compact)
        compact_output_section = ""
        if compact:
            compact_output_section = (
                "Compact JSON repair mode:\n"
                "- A previous reflection output did not parse. Return one complete JSON object only.\n"
                "- Use at most 4 evidence_assessments and 2 top_two_candidates.\n"
                "- If needed, set evidence_assessments=[] so the required top-level decision fields fit.\n\n"
            )

        return (
            f"Question:\n{user_text}\n\n"
            f"{wording_section}"
            f"{plan_section}"
            f"{choice_table_section}"
            f"{option_evidence_hints}"
            f"{compact_output_section}"
            f"Reasoning trace:\n{reasoning_trace}\n\n"
            f"Draft answer:\n{draft_answer}\n\n"
            f"Retrieved context:\n{retrieved_context}\n\n"
            "Reflection focus:\n"
            "- The source-preserved evidence ledger is ranked to keep authoritative, local, and decision-critical records visible.\n"
            "- If a high-authority record says Category 4, contraindicated, unacceptable risk, not to be used, or gives a guideline recommendation, explicitly compare that claim against the draft answer's main rationale.\n"
            "- Revising the final answer requires direct cited support for the replacement and a cited contradiction, exact-stem mismatch, unsupported parameter, or format defect in the draft. Broad plausibility alone is not enough.\n"
            "- Before preferring a candidate, verify that it covers the stem's discriminative positive anchors and does not ignore explicit negative anchors or narrower mechanical/structural/source clues.\n"
            "- Do not use a retry to search for support for the draft when the visible ledger already supports a different patient-specific candidate; revise instead.\n"
            "- If top candidates share an action but differ in dose, route, monitoring, timing, or intensity, verify that differing parameter directly rather than treating broad action evidence as sufficient.\n"
            "- If the stem asks for a source, trigger, mechanism, predisposing factor, treatment purpose, or adverse-effect management role, compare candidates against that exact relationship type.\n"
            "- Treat any local support/differentiator atom as a weak retrieval hint: use it to identify a parameter to verify, but revise only from direct source-preserved evidence.\n"
            "- For non-multiple-choice questions, still identify the two most plausible final short-answer candidates.\n\n"
            f"{safety_flags}\n\n"
            f"Source-preserved evidence ledger:\n{evidence_ledger_text}\n"
        )

    def _build_reflection_decision_only_prompt(self, user_text: str, draft_answer: str) -> str:
        evidence_ledger = self._reflection_evidence_ledger(compact=True)
        evidence_ledger_text = json.dumps(evidence_ledger, ensure_ascii=False, indent=2)
        choice_table_section = self._reflection_choice_table(user_text, draft_answer)
        wording_constraints = visible_wording_constraints(user_text)
        wording_section = ""
        if wording_constraints:
            wording_section = (
                "Stem wording constraints:\n"
                + "\n".join(f"- {item}" for item in wording_constraints)
                + "\n\n"
            )
        plan_section = ""
        if bool(self.config.get("enable_planner", False)):
            plan_section = (
                "Decision goal from plan:\n"
                + plan_goal_context(self.current_plan)[:900]
                + "\n\n"
            )
        option_evidence_hints = self._reflection_option_evidence_hints(user_text, compact=True)
        return (
            "A previous reflection response failed JSON parsing because it was too verbose.\n"
            "Return one minimal JSON object only. Do not include explanations outside JSON.\n"
            "Set evidence_assessments=[] unless one short record assessment is absolutely necessary.\n"
            "Use only record_id values visible in the evidence ledger. If the answer should change, "
            "put the complete corrected response including FINAL_ANSWER in revised_answer only when visible cited evidence directly supports the replacement.\n\n"
            f"Question:\n{user_text}\n\n"
            f"{wording_section}"
            f"{plan_section}"
            f"{choice_table_section}"
            f"{option_evidence_hints}"
            f"Draft answer:\n{draft_answer}\n\n"
            f"Compact source-preserved evidence ledger:\n{evidence_ledger_text}\n\n"
            "Required JSON shape:\n"
            "{\"evidence_assessments\":[],"
            "\"top_two_candidates\":[{\"label\":\"A\",\"text\":\"...\",\"supporting_evidence_ids\":[],\"contradicting_evidence_ids\":[],\"case_summary\":\"...\"}],"
            "\"preferred_candidate\":\"A\",\"runner_up\":\"B\",\"why_preferred_over_runner_up\":\"one sentence\","
            "\"confidence\":0.0,\"missing_fact\":\"\",\"retry\":{\"needed\":false,\"channel\":\"none\",\"query\":\"\",\"reason\":\"\"},"
            "\"needs_revision\":false,\"revised_answer\":\"\"}"
        )

    def _run_reflection(self, user_text: str, draft_answer: str) -> Dict[str, Any]:
        raw = ""
        data: Dict[str, Any] = {}
        parse_errors: List[str] = []
        base_max_tokens = max(500, int(self.config.get("reflection_max_tokens", 2400) or 2400))
        try:
            for attempt in range(3):
                prompt = (
                    self._build_reflection_decision_only_prompt(user_text, draft_answer)
                    if attempt == 2
                    else self._build_reflection_prompt(
                        user_text,
                        draft_answer,
                        compact=attempt > 0,
                    )
                )
                raw = self._chat(
                    [
                        {"role": "system", "content": self.reflection_system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    json_mode=True,
                    max_tokens=(
                        base_max_tokens
                        if attempt == 0
                        else max(base_max_tokens, 3600)
                        if attempt == 1
                        else max(1800, min(base_max_tokens, 2600))
                    ),
                    json_thinking_enabled=bool(
                        self.config.get("structured_json_thinking_enabled", False)
                    ),
                )
                data = extract_json_object(raw, required_keys=_REFLECTION_REQUIRED_KEYS)
                if data:
                    break
                parse_errors.append(f"attempt_{attempt + 1}: missing required top-level reflection keys")
            if not data:
                raise ValueError("; ".join(parse_errors) or "reflection did not return a complete JSON object")
        except Exception as exc:
            return {
                "needs_retry": False,
                "needs_revision": False,
                "error_reason": f"reflection_failed: {exc}",
                "evidence_assessments": [],
                "top_two_candidates": [],
                "preferred_candidate": "",
                "runner_up": "",
                "why_preferred_over_runner_up": "",
                "confidence": 0.0,
                "missing_fact": "",
                "retry": {"needed": False, "channel": "none", "query": "", "reason": ""},
                "revised_answer": "",
                "raw": raw,
            }

        assessments = []
        valid_evidence_ids = {str(item.get("record_id", "")) for item in self.evidence_records}
        for item in list(data.get("evidence_assessments") or [])[:30]:
            if not isinstance(item, dict):
                continue
            record_id = str(item.get("record_id", "") or "").strip()
            if record_id not in valid_evidence_ids:
                continue
            def _bounded_int(key: str, minimum: int, maximum: int) -> int:
                try:
                    value = float(item.get(key, 0) or 0)
                    if minimum == 0 and maximum == 4:
                        if 0.0 <= value <= 1.0:
                            value *= 4.0
                        elif 4.0 < value <= 5.0:
                            value *= 0.8
                    return max(minimum, min(maximum, int(round(value))))
                except Exception:
                    return minimum
            assessments.append(
                {
                    "record_id": record_id,
                    "source_reliability": _bounded_int("source_reliability", 0, 4),
                    "relevance": _bounded_int("relevance", 0, 4),
                    "patient_applicability": _bounded_int("patient_applicability", 0, 4),
                    "consistency": _bounded_int("consistency", -2, 2),
                    "supports": str(item.get("supports", "") or "").strip()[:300],
                    "concern": str(item.get("concern", "") or "").strip()[:500],
                }
            )

        top_two = []
        for item in list(data.get("top_two_candidates") or [])[:2]:
            if not isinstance(item, dict):
                continue
            top_two.append(
                {
                    "label": str(item.get("label", "") or "").strip()[:40],
                    "text": str(item.get("text", "") or "").strip()[:1000],
                    "supporting_evidence_ids": [
                        str(value) for value in list(item.get("supporting_evidence_ids") or [])[:12]
                        if str(value) in valid_evidence_ids
                    ],
                    "contradicting_evidence_ids": [
                        str(value) for value in list(item.get("contradicting_evidence_ids") or [])[:12]
                        if str(value) in valid_evidence_ids
                    ],
                    "case_summary": str(item.get("case_summary", "") or "").strip()[:1200],
                }
            )

        try:
            confidence = max(0.0, min(1.0, float(data.get("confidence", 0.0) or 0.0)))
        except Exception:
            confidence = 0.0
        raw_retry = data.get("retry") if isinstance(data.get("retry"), dict) else {}
        retry_query = str(raw_retry.get("query", "") or "").strip()[:800]
        retry_channel = str(raw_retry.get("channel", "none") or "none").strip().lower()
        retry_enabled = bool(self.config.get("reflection_retry_enabled", True))
        if retry_channel not in {"online", "local", "either", "none"}:
            retry_channel = "none"
        online_available = bool(self.allowed_tool_set & ONLINE_EVIDENCE_TOOLS)
        local_available = bool(self.allowed_tool_set & LOCAL_EVIDENCE_TOOLS)
        if retry_channel == "either":
            if online_available and local_available:
                pass
            elif online_available:
                retry_channel = "online"
            elif local_available:
                retry_channel = "local"
            else:
                retry_channel = "none"
        elif retry_channel == "online" and not online_available:
            retry_channel = "none"
        elif retry_channel == "local" and not local_available:
            retry_channel = "none"
        missing_fact = str(data.get("missing_fact", "") or "").strip()[:800]
        requested_retry = bool(raw_retry.get("needed", False))
        threshold = float(self.config.get("reflection_retry_confidence_threshold", 0.58) or 0.58)
        if (
            retry_enabled
            and bool(self.config.get("reflection_generic_mcq_evidence_gap_local_retry", True))
            and local_available
            and int(self.execution_budget.local_tool_calls) == 0
            and not requested_retry
            and self._looks_like_multiple_choice_task(user_text)
            and not self._has_inspected_or_structured_evidence()
        ):
            missing_fact = missing_fact or (
                "Source-backed evidence is needed to distinguish the leading multiple-choice candidates for the exact visible stem."
            )
            retry_query = self._generic_mcq_local_retry_query(
                user_text,
                top_two=top_two,
                missing_fact=missing_fact,
                retry_query=retry_query,
            )
            retry_channel = "local"
            requested_retry = True
        if (
            retry_enabled
            and bool(self.config.get("reflection_generic_clinical_context_local_retry", True))
            and local_available
            and int(self.execution_budget.local_tool_calls) == 0
            and not requested_retry
            and self._looks_like_clinical_context_task(user_text)
            and not self._has_inspected_or_structured_evidence()
        ):
            missing_fact = missing_fact or (
                "Source-backed local evidence is needed to verify the patient-specific clinical context, "
                "medication context, and decision boundary before finalizing."
            )
            retry_query = self._build_local_prefetch_query(user_text)
            retry_channel = "local"
            requested_retry = True
        if (
            retry_enabled
            and bool(self.config.get("reflection_prefer_unused_local_retry", True))
            and local_available
            and int(self.execution_budget.local_tool_calls) == 0
            and retry_channel in {"online", "either"}
            and (requested_retry or confidence < threshold)
        ):
            local_parts = [
                retry_query,
                missing_fact,
                plan_goal_context(self.current_plan) if bool(self.config.get("enable_planner", False)) else "",
                " ".join(plan_entity_names(self.current_plan)[:12]) if bool(self.config.get("enable_planner", False)) else "",
            ]
            retry_query = " | ".join(part for part in local_parts if str(part).strip())[:800]
            retry_channel = "local"
        local_gap_threshold = float(
            self.config.get("reflection_local_retry_missing_fact_confidence_threshold", 0.75) or 0.75
        )
        if (
            retry_enabled
            and bool(self.config.get("reflection_missing_fact_local_retry", False))
            and local_available
            and int(self.execution_budget.local_tool_calls) == 0
            and not requested_retry
            and missing_fact
            and self._looks_like_multiple_choice_task(user_text)
            and confidence < local_gap_threshold
        ):
            retry_query = self._generic_mcq_local_retry_query(
                user_text,
                top_two=top_two,
                missing_fact=missing_fact,
                retry_query=retry_query,
            )
            retry_channel = "local"
            requested_retry = True
        followup_missing_fact_threshold = float(
            self.config.get("reflection_followup_missing_fact_confidence_threshold", 0.86) or 0.86
        )
        if (
            retry_enabled
            and bool(self.config.get("reflection_followup_missing_fact_local_retry", False))
            and local_available
            and int(self.execution_budget.local_tool_calls) > 0
            and not requested_retry
            and missing_fact
            and self._looks_like_multiple_choice_task(user_text)
            and confidence < followup_missing_fact_threshold
            and self._can_retry_local_evidence()
        ):
            retry_query = self._generic_mcq_local_retry_query(
                user_text,
                top_two=top_two,
                missing_fact=missing_fact,
                retry_query=retry_query,
            )
            retry_channel = "local"
            requested_retry = True
        if (
            retry_enabled
            and bool(self.config.get("reflection_weak_mcq_local_retry", False))
            and local_available
            and not requested_retry
            and self._can_retry_local_evidence()
            and self._weak_mcq_local_evidence_gap(
                assessments=assessments,
                top_two=top_two,
                user_text=user_text,
            )
        ):
            missing_fact = missing_fact or (
                "The visible local evidence has not yet directly supported the top-two multiple-choice comparison; "
                "retrieve one focused source-backed record that distinguishes the leading options for the exact stem."
            )
            retry_query = self._generic_mcq_local_retry_query(
                user_text,
                top_two=top_two,
                missing_fact=missing_fact,
                retry_query=retry_query,
            )
            retry_channel = "local"
            requested_retry = True
        if (
            retry_enabled
            and bool(self.config.get("reflection_extra_parameter_local_retry", True))
            and local_available
            and not requested_retry
            and self._looks_like_multiple_choice_task(user_text)
            and self._can_retry_local_evidence()
            and self._extra_parameter_gap_for_reflection(
                top_two=top_two,
                preferred_candidate=str(data.get("preferred_candidate", "") or ""),
            )
        ):
            missing_fact = missing_fact or (
                "One leading option adds an extra dose, route, timing, monitoring, laboratory, "
                "or treatment-intensity parameter that is not uniquely supported by cited evidence."
            )
            retry_query = self._generic_mcq_local_retry_query(
                user_text,
                top_two=top_two,
                missing_fact=missing_fact,
                retry_query=retry_query,
            )
            retry_channel = "local"
            requested_retry = True
        retry_possible, _ = self.execution_budget.can_start_tool_turn()
        needs_retry = bool(
            retry_enabled
            and retry_possible
            and retry_query
            and missing_fact
            and retry_channel != "none"
            and (requested_retry or confidence < threshold)
        )

        needs_revision = bool(data.get("needs_revision", False))
        revised_answer = str(data.get("revised_answer", "")).strip()
        preferred_candidate = str(data.get("preferred_candidate", "") or "").strip()[:500]
        if not needs_revision or not revised_answer:
            auto_revised = self._auto_revision_from_reflection(
                user_text=user_text,
                draft_answer=draft_answer,
                top_two=top_two,
                preferred_candidate=preferred_candidate,
                confidence=confidence,
            )
            if auto_revised:
                needs_revision = True
                revised_answer = auto_revised

        return {
            "needs_retry": needs_retry,
            "needs_revision": needs_revision,
            "error_reason": str(raw_retry.get("reason", "") or "").strip(),
            "evidence_assessments": assessments,
            "top_two_candidates": top_two,
            "preferred_candidate": preferred_candidate,
            "runner_up": str(data.get("runner_up", "") or "").strip()[:500],
            "why_preferred_over_runner_up": str(data.get("why_preferred_over_runner_up", "") or "").strip()[:1500],
            "confidence": confidence,
            "missing_fact": missing_fact,
            "retry": {
                "needed": needs_retry,
                "channel": retry_channel,
                "query": retry_query,
                "reason": str(raw_retry.get("reason", "") or "").strip()[:800],
            },
            "revised_answer": revised_answer,
            "raw": raw,
        }

    def _build_retry_prompt(self, original_user_text: str, draft_answer: str, reflection: Dict[str, Any]) -> str:
        retry = reflection.get("retry") if isinstance(reflection.get("retry"), dict) else {}
        retry_query = str(retry.get("query", "") or "").strip()
        retry_channel = str(retry.get("channel", "either") or "either").strip()
        error_reason = reflection.get("error_reason") or "One decision-critical fact remains unresolved."
        wording_constraints = visible_wording_constraints(original_user_text)
        wording_section = ""
        if wording_constraints:
            wording_section = (
                "Stem wording constraints to preserve:\n"
                + "\n".join(f"- {item}" for item in wording_constraints)
                + "\n\n"
            )
        channel_instruction = ""
        if retry_channel == "local":
            channel_instruction = "Use search_local for this retry; do not call online tools for this missing fact.\n"
        elif retry_channel == "online":
            channel_instruction = "Use online retrieval for this retry; do not call search_local for this missing fact.\n"
        return (
            "Reflection feedback for the previous answer.\n\n"
            f"Original question:\n{original_user_text}\n\n"
            f"{wording_section}"
            f"Previous answer:\n{draft_answer}\n\n"
            f"Problem summary:\n{error_reason}\n\n"
            f"Missing fact:\n{reflection.get('missing_fact', '')}\n\n"
            f"Bounded retry channel: {retry_channel}\n"
            f"Bounded retry query: {retry_query}\n\n"
            f"{channel_instruction}"
            "Treat the problem summary and retry query as a hypothesis to test, not as an instruction "
            "to preserve the previous answer. If the new evidence contradicts that premise or better "
            "supports another listed option, compare it against the previous answer and revise. "
            "Resolve only this missing fact using at most the remaining shared execution budget. "
            "Do not broaden the search or inspect every option. If the bounded retrieval surfaces a higher-authority "
            "patient-specific contraindication, guideline exclusion, or safety issue for another visible medication, "
            "do not ignore it. Then re-answer the original question with the required FINAL_ANSWER format."
        )

    def _debate_evidence_ledger(self) -> List[Dict[str, Any]]:
        records = self._reflection_evidence_ledger(compact=False)
        seen = {item["record_id"] for item in records}
        # Ensure newly acquired evidence is available to both sides even if
        # older high-authority records occupy the ranked reflection window.
        for record in self.evidence_records[-4:]:
            if record.get("record_id") in seen:
                continue
            item = {key: record.get(key) for key in (
                "record_id", "tool", "title", "url", "source_type",
                "authority_tier", "content_kind", "entity", "goal",
            )}
            item["content"] = self._evidence_content_for_reflection(record)
            records.append(item)
            seen.add(record["record_id"])
        return records

    def _retrieve_debate_evidence(self, request: Dict[str, Any]) -> Dict[str, Any]:
        self._refresh_available_tools()
        if not bool(self.config.get("enable_tools", True)) or not bool(self.config.get("reflection_retry_enabled", True)):
            return {"status": "retrieval_disabled"}
        channel = request["channel"]
        tool = str(request.get("tool") or "").strip()
        if not tool:
            if channel in {"local", "either"} and "search_local" in self.allowed_tool_set:
                tool = "search_local"
            elif "web_search" in self.allowed_tool_set:
                tool = "web_search"
            else:
                return {"status": "blocked", "reason": "no_available_query_tool", "available_tools": self.allowed_tools}
        if tool not in self.allowed_tool_set:
            return {"status": "blocked", "reason": "tool_unavailable", "tool": tool, "available_tools": self.allowed_tools}
        if (channel == "local" and tool not in LOCAL_EVIDENCE_TOOLS
                or channel == "online" and tool not in ONLINE_EVIDENCE_TOOLS
                or tool not in LOCAL_EVIDENCE_TOOLS | ONLINE_EVIDENCE_TOOLS):
            return {"status": "blocked", "reason": "channel_tool_mismatch", "tool": tool}
        allowed, reason = self.execution_budget.can_start_tool_turn()
        if allowed:
            allowed, reason = self.execution_budget.can_call_tool(tool)
        if not allowed:
            return {"status": "blocked", "reason": reason, "tool": tool}
        args = dict(request.get("arguments") or {})
        if tool in {"search_local", "web_search"}:
            args["query"] = request["query"]
        if tool == "search_local":
            args.setdefault("require_rag", True)
            args.setdefault("max_evidence", int(self.config.get("reflection_retry_max_evidence", 12)))
        if tool == "web_visit":
            args.setdefault("goal", request["query"])
        before = {record["record_id"] for record in self.evidence_records}
        self.memory.add_assistant(render_tool_call_block([(tool, args)]))
        executed = self._execute_tool_calls(
            [(tool, args)], max_tools_this_turn=1, phase_label="REC DEBATE TARGETED RETRIEVAL",
        )
        if executed:
            self.execution_budget.record_tool_turn()
        payload = next((message.get("_tool_payload", {}) for message in reversed(self._memory_context(include_tool_payload=True))
                        if message.get("_tool_name") == tool), {})
        added = [record["record_id"] for record in self.evidence_records if record["record_id"] not in before]
        result = {"status": "retrieved" if added else "no_new_evidence", "tool": tool,
                  "executed_tools": [tool] if executed else [], "new_evidence_ids": added}
        if payload.get("success") is False:
            if tool == "search_local" and payload.get("status") == "no_results" and not payload.get("error"):
                return {**result, "status": "no_new_evidence", "reason": "no_matching_records"}
            return {**result, "status": "retrieval_error", "error": payload.get("error", "Tool failed")}
        if tool == "web_search":
            urls = [record.get("url") for record in self.evidence_records
                    if record["record_id"] in added and record.get("url")]
            if not urls:
                return {**result, "status": "no_new_evidence"}
            result["status"] = "discovery_only"
            if "web_visit" in self.allowed_tool_set:
                visit = self._retrieve_debate_evidence({"channel": "online", "tool": "web_visit",
                    "query": request["query"], "arguments": {"url": urls[0], "goal": request["query"]}})
                result.update(status="retrieved" if visit["status"] == "retrieved" else "discovery_only",
                    executed_tools=result["executed_tools"] + visit.get("executed_tools", []),
                    inspection=visit,
                    new_evidence_ids=list(dict.fromkeys(added + visit.get("new_evidence_ids", []))))
        return result

    def _run_rec_debate(self, user_text: str, draft_answer: str) -> str:
        self._refresh_available_tools()
        self.debate_history = []
        self.debate_state = {"status": "stopped", "winner": "H0", "stop_reason": "initializing"}
        if int(self.config.get("debate_max_rounds", 2)) <= 0:
            self.debate_state["stop_reason"] = "debate_disabled"
            return draft_answer
        seed = self._run_reflection(user_text, draft_answer)
        self.reflection_history.append({**seed, "stage": "debate_initialization"})
        self._log("[REC DEBATE | INITIAL ASSESSMENT]\n" + json.dumps(seed, ensure_ascii=False, indent=2))
        choices = self._choice_map_for_reflection(user_text)
        draft_label = self._extract_mcq_final_label(user_text, draft_answer) if choices else ""
        draft_text = self._extract_final_answer_line(draft_answer).strip()
        candidates = list(seed.get("top_two_candidates") or [])
        preferred = str(seed.get("preferred_candidate") or "").strip()
        candidates.sort(key=lambda item: 0 if str(item.get("label") or "").strip() == preferred else 1)
        alternative = None
        for candidate in candidates:
            label = str(candidate.get("label") or "").strip().upper()
            text = str(candidate.get("text") or "").strip()
            if choices:
                if draft_label and label in choices and label != draft_label:
                    alternative = {**candidate, "label": label, "text": choices[label]}
                    break
            elif text and re.sub(r"\W+", " ", self._extract_final_answer_line(text).casefold()).strip() != re.sub(r"\W+", " ", draft_text.casefold()).strip():
                alternative = {**candidate, "text": self._extract_final_answer_line(text)}
                break
        if alternative is None:
            self.debate_state["stop_reason"] = "no_valid_alternative"
            self._log("[REC DEBATE | STOP]\n" + json.dumps(self.debate_state))
            return draft_answer
        hypotheses = {
            "H0": {"label": draft_label, "text": choices.get(draft_label, draft_text), "rationale": "Provisional answer to audit."},
            "H1": {
                "label": alternative.get("label", ""), "text": alternative["text"],
                "rationale": alternative.get("case_summary", ""),
                "supporting_evidence_ids": alternative.get("supporting_evidence_ids", []),
            },
        }
        context = {
            "visible_case": user_text, "structured_plan": self.current_plan,
            "retrieval_tools": get_tool_specs(enabled_tools=self.allowed_tools, disabled_tools=self.disabled_tools) if self.allowed_tools else {},
        }
        try:
            debate = EvidenceDebate(
                hypotheses=hypotheses, context=context, chat=self._chat,
                ledger=self._debate_evidence_ledger, retrieve=self._retrieve_debate_evidence,
                can_continue=lambda: self.execution_budget.can_call_llm(self._current_total_tokens()),
                strong_support=self._strong_revision_support_ids,
                max_rounds=int(self.config.get("debate_max_rounds", 2)),
                max_retrievals=int(self.config.get("debate_max_retrievals", 2)),
                max_tokens=int(self.config.get("debate_max_tokens", 1800)),
                max_retries=int(self.config.get("debate_max_retries", 1)),
            )
            result = debate.run()
            self.debate_history = result.pop("history")
            self.debate_state = {"status": "finished", "hypotheses": hypotheses, **result}
        except Exception as exc:
            self.debate_state["stop_reason"] = f"debate_failed:{exc}"
        self._log("[REC DEBATE | HISTORY]\n" + json.dumps(self.debate_history, ensure_ascii=False, indent=2))
        self._log("[REC DEBATE | RESULT]\n" + json.dumps(self.debate_state, ensure_ascii=False, indent=2))
        if self.debate_state.get("winner") != "H1":
            return draft_answer
        answer = alternative["label"] if choices else self._extract_final_answer_line(alternative["text"])
        if choices or "FINAL_ANSWER" in user_text or "FINAL_ANSWER" in draft_answer:
            return f"FINAL_ANSWER: {answer}"
        return answer

    def _maybe_reflect(self, user_text: str, draft_answer: str) -> str:
        self.debate_history = []
        self.debate_state = {"status": "not_run"}
        if not bool(self.config.get("enable_reflection", False)):
            return draft_answer

        if bool(self.config.get("enable_rec_debate", True)):
            if int(self.config.get("reflection_max_passes", 1)) <= 0:
                return draft_answer
            return self._run_rec_debate(user_text, draft_answer)

        max_passes = max(0, int(self.config.get("reflection_max_passes", 1)))
        retry_followup_passes = max(0, int(self.config.get("reflection_retry_followup_passes", 1) or 1))
        current_answer = draft_answer

        pass_idx = 0
        pass_budget = max_passes
        while pass_idx < pass_budget:
            reflection = self._run_reflection(user_text, current_answer)
            pass_idx += 1
            self.reflection_history.append(dict(reflection))
            self._log(f"[REFLECTION | PASS {pass_idx}]\n" + json.dumps(reflection, ensure_ascii=False, indent=2))

            if reflection.get("needs_retry"):
                retry_prompt = self._build_retry_prompt(user_text, current_answer, reflection)
                self.memory.add_assistant(current_answer)
                self.memory.add_user(retry_prompt)
                if self._execute_direct_local_retry(user_text, reflection):
                    if bool(self.config.get("reflection_direct_retry_answer_only", True)):
                        current_answer = self._answer_after_direct_retry()
                    else:
                        current_answer = self._run_main_agent()
                else:
                    current_answer = self._run_main_agent()
                if retry_followup_passes > 0:
                    pass_budget += 1
                    retry_followup_passes -= 1
                continue

            revised_answer = str(reflection.get("revised_answer", "")).strip()
            if reflection.get("needs_revision") and revised_answer:
                return revised_answer
            return current_answer

        return current_answer

    def run_turn(self, user_text: str) -> str:
        self.current_user_text = str(user_text or "")
        self._log("[USER]\n" + user_text)
        self.memory.add_user(user_text)

        if bool(self.config.get("enable_planner", False)):
            self._run_planner(user_text)
        else:
            self.current_plan = disabled_plan()
            self._refresh_answer_prompt()
        self._prefetch_local_evidence(user_text)
        self._refresh_answer_prompt()
        draft_answer = self._run_main_agent()
        self._log("[ASSISTANT | DRAFT]\n" + (draft_answer or ""))

        final_answer = self._maybe_reflect(user_text, draft_answer)
        final_answer = self._maybe_compact_final_short_answer(user_text, final_answer)
        self.memory.add_assistant(final_answer)
        self._log("[ASSISTANT | FINAL]\n" + (final_answer or ""))
        return final_answer
