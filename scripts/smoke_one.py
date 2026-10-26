#!/usr/bin/env python3
"""Run one low-cost synthetic case through the real TracePharm pipeline."""
from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
import getpass
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
import threading
import traceback
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path[:0] = [str(ROOT / "models/agent/runtime"), str(ROOT)]

# Synthetic connectivity/behavior check, not a released benchmark item.
QUESTION = """A hypothetical adult with type 2 diabetes has an eGFR of
25 mL/min/1.73 m2 and is not currently taking metformin. According to US FDA
metformin labeling, which action is appropriate?
Choices:
A. Initiate metformin 500 mg daily because the eGFR is above 15.
B. Do not initiate metformin because it is contraindicated when eGFR is below 30.
Use source-backed evidence to verify the renal threshold. End with exactly:
FINAL_ANSWER: <one displayed option letter>"""
# Used only after inference; never included in prompts, tools or agent context.
EXPECTED_LABEL = "B"
REFERENCE_SOURCE = "https://www.accessdata.fda.gov/drugsatfda_docs/label/2026/021748s031lbl.pdf"
CASE_ID = "simple"
CHOICE_LABELS = {"A", "B"}
PREFLIGHT_QUERY = "medication reconciliation"
SIMPLE_CASE = (QUESTION, EXPECTED_LABEL, REFERENCE_SOURCE, CHOICE_LABELS, PREFLIGHT_QUERY)

# A harder synthetic case: baseline dose criteria plus a dose-dependent interaction.
# Expected answers and reference URLs are used only in the post-inference report.
COMPLEX_QUESTION = """A hypothetical 82-year-old woman weighs 58 kg and has stable serum
creatinine of 1.6 mg/dL. She has nonvalvular atrial fibrillation and is already
taking apixaban 2.5 mg orally twice daily for stroke prevention. She has no active
bleeding and no severe hepatic impairment. Oral systemic ketoconazole has been
proposed but has not yet been started. Do not evaluate the antifungal indication;
focus on the US FDA apixaban label's adult AF dose and interaction rules.
Which paired assessment of her baseline apixaban dose and the proposed combination
is consistent with that label?
Choices:
A. Her baseline dose is 2.5 mg twice daily; reduce it further to 1.25 mg twice daily and start ketoconazole.
B. Her baseline dose should be 5 mg twice daily; adding ketoconazole reduces that dose to 2.5 mg twice daily, so the proposed combination is acceptable.
C. Her baseline dose is 2.5 mg twice daily; avoid coadministration with systemic ketoconazole rather than applying another 50% dose reduction. Arrange an alternative treatment plan; do not simply leave her without anticoagulation.
D. Her baseline dose is 2.5 mg twice daily; keep both drugs at their proposed doses because the age, weight and creatinine adjustment already accounts for the interaction.
Use source-backed evidence to verify both the patient-specific baseline dose and
the interaction rule at that dose. End with exactly:
FINAL_ANSWER: <one displayed option letter>"""


def select_case(case_id, question_file=None):
    global QUESTION, EXPECTED_LABEL, REFERENCE_SOURCE, CASE_ID, CHOICE_LABELS, PREFLIGHT_QUERY
    if question_file is not None:
        question = Path(question_file).read_text(encoding="utf-8").strip()
        if not question:
            raise ValueError("The question file is empty.")
        CASE_ID, QUESTION = "custom", question
        EXPECTED_LABEL, REFERENCE_SOURCE = None, None
        CHOICE_LABELS = set(re.findall(r"(?m)^\s*([A-Z])[.)]\s+", question))
        PREFLIGHT_QUERY = "medication reconciliation"
        return
    if case_id == "simple":
        CASE_ID = case_id
        QUESTION, EXPECTED_LABEL, REFERENCE_SOURCE, CHOICE_LABELS, PREFLIGHT_QUERY = SIMPLE_CASE
        return
    if case_id != "complex":
        raise ValueError("Unknown smoke case: " + case_id)
    CASE_ID = case_id
    QUESTION = COMPLEX_QUESTION
    EXPECTED_LABEL = "C"
    REFERENCE_SOURCE = "https://www.accessdata.fda.gov/drugsatfda_docs/label/2025/202155Orig1s039,s040lbl.pdf"
    CHOICE_LABELS = {"A", "B", "C", "D"}
    PREFLIGHT_QUERY = "medication reconciliation"


def arguments():
    parser = argparse.ArgumentParser(description="Run one low-cost case; enter the API key privately at the terminal.")
    parser.add_argument("--preflight", action="store_true", help="Check local dependencies and index without reading credentials or calling models.")
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--case", choices=("simple", "complex"), default="simple", help="simple: metformin threshold; complex: apixaban dosing and interaction.")
    inputs.add_argument("--question-file", type=Path, help="Read a custom UTF-8 question; no reference-answer scoring is applied.")
    parser.add_argument("--answer-type", choices=("auto", "mcq", "open"), default="auto", help="open: fill-in/short-answer text, never an option letter; default: infer from choices.")
    parser.add_argument("--model", help="Override the default model (deepseek-flash).")
    parser.add_argument("--thinking", choices=("on", "off"), help="Override DeepSeek thinking for agent stages; reranking stays off.")
    parser.add_argument("--reasoning-effort", help="Provider-supported reasoning effort; empty string uses the provider default.")
    parser.add_argument("--base-url", help="Override the compatible endpoint; defaults to the official DeepSeek API.")
    parser.add_argument("--api-style", choices=("deepseek", "openai"), default="deepseek",
                        help="openai: standard Chat Completions without DeepSeek-specific thinking parameters.")
    parser.add_argument("--api-key-env", help="API key environment variable (default: DEEPSEEK_API_KEY, or API_KEY for openai style).")
    parser.add_argument("--output-dir", type=Path, help="Save this run in a new directory instead of an automatically named .runs directory.")
    parser.add_argument("--save-api-transcript", action="store_true", help="Save full redacted model requests and responses in api_calls.jsonl.")
    args = parser.parse_args()
    if args.api_style == "openai" and (not args.base_url or not args.model):
        parser.error("--api-style openai requires --base-url and --model.")
    return args


def check_dependencies():
    required = {"yaml": "PyYAML", "openai": "openai", "requests": "requests",
                "bs4": "beautifulsoup4", "dateutil": "python-dateutil", "numpy": "numpy", "faiss": "faiss-cpu"}
    missing = [package for module, package in required.items() if importlib.util.find_spec(module) is None]
    if missing:
        raise RuntimeError("Missing dependencies: " + ", ".join(missing) + ". Run run_smoke.command or install requirements.txt.")


def prepare(run_dir: Path, args):
    import yaml
    from tool_support.local_knowledge_base_query_cache import QueryEmbeddingCache
    profile = yaml.safe_load((ROOT / "configs/smoke.yaml").read_text())
    profile["agent_config_overrides"]["answer_type"] = getattr(args, "answer_type", "auto")
    if args.model:
        profile["model"] = args.model
    if args.base_url:
        profile["base_url"] = args.base_url
    generic_api = getattr(args, "api_style", "deepseek") == "openai"
    profile["api_key_env"] = getattr(args, "api_key_env", None) or ("API_KEY" if generic_api else "DEEPSEEK_API_KEY")
    profile["send_thinking_parameter"] = not generic_api
    thinking = getattr(args, "thinking", None)
    effort = getattr(args, "reasoning_effort", None)
    if thinking is not None:
        if generic_api:
            raise ValueError("--thinking is specific to DeepSeek; use provider-supported --reasoning-effort with openai style.")
        profile["thinking_enabled"] = thinking == "on"
        profile["agent_config_overrides"]["structured_json_thinking_enabled"] = thinking == "on"
    if effort is not None:
        profile["reasoning_effort"] = effort.strip()
    elif thinking == "off":
        profile["reasoning_effort"] = ""
    if generic_api and profile.get("thinking_enabled"):
        raise ValueError("thinking_enabled requires DeepSeek style; generic APIs use their own reasoning controls.")
    if not generic_api and profile.get("reasoning_effort") and not profile.get("thinking_enabled"):
        raise ValueError("DeepSeek reasoning_effort requires thinking_enabled: true or --thinking on.")
    if not generic_api and profile["agent_config_overrides"].get("structured_json_thinking_enabled") and not profile.get("thinking_enabled"):
        raise ValueError("structured_json_thinking_enabled requires thinking_enabled: true.")
    profile["max_api_requests"] = int(profile.get("max_api_requests", 16))
    if profile["max_api_requests"] < 1:
        raise ValueError("max_api_requests must be positive.")
    # Keep auxiliary clients on the same Chat Completions transport.
    profile["llm_backend"] = "deepseek"
    # Credentials are read separately and never written to a config file.
    profile["api_key"] = ""
    llm_path = run_dir / "llm.yaml"
    llm_path.write_text(yaml.safe_dump(profile, sort_keys=False))
    local = yaml.safe_load((ROOT / "models/agent/runtime/config/search_local.yaml").read_text())
    corpus = ROOT / "resources/general_knowledge"
    cache = run_dir / "cache/query.sqlite"
    QueryEmbeddingCache(cache_path=cache, read_only=False)
    local.update({
        "source_csv_paths": [str(corpus / "general_medical_knowledge.csv")],
        "entity_info_column": "all_entity_info_json",
        "database_path": str(corpus / "drug_knowledge_base.sqlite"),
        "faiss_index_path": str(corpus / "drug_knowledge_base.faiss"),
        "embedding_config_path": str(ROOT / "models/text_embedding_3_large/config.yaml"),
        "query_cache_path": str(cache), "read_only_runtime": True,
        "allow_legacy_artifacts": False, "dense_enabled": False,
        "hybrid_weights": {"dense": 0, "bm25": 1}, "bm25_branch_k": 4,
        "rerank_enabled": True, "rerank_model_config_path": str(llm_path),
        "rerank_candidate_k": 3, "rerank_batch_size": 3, "rerank_max_workers": 1,
        "rerank_max_retries": 1, "rerank_cache_enabled": False,
        "benchmark_affinity_enabled": False, "overlay_csv_paths": [],
    })
    local_path = run_dir / "search_local.yaml"
    local_path.write_text(yaml.safe_dump(local, sort_keys=False))
    os.environ["SEARCH_LOCAL_CONFIG_PATH"] = str(local_path)
    os.environ["AGENT_LLM_CONFIG_PATH"] = str(llm_path)
    return profile, llm_path, local_path


class TraceSink:
    def __init__(self, path: Path, key: str, console):
        self.file = path.open("w", encoding="utf-8")
        self.key, self.console = key, console
        self.shown = set()

    def write(self, text):
        clean = text.replace(self.key, "[REDACTED]") if self.key else text
        self.file.write(clean)
        for marker, label in [
            ("[PLANNER | CANDIDATE TREE]", "GAES: candidate tree generated."),
            ("[RELATION SCREENING | RESULT]", "GAES: relation screening returned; validation follows."),
            ("[REC DEBATE | INITIAL ASSESSMENT]", "REC: initial evidence assessment returned."),
            ("[REC DEBATE | RESULT]", "REC: debate stopped; termination checks follow."),
        ]:
            if marker in clean and marker not in self.shown:
                self.console.write(label + "\n")
                self.console.flush()
                self.shown.add(marker)
        return len(text)

    def flush(self):
        self.file.flush()
    def close(self):
        self.file.close()


def build_report(agent, final, api_calls, error):
    plan, debate = agent.current_plan, agent.debate_state
    history = agent.debate_history
    phases = {entry.get("phase") for entry in history}
    selected = {item["relation_id"] for item in plan.get("selected_relations", [])}
    requirements = {item["id"] for item in plan.get("evidence_requirements", [])}
    tasks = plan.get("search_tasks", [])
    screening_ok = plan.get("screening_status") == "ok"
    budget_stop = debate.get("stop_reason") in {
        "budget_exhausted:max_total_tokens", "budget_exhausted:max_llm_calls",
        "budget_exhausted:deadline_seconds",
    }
    retrievals = [entry for entry in history if entry.get("phase") == "retrieval"]
    incomplete_retrievals = [entry for entry in retrievals if entry.get("result", {}).get("status") not in {"retrieved", "no_new_evidence"}]
    checks = {
        "GAES_candidate_tree": bool(agent.candidate_plan.get("entities")),
        "GAES_relation_screening": screening_ok,
        "GAES_task_traceability": screening_ok and bool(tasks) and all(
            item.get("requirement_id") in requirements and set(item.get("relation_ids", [])) <= selected
            for item in tasks
        ),
        "SPEC_source_records": bool(agent.evidence_records),
        "REC_two_hypotheses": set(debate.get("hypotheses", {})) == {"H0", "H1"},
        "REC_challenge_defend_update": {"challenge", "defend", "update"} <= phases,
        "REC_explicit_termination": budget_stop or debate.get("stop_reason") in {
            "converged", "position_withdrawn", "round_limit", "retrieval_limit",
            "unsupported_consensus", "revision_without_counterevidence", "both_positions_withdrawn",
        },
        "REC_within_budget": not budget_stop,
        "REC_requested_retrieval_completed": not incomplete_retrievals if retrievals else None,
        "final_answer_format": valid_final_answer(agent, final),
        "thinking_explicitly_disabled": (
            all(item["thinking"] == "disabled" for item in api_calls)
            if api_calls and all(item["thinking"] == "disabled" for item in api_calls) else None
        ),
        "provider_options_appropriate": bool(api_calls) and all(item.get("provider_options_valid", item["thinking"] == "disabled") for item in api_calls),
    }
    reference_match = None
    if EXPECTED_LABEL is not None:
        reference_match = agent._extract_mcq_final_label(QUESTION, final) == EXPECTED_LABEL
        checks["single_case_reference_match"] = reference_match
    invalid = [item for item in history if item.get("phase") == "invalid_response"]
    unresolved_invalid = [item for item in invalid if not item.get("repaired", False)]
    warnings = []
    if budget_stop:
        warnings.append("REC stopped at an execution limit (" + debate["stop_reason"] + "); the complete debate was not verified.")
    if invalid:
        warnings.append("REC produced an invalid response; inspect the trace." if unresolved_invalid
                        else "REC repaired an invalid response; the original and corrected turns are retained in the trace.")
    if any(item.get("finish_reason") == "length" for item in api_calls):
        warnings.append("At least one model response was truncated by its output-token limit.")
    if any(item.get("error") for item in api_calls):
        warnings.append("At least one API request failed; inspect api_calls and the trace.")
    if debate.get("decision_status") == "provisional":
        warnings.append("REC did not reach a supported consensus; the provisional answer was retained.")
    if incomplete_retrievals:
        warnings.append("A requested REC retrieval was blocked, failed, or returned only discovery snippets; the retrieval branch is incomplete.")
    if any(entry.get("result", {}).get("status") == "no_new_evidence" for entry in retrievals):
        warnings.append("A requested REC retrieval completed but found no new evidence.")
    passed = all(value for value in checks.values() if value is not None) and not error and not unresolved_invalid
    return {
        "status": ("PASS_WITH_WARNINGS" if warnings else "PASS") if passed else "PARTIAL_OR_FAILED",
        "error": error, "checks": checks, "warnings": warnings,
        "case_id": CASE_ID, "answer_type": "mcq" if CHOICE_LABELS else "open", "question": QUESTION, "final_answer": final,
        "answer_matches_reference": reference_match,
        "reference_label": EXPECTED_LABEL,
        "reference_source": REFERENCE_SOURCE,
        "candidate_tree": agent.candidate_plan, "screened_plan": plan,
        "evidence_records": agent.evidence_records, "debate_state": debate,
        "debate_history": history, "execution_budget": agent.execution_budget.export(),
        "api_calls": api_calls,
        "coverage": {
            "selected_relations_count": len(selected),
            "evidence_requirements_count": len(requirements),
            "role_reversal_observed": {"H0", "H1"} <= {item.get("side") for item in history if item.get("phase") == "challenge"},
            "targeted_retrieval_requested": bool(retrievals),
            "targeted_retrieval_executed": any(entry.get("result", {}).get("executed_tools") for entry in retrievals),
            "targeted_retrieval_observed": any(entry.get("result", {}).get("status") == "retrieved" and entry.get("result", {}).get("new_evidence_ids") for entry in retrievals),
            "post_retrieval_reassessment_observed": "reassess" in phases,
        },
        "paper_scope": {
            "full_paper_reproduction": False,
            "differences": [
                "Selected API model with reduced output and call budgets; inspect api_calls for requested thinking options.",
                "16 general knowledge records; BM25 and model reranking without dense queries.",
                "Tools are filtered by enabled configuration and local prerequisites; inspect tool_availability.",
                "Early convergence can leave role reversal and retrieval branches untested.",
                "A single case does not establish clinical accuracy or reproduce paper results.",
            ],
        },
    }


def valid_final_answer(agent, final):
    if CHOICE_LABELS:
        return agent._extract_mcq_final_label(QUESTION, final) in CHOICE_LABELS
    if not str(final).strip() or agent._is_tool_call(final):
        return False
    if "FINAL_ANSWER" in QUESTION:
        lines = re.findall(r"(?m)^[ \t]*FINAL_ANSWER:[ \t]*([^\r\n]*)$", str(final))
        return len(lines) == 1 and bool(lines[0].strip()) and not bool(re.fullmatch(r"<[^>]+>|_+", lines[0].strip()))
    return True


def main():
    global CHOICE_LABELS, EXPECTED_LABEL, REFERENCE_SOURCE
    args = arguments()
    select_case(args.case, args.question_file)
    answer_type = getattr(args, "answer_type", "auto")
    if answer_type == "open":
        if not args.question_file:
            raise ValueError("Use --answer-type open with --question-file; built-in MCQ cases require their displayed choices.")
        CHOICE_LABELS = set()
        EXPECTED_LABEL, REFERENCE_SOURCE = None, None
    elif answer_type == "mcq" and len(CHOICE_LABELS) < 2:
        raise ValueError("MCQ input requires at least two choices labeled A., B., etc.")
    check_dependencies()
    output_dir = getattr(args, "output_dir", None)
    save_transcript = getattr(args, "save_api_transcript", False)
    run_dir = output_dir.resolve() if output_dir else ROOT / ".runs" / datetime.now(ZoneInfo("Asia/Singapore")).strftime("%Y%m%d-%H%M%S-%f")
    run_dir.mkdir(parents=True)
    os.environ.update({
        "PRIMEKG_CACHE_DIR": str(run_dir / "cache/primekg"),
        "ORANGE_BOOK_CACHE_DIR": str(run_dir / "cache/orangebook"),
        "DISABLE_TOOL_COMPRESSOR": "1", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
    })
    profile, llm_path, local_path = prepare(run_dir, args)
    from tool_support.local_knowledge_base_store import get_store
    from tools.tool_registry import get_allowed_tools, get_tool_availability
    store = get_store(local_path)
    tools = get_allowed_tools(enabled_tools=profile["agent_config_overrides"]["enabled_tools"], strict=True)
    availability = get_tool_availability(enabled_tools=profile["agent_config_overrides"]["enabled_tools"])
    (run_dir / "tool_availability.json").write_text(json.dumps(availability, indent=2))
    lexical = store.bm25_search(PREFLIGHT_QUERY, 4)
    print(f"Preflight passed: frozen index readable; tools: {', '.join(tools)}.")
    for name, info in availability.items():
        if not info["available"]:
            print(f"Unavailable tool: {name} ({info['reason']}).")
    print("Case: " + {
        "simple": "metformin renal threshold (two choices)",
        "complex": "apixaban dose and ketoconazole interaction (four choices)",
        "custom": "custom question",
    }[CASE_ID])
    thinking_label = ("thinking enabled" if profile.get("thinking_enabled") else "thinking disabled") if profile["send_thinking_parameter"] else "provider-default reasoning (no thinking parameter)"
    rec_rounds = profile["agent_config_overrides"]["debate_max_rounds"]
    print(f"Configuration: {profile['model']}; {thinking_label}; reasoning effort: {profile.get('reasoning_effort') or 'provider default'}; at most {rec_rounds} REC round(s).")
    if args.preflight:
        print(f"No model-key prompt or model requests made. Tool credentials checked locally. BM25 probe hits: {len(lexical)}.")
        return 0
    key_env = profile["api_key_env"]
    key = os.environ.get(key_env, "").strip()
    if not key:
        if not sys.stdin.isatty():
            raise RuntimeError(f"Run in an interactive terminal for hidden key input, or set {key_env}.")
        key = getpass.getpass("API key (hidden, not saved): ").strip()
    if not key:
        raise RuntimeError("API key is empty; no model requests made.")
    os.environ[key_env] = key
    from agent.agent import Agent, _load_agent_config
    from agent.telemetry import TimingRecorder, bind_timing_recorder
    from llm.deepseek_client import DeepSeekClient
    console = sys.stdout
    sink = TraceSink(run_dir / "trace.log", key, console)
    calls, lock = [], threading.Lock()
    original = DeepSeekClient._request_once
    max_api_requests = profile["max_api_requests"]

    def save_api_event(event):
        if save_transcript:
            event = {"time": datetime.now(ZoneInfo("Asia/Singapore")).isoformat(), **event}
            line = json.dumps(event, ensure_ascii=False, default=str).replace(key, "[REDACTED]")
            with lock, (run_dir / "api_calls.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")

    def observed(client, kwargs):
        mode = kwargs.get("extra_body", {}).get("thinking", {}).get("type", "not_requested")
        allowed_modes = {"enabled", "disabled"} if profile.get("thinking_enabled") else {"disabled"}
        if not profile["send_thinking_parameter"]:
            allowed_modes = {"not_requested"}
        if mode not in allowed_modes:
            raise RuntimeError("Blocked a request with unexpected provider-specific thinking options.")
        with lock:
            if len(calls) >= max_api_requests:
                raise RuntimeError(f"Reached the {max_api_requests}-request limit for this run.")
            item = {"index": len(calls) + 1, "requested_model": kwargs["model"], "thinking": mode,
                    "reasoning_effort": kwargs.get("reasoning_effort"), "max_tokens": kwargs.get("max_tokens"), "provider_options_valid": True}
            calls.append(item)
        console.write(f"API request {item['index']}/{max_api_requests} (including retries and tool-internal model requests)\n")
        console.flush()
        save_api_event({"event": "request", "index": item["index"], "request": kwargs})
        try:
            response = original(client, kwargs)
            if save_transcript:
                save_api_event({"event": "response", "index": item["index"], "response": response.model_dump()})
            item.update(model=response.model, finish_reason=response.choices[0].finish_reason,
                        usage=response.usage.model_dump() if response.usage else {})
            return response
        except Exception as exc:
            item["error"] = str(exc).replace(key, "[REDACTED]")
            save_api_event({"event": "error", "index": item["index"], "error": item["error"]})
            raise

    DeepSeekClient._request_once = observed
    agent, llm, final, error = None, None, "", ""
    recorder = TimingRecorder()
    try:
        with redirect_stdout(sink), redirect_stderr(sink), bind_timing_recorder(recorder):
            config = _load_agent_config()
            config.update(profile["agent_config_overrides"])
            llm = DeepSeekClient(str(llm_path))
            agent = Agent(llm=llm, config=config, output_sink=sink)
            recorder.start_turn()
            try:
                final = agent.run_turn(QUESTION)
            finally:
                recorder.finish_turn()
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}".replace(key, "[REDACTED]")
        sink.write(traceback.format_exc())
    finally:
        DeepSeekClient._request_once = original
        if llm is not None:
            llm.close()
        sink.close()
    report = build_report(agent, final, calls, error) if agent is not None else {
        "status": "FAILED_BEFORE_AGENT", "error": error, "api_calls": calls,
    }
    report["timing"] = recorder.export()
    report["tool_availability"] = availability
    report["llm_settings"] = {name: profile.get(name) for name in (
        "model", "base_url", "thinking_enabled", "send_thinking_parameter", "reasoning_effort",
        "temperature", "request_timeout_seconds", "max_retries", "max_api_requests")}
    report["llm_settings"]["structured_json_thinking_enabled"] = profile["agent_config_overrides"].get("structured_json_thinking_enabled", False)
    (run_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2).replace(key, "[REDACTED]"))
    print(f"\nResult: {report['status']}")
    for name, passed in report.get("checks", {}).items():
        print(f"  {'N/A' if passed is None else 'PASS' if passed else 'FAIL'}  {name}")
    print("Final answer: ", (final or "No answer produced.").replace(key, "[REDACTED]"))
    for warning in report.get("warnings", []):
        print("Warning: ", warning)
    if error:
        print("Error: ", error)
    print(f"Trace: {run_dir / 'trace.log'}\nReport: {run_dir / 'report.json'}")
    return 0 if report["status"].startswith("PASS") else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (KeyboardInterrupt, EOFError):
        print("\nRun cancelled.", file=sys.stderr)
        sys.exit(130)
    except Exception as exc:
        print(f"Could not start: {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(1)
