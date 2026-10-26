#!/usr/bin/env python3
"""Run three independent low-cost cases and retain a complete local review bundle."""
from __future__ import annotations

import argparse
from datetime import datetime
import getpass
import json
import os
from pathlib import Path
import subprocess
import sys
from zoneinfo import ZoneInfo

from smoke_one import ROOT, check_dependencies

def load_cases(path):
    """Read local inputs; only visible fields are rendered into model questions."""
    cases = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(cases, list) or len(cases) != 3:
        raise ValueError("The cases file must contain exactly three cases.")
    for case in cases:
        if not isinstance(case, dict) or not isinstance(case.get("question"), str) or not case["question"].strip():
            raise ValueError("Each case requires a nonempty question string.")
        if case.get("answer_type") not in {"mcq", "open"}:
            raise ValueError("Each case requires answer_type mcq or open.")
        if not isinstance(case.get("context", ""), str):
            raise ValueError("Context must be text.")
        choices = case.get("choices")
        if case["answer_type"] == "mcq" and (
            not isinstance(choices, dict) or len(choices) < 2
            or any(not isinstance(k, str) or len(k) != 1 or not "A" <= k <= "Z"
                   or not isinstance(v, str) or not v.strip() for k, v in choices.items())
        ):
            raise ValueError("MCQ cases require at least two lettered choices.")
        if case["answer_type"] == "open" and choices:
            raise ValueError("Open cases must not include answer choices.")
    return cases


def visible_question(case):
    parts = []
    if case.get("context", "").strip():
        parts.append("Context:\n" + case["context"])
    parts.append(case["question"])
    if case["answer_type"] == "mcq":
        parts.append("Choices:\n" + "\n".join(f"{label}. {text}" for label, text in case["choices"].items()))
        parts.append("End with exactly: FINAL_ANSWER: <one displayed option letter>")
    else:
        parts.append("End with exactly: FINAL_ANSWER: <short answer, including units when applicable>")
    return "\n\n".join(parts)


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true", help="Check all three setups without reading a key or calling a model.")
    parser.add_argument("--cases-file", type=Path, default=ROOT / ".runs/review_cases.json", help="Local JSON file containing three question-bank cases; never staged in Git.")
    parser.add_argument("--model")
    parser.add_argument("--thinking", choices=("on", "off"))
    parser.add_argument("--reasoning-effort")
    parser.add_argument("--base-url")
    parser.add_argument("--api-style", choices=("deepseek", "openai"), default="deepseek")
    parser.add_argument("--api-key-env", help="Default: DEEPSEEK_API_KEY, or API_KEY for openai style.")
    args = parser.parse_args()
    if args.api_style == "openai" and (not args.base_url or not args.model):
        parser.error("--api-style openai requires --base-url and --model.")
    if args.api_style == "openai" and args.thinking is not None:
        parser.error("--thinking is DeepSeek-specific; generic APIs use provider-supported --reasoning-effort.")
    return args


def save_summary(folder, results, interrupted=False):
    summary = {"interrupted": interrupted, "completed_cases": len(results), "cases": results}
    (folder / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["TracePharm three-case review", "", "Each case uses an independent budget and process.",
             "PASS indicates workflow checks only. Reference answers are for manual review, not automatic scoring.", ""]
    for item in results:
        lines += [f"{item['case']}: {item['status']} (exit {item['exit_code']})",
                  "Source: " + json.dumps(item.get("source", {}), ensure_ascii=False),
                  "Final answer: " + (item.get("final_answer") or "No answer produced."),
                  "Reference answer (manual review): " + str(item.get("reference_answer") if item.get("reference_answer") is not None else "unavailable"),
                  "Failed checks: " + (", ".join(item.get("failed_checks", [])) or "none"),
                  "Warnings: " + ("; ".join(item.get("warnings", [])) or "none"),
                  "Error: " + (item.get("error") or "none"),
                  "Records: " + item["directory"], ""]
    if interrupted:
        lines.append("Cancelled before all three cases finished; inspect the last case's console.log and api_calls.jsonl.")
    (folder / "summary.txt").write_text("\n".join(lines), encoding="utf-8")


def main():
    args = arguments()
    check_dependencies()
    selected = load_cases(args.cases_file)
    key_env = args.api_key_env or ("API_KEY" if args.api_style == "openai" else "DEEPSEEK_API_KEY")
    key = ""
    if not args.preflight:
        key = os.environ.get(key_env, "").strip()
        if not key:
            if not sys.stdin.isatty():
                raise RuntimeError(f"Run in an interactive terminal for hidden input, or set {key_env}.")
            key = getpass.getpass("API key for all three cases (hidden, not saved): ").strip()
        if not key:
            raise ValueError("API key is empty; no model requests made.")
    folder = ROOT / ".runs" / ("three-" + datetime.now(ZoneInfo("Asia/Singapore")).strftime("%Y%m%d-%H%M%S-%f"))
    folder.mkdir(parents=True)
    cases = []
    for index, case in enumerate(selected, 1):
        name = f"{index:02d}-{case['answer_type']}"
        question = folder / (name + "-question.txt")
        question.write_text(visible_question(case), encoding="utf-8")
        cases.append((name, ["--question-file", str(question), "--answer-type", case["answer_type"]], case))
    # Provenance and references are audit-only: no path to this file is passed to the agent.
    (folder / "case_manifest.json").write_text(json.dumps([
        {k: case.get(k) for k in ("id", "answer_type", "source", "reference_answer")}
        for case in selected], ensure_ascii=False, indent=2), encoding="utf-8")
    environment = dict(os.environ)
    if key:
        environment[key_env] = key
    options = ["--api-style", args.api_style, "--api-key-env", key_env]
    for name in ("model", "base_url", "thinking", "reasoning_effort"):
        value = getattr(args, name)
        if value is not None:
            options += ["--" + name.replace("_", "-"), value]
    if args.preflight:
        options.append("--preflight")
    results = []
    save_summary(folder, results)
    print(f"All records: {folder}", flush=True)
    process = None
    try:
        for name, case_args, case in cases:
            directory = folder / name
            print(f"\n=== {name} ===", flush=True)
            # The child creates its own output directory; never pass the key as an argument.
            console_path = folder / (name + "-console.log")
            command = [sys.executable, "-u", "-B", str(ROOT / "scripts/smoke_one.py"),
                       *case_args, *options, "--output-dir", str(directory), "--save-api-transcript"]
            with console_path.open("w", encoding="utf-8") as console:
                process = subprocess.Popen(command, cwd=ROOT, env=environment, stdin=subprocess.DEVNULL,
                                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8")
                for line in process.stdout:
                    clean = line.replace(key, "[REDACTED]") if key else line
                    console.write(clean); console.flush()
                    print(clean, end="", flush=True)
                code = process.wait()
                process.stdout.close()
            report_path = directory / "report.json"
            try:
                report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
            except (OSError, ValueError) as exc:
                report = {"status": "FAILED_REPORT_READ", "error": str(exc)}
                code = code or 1
            results.append({"case": name, "directory": name, "console": console_path.name,
                "source": case.get("source", {}), "source_id": case.get("id"),
                "answer_type": case["answer_type"], "reference_answer": case.get("reference_answer"),
                "exit_code": code, "status": report.get("status", "PREFLIGHT_PASS" if args.preflight and code == 0 else "FAILED_WITHOUT_REPORT"),
                "final_answer": report.get("final_answer", ""),
                "failed_checks": [k for k, v in report.get("checks", {}).items() if v is False],
                "warnings": report.get("warnings", []), "error": report.get("error", ""),
                "api_requests": len(report.get("api_calls", []))})
            save_summary(folder, results)
    except KeyboardInterrupt:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill(); process.wait()
        save_summary(folder, results, interrupted=True)
        print(f"\nCancelled. Partial records: {folder}")
        return 130
    print(f"\nReview summary: {folder / 'summary.txt'}")
    print(f"Structured summary: {folder / 'summary.json'}")
    return 0 if all(item["exit_code"] == 0 for item in results) else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (KeyboardInterrupt, EOFError):
        print("\nRun cancelled.", file=sys.stderr)
        sys.exit(130)
    except Exception as exc:
        print(f"Could not run: {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(1)
