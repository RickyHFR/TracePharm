# TracePharm

A runnable research implementation of a source-grounded pharmacotherapy agent.

TracePharm combines three modules: **GAES (Goal-anchored entity tree screening)** builds a patient-specific entity tree and screens relevant relations to identify the evidence needed for the treatment decision. **SPEC (Source-preserving evidence curation)** retrieves and deduplicates evidence while preserving source links and decision-critical details such as doses, thresholds, and patient eligibility. **REC (Reflective evidence defensible calibration)** compares the provisional decision with its strongest alternative through alternating evidence-grounded debate, requests additional evidence when needed, and upholds or revises the decision. The plan, evidence, debate history, and final answer are saved for inspection.

## Quick start

Tested with **Python 3.12 on macOS**. From the repository root:

```bash
./run_smoke.command --preflight   # Check dependencies and the local index
./run_smoke.command               # Run the simple example
./run_complex_smoke.command       # Run the drug-interaction example
```

The launchers create `.venv` and install `requirements.txt` when needed. Enter your DeepSeek API key at the hidden terminal prompt, or supply it through `DEEPSEEK_API_KEY`. Preflight checks local tool prerequisites and credential presence without prompting for a model key or calling a model.

Without the macOS launchers, set up a POSIX environment manually:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -B scripts/smoke_one.py --case complex
```

## Your own question

To review three local question-bank cases in one session, run `./run_three.command` (or add `--preflight` for a local setup check). It reads `.runs/review_cases.json`; use `--cases-file path/to/cases.json` for another local selection. This private input is not included in Git. It must be a JSON array of three objects with `question`, `answer_type` (`mcq` or `open`), optional `context`, and `choices` (a letter-to-text object for MCQ). Optional `id`, `source`, and `reference_answer` fields are retained for manual review. Only question, context, choices, and output-format instructions enter model prompts.

The script reads your API key once and runs the three cases sequentially with independent low-cost budgets. The same API options described below are supported. A failed case does not prevent the remaining cases from running. There is no synthetic fallback if the local input is missing.

Records are grouped in `.runs/three-<timestamp>/`: `summary.txt` and `summary.json`, the exact input question files, `case_manifest.json` with source provenance and available reference answers, a console log for each case, and each case's `trace.log`, `report.json`, configurations, caches, and `api_calls.jsonl`. The transcript retains complete model requests and responses with the API key redacted. It contains the question and retrieved content; these local records are ignored by Git. Reference answers are not automatically scored; `PASS` reflects workflow checks only.

Save a question as a UTF-8 text file, including any choices labeled `A.`, `B.`, etc.:

```bash
./run_smoke.command --question-file question.txt
```

Use the same option with `--preflight` for a local setup check. Custom questions are not scored against either example's reference answer. For choices, request `FINAL_ANSWER: <letter>` in the input. The core Python interface is `Agent.run_turn(user_text)` in `models/agent/runtime/agent/agent.py`.

Fill-in and short-answer questions are supported. Put the question and any blanks in the text file, omit answer choices, and request `FINAL_ANSWER: <short answer>` (including units when applicable). Run `./run_smoke.command --question-file question.txt --answer-type open`. This explicitly treats the answer as text even if the question contains lettered background lists; REC compares two proposed answer texts. The default `--answer-type auto` infers the type from displayed choices; `--answer-type mcq` requires at least two choices. Open-answer checks validate format and workflow, not correctness against a reference answer.

## Configuration

`configs/smoke.yaml` is the configuration used by the runnable entry point:

- `deepseek-flash`, temperature `0`, and thinking explicitly disabled.
- Local BM25 retrieval with model reranking and automatically filtered drug, trial, regulatory, and web tools.
- At most one REC round, one REC retrieval attempt, and three tool calls.
- Limits of 14 agent model calls, 16 actual model requests, 120,000 cumulative tokens, and 300 seconds.

These are defaults, not fixed requirements. The token threshold is checked before another agent call; it is not a hard billing cap. Model requests inside tools and transport retries count toward `max_api_requests` (default 16). Use an endpoint you trust with the question and retrieved evidence.

Core defaults are in `models/agent/runtime/config/`. `models/agent/config.yaml` retains higher-reasoning reference settings and configures the default local reranker. The runnable entry point overrides those settings with its low-cost profile.

### Model, thinking, and request settings

Settings at the **top level** of `configs/smoke.yaml` configure the shared model client:

| Setting | Default | Meaning |
| --- | --- | --- |
| `model` | `deepseek-flash` | Provider model ID; CLI override: `--model`. |
| `base_url` | `https://api.deepseek.com` | API endpoint; CLI override: `--base-url`. |
| `api_key_env` | `DEEPSEEK_API_KEY` | Key environment variable; CLI override: `--api-key-env`. OpenAI-compatible mode defaults to `API_KEY`. Keys may also be entered at the hidden prompt. |
| `thinking_enabled` | `false` | DeepSeek thinking switch. `--thinking on` or `--thinking off` overrides it and the structured-JSON thinking switch together. |
| `reasoning_effort` | `""` | Optional provider-supported effort string; CLI override: `--reasoning-effort`. Empty means omit the parameter. With DeepSeek style, a nonempty effort requires thinking enabled. |
| `temperature` | `0` | Sampling temperature, passed to the API. The chosen model must support the value. |
| `request_timeout_seconds` | `60` | Timeout passed to each model request. |
| `max_retries` | `0` | Additional transport attempts for retryable errors; distinct from JSON repair retries. Every attempt counts against the request limit. |
| `max_api_requests` | `16` | Per-case maximum actual model requests across the agent and tool-internal clients, including transport retries. |

For example, all launchers accept:

```bash
./run_three.command --thinking off
./run_three.command --thinking on --reasoning-effort high
```

The second command assumes the selected provider/model accepts `high`; the runner forwards effort strings rather than translating them. Accepted values and reasoning support are provider-specific. An unsupported setting is reported as an API error. Enabling thinking does not raise output-token, request, round, or time limits automatically; allow enough output tokens for the provider's reasoning and final response.

`agent_config_overrides.structured_json_thinking_enabled` separately controls planner, relation-screening, and initial reflection JSON calls when editing YAML. It defaults to `false` and requires the top-level thinking switch to be on if enabled. Debate phases and label/page extraction inherit the shared client's thinking setting. Local reranking explicitly keeps thinking off and omits effort in DeepSeek mode. CLI `--thinking` sets both switches together for the agent stages.

With `--api-style openai`, the runner omits DeepSeek's `thinking` extension and rejects `--thinking`; use a provider-supported `--reasoning-effort` if available. Omitting effort leaves reasoning behavior to the provider. This mode cannot promise to disable reasoning for every model.

Command-line model/API/thinking options override YAML. `--thinking off` also clears an inherited effort setting unless an effort is explicitly supplied; explicitly combining DeepSeek thinking-off with a nonempty effort is rejected. Each run saves the effective `llm.yaml`; `report.json` includes `llm_settings`, and each `api_calls` entry records the actual thinking mode and requested effort. `thinking_explicitly_disabled` is informational and marked N/A for thinking-enabled or provider-default runs; a valid thinking-enabled run is not failed merely for using thinking.

### REC and debate rounds

For all launchers, edit the existing `agent_config_overrides` block in `configs/smoke.yaml`. The current settings are:

```yaml
agent_config_overrides:
  enable_reflection: true
  enable_rec_debate: true
  debate_max_rounds: 1
  debate_max_retrievals: 1
  debate_max_tokens: 1200
  debate_max_retries: 1
```

Merge these fields into the existing block; do not replace its other settings or create a second block.

| Setting | Meaning |
| --- | --- |
| `debate_max_rounds` | Maximum full REC rounds. One round contains up to two exchanges: H1 challenges H0, then H0 challenges H1. Each exchange includes challenge, defense, and a position update. |
| `debate_max_retrievals` | Maximum targeted retrieval attempts across the entire debate, not per round. One web-search attempt can include a source-page visit; both count against tool-call budgets. |
| `debate_max_tokens` | Output-token limit per debate phase, not the total debate token budget. |
| `debate_max_retries` | Additional attempts to repair an invalid response in each phase. `1` permits one repair attempt after the initial response. |
| `enable_reflection` / `enable_rec_debate` | Both must be `true` for REC debate. Setting `enable_reflection: false` skips REC; setting only `enable_rec_debate: false` selects the older reflection path. |

For example, changing `debate_max_rounds` to `2` permits up to four exchanges. These are ceilings: convergence, withdrawal, invalid responses, or execution limits can stop the debate earlier. Changing the round count does not change the model's thinking settings.

REC starts with an initial assessment that selects the alternative hypothesis. A full round then uses up to six phase calls, plus any repair calls and a defender reassessment when retrieval adds evidence. Raising rounds may require more room in `max_llm_calls`, `max_total_tokens`, `deadline_seconds`, and the tool budgets. Also adjust the top-level `max_api_requests` when necessary; raising `max_llm_calls` alone does not change the actual-request limit.

`max_rounds` controls the main agent loop, not REC. In debate mode, `reflection_max_passes` is an enable/disable gate (`0` skips REC); use `debate_max_rounds` to set debate rounds.

Inspect `report.json`: `debate_state.completed_exchanges` counts completed directional exchanges, `debate_history` records each phase's round and side, and `debate_state.stop_reason` explains why execution ended. Coverage fields separately show role reversal, retrieval execution, new evidence, and reassessment. Each case in a three-case run has an independent budget and saved configuration.

### Stage outputs and shared budgets

The following settings belong in the existing **`agent_config_overrides` block**, rather than at the YAML top level:

| Settings | Defaults | Meaning |
| --- | --- | --- |
| `planner_max_tokens` / `planner_retry_max_tokens` / `planner_max_retries` | `3600` / `5000` / `1` | Initial planner output limit, repair output limit, and additional planner attempts. |
| `screening_max_tokens` / `screening_max_retries` | `1800` / `1` | Relation-screening output limit and additional repair attempts. |
| `main_max_tokens` / `final_max_tokens` | `900` / `600` | Main-loop output limit and forced-final output limit after loop/tool limits. |
| `reflection_max_tokens` | `1800` | Initial REC assessment output limit; its built-in parse repairs can use a larger limit. |
| `debate_max_tokens` / `debate_max_retries` | `1200` / `1` | Per-debate-phase output limit and additional JSON repair attempts. |
| `max_rounds` | `3` | Main agent loop ceiling, independent of debate rounds. |
| `max_llm_calls` | `14` | Agent-level model-call ceiling, shared by planning, main reasoning, and REC. Tool-internal model requests use the separate top-level actual-request counter. |
| `max_total_tokens` / `deadline_seconds` | `120000` / `300` | Agent token-usage threshold and elapsed-time checks before subsequent operations. In-flight calls are not forcibly cancelled. |
| `max_total_tool_calls` / `max_online_tool_calls` / `max_local_tool_calls` | `3` / `2` / `2` | Shared tool-call ceilings across main reasoning and REC retrieval. |
| `max_tool_turns` / `max_tools_per_turn` | `3` / `1` | Shared tool-use turns and tools per main turn. A REC search plus source visit consumes two calls/turns. |
| `reflection_max_evidence_records` | `8` | Ranked evidence window supplied to REC; the latest records may also be included. Selected record content is preserved in full. |
| `max_model_visible_tool_result_chars` | `8000` | Bound on rendered tool messages in main-loop memory, separate from the full evidence ledger and saved API transcript. |

The runnable entry point also applies a fixed retrieval profile: BM25-only queries, at most three reranking candidates, one reranking worker, a frozen corpus, and run-local caches. These overrides are in `scripts/smoke_one.py:prepare`; changing the base retrieval YAML alone does not replace that profile. The effective retrieval configuration is saved as `search_local.yaml` in each run. For the core Python interface, pass a configured model client and agent settings to `Agent` directly.

### Tool configuration and availability

Tools must be enabled in `agent_config_overrides.enabled_tools` in `configs/smoke.yaml`. At startup and execution, the registry checks local prerequisites and removes unavailable tools from the model's tool list. `web_search` is excluded when its Serper key is missing; REC uses the available tool descriptions rather than assuming web search exists. These checks do not contact remote services or validate keys. Network failures, rejected credentials, missing source records, and exhausted budgets are reported during execution.

| Tool | Configuration |
| --- | --- |
| `search_local` | Included frozen corpus and `models/agent/runtime/config/search_local.yaml`. The runnable entry point uses BM25 and the selected model for reranking; no embedding key is needed. |
| `get_all_info_by_drug` | Internet access to OpenFDA and Open Targets; label selection uses the selected model API. PrimeKG is off by default; enabling `use_primekg` additionally requires its optional PyTDC package and data. |
| `web_search` | A separate Serper account/API key, supplied as `SERPER_API_KEY`. A DeepSeek key cannot authenticate to Serper. |
| `web_visit` | A known public HTTP(S) URL, internet access to Jina or the source page, and the selected model API for extracting verified quotes. No Serper key is required. |
| `get_trial_info` | Internet access to ClinicalTrials.gov; the request must include a valid NCT identifier. No extra key is required. |
| `get_drug_approvals`, `get_drug_patents`, `get_drug_exlusivities` | Internet access to the FDA Orange Book. Data is downloaded into ignored run caches. Optional `ORANGE_BOOK_DATA_DIR` may point to a local directory containing `products.txt`, `patent.txt`, and `exclusivity.txt`. No extra key is required. |

To enable query-based web search, export `SERPER_API_KEY` in the terminal before starting the launcher. For hidden input in Bash or Zsh:

```bash
export SERPER_API_KEY="$(python3 -c 'import getpass; print(getpass.getpass("Serper API key: "))')"
./run_three.command --preflight
./run_three.command
```

The launchers inherit environment variables; `.env` files are not loaded automatically. Keep keys out of tracked YAML files. Without a Serper key, known-URL visits and the other configured tools remain usable, but general web search is unavailable. Optional dense retrieval additionally needs `OPENAI_API_KEY` and the settings in `models/text_embedding_3_large/config.yaml`.

Each run saves `tool_availability.json` and includes the availability map in `report.json`. REC web-search retrieval inspects the first returned source with `web_visit` when the remaining budgets permit; search snippets alone are recorded as `discovery_only`. Reports distinguish requested, executed, and evidence-returning retrievals. A blocked/failed requested retrieval is marked incomplete rather than counted as a completed branch. Early convergence can still legitimately skip retrieval or role reversal.

### Other API providers

DeepSeek is the default. To use another **OpenAI-compatible Chat Completions** service, supply its base URL and model ID:

```bash
./run_smoke.command --api-style openai \
  --base-url https://your-provider.example/v1 \
  --model YOUR_MODEL_ID \
  --api-key-env YOUR_PROVIDER_API_KEY
```

Replace the URL and model ID with your provider's values. Enter the key at the hidden prompt, or set the named environment variable beforehand. If `--api-key-env` is omitted, this mode reads `API_KEY`. The agent, local reranker, and drug-label field selector share the selected endpoint, model, and key.

This mode sends Chat Completions fields, including `max_tokens`, `temperature`, and `response_format: {"type": "json_object"}` for structured calls. It omits DeepSeek's `thinking` extension; effort is omitted by default but can be supplied through `--reasoning-effort` if the provider supports it. Choose a model that accepts the requested fields and returns assistant text and JSON. Responses-only APIs and provider-native APIs need an adapter. Compatibility has been checked offline; other providers have not been tested live.

## Outputs

Every live run creates `.runs/<timestamp>/trace.log` and `report.json`, together with credential-free configurations and run-local caches. The report includes stage checks, API usage, evidence records, debate history, and the stopping reason.

`PASS` means the workflow checks passed, plus the reference-answer check for a built-in example. It does not establish clinical accuracy. Coverage fields show whether role reversal, targeted retrieval, and reassessment actually occurred; early convergence can leave these branches untested.

## Repository layout

```text
scripts/smoke_one.py               Runnable entry point and report writer
configs/smoke.yaml                 Low-cost profile
models/agent/runtime/agent/        Planning, evidence, debate, memory, and budgets
models/agent/runtime/tools/        Drug, web, trial, regulatory, and local tools
models/agent/runtime/tool_support/ Retrieval and source providers
models/agent/runtime/llm/          Model clients
models/text_embedding_3_large/     Optional dense-embedding client
reference/cureMAS_tools.py         Shared drug-tool helpers
resources/general_knowledge/      16 source-linked records and frozen retrieval indexes
```

The [sample knowledge base](resources/general_knowledge/README.md) contains general source passages. Its genuine stored embeddings and indexes are included. The default local configuration supports dense plus BM25 retrieval with an `OPENAI_API_KEY`; the low-cost entry point disables dense queries and only needs the selected model API. Corpus fingerprints are checked, and runtime rebuilding is disabled.

## Scope

Web search requires `SERPER_API_KEY`; other included tools and optional PrimeKG resources require their own service availability or setup.

Selected evidence records reach REC with their full stored content. Record counts are bounded, but long sources can exhaust the token budget. Citation IDs and response formats are checked structurally; models can still misread evidence or converge prematurely. Earlier live examples preceded the corpus reduction and do not validate live behavior with the current sample. This research software requires independent clinical review.

The repository excludes the question-generation pipeline, benchmark evaluation code, private datasets, local environments, and run traces. `.gitignore` keeps credentials, caches, and local outputs out of normal Git staging.
