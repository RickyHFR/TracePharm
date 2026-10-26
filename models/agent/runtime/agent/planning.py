from __future__ import annotations

import json
import re
from typing import Any


ENTITY_PLAN_SYSTEM_PROMPT = """
You are the planning stage of a medical evidence agent. Build a patient- and
goal-conditioned entity plan before retrieval. Do not answer the question and
do not guess which option is correct.

Integrity rules:
- Use only the user-visible question, context, and choices.
- Never use an answer key, rationale, benchmark metadata, or hidden evidence.
- Treat all answer choices symmetrically. Choice entities may be included as
  candidate entities, but never mark one as correct or preferred.
- A 2-hop entity must be a medically plausible bridge needed to verify the
  decision, not a disguised answer.

Planning rules:
- State the exact clinical/verification goal and the patient-specific goal.
- Extract decision-critical patient anchors and exclusions.
- Patient anchors must be grounded in visible wording. Do not invent absent
  negatives (for example, "no rash" or "no anaphylaxis") unless the stem says so.
- In patient_anchors and constraints, separate visible facts from medical
  interpretation. Do not convert a symptom, adverse event, lab, or history item
  into a confirmed diagnosis, allergy, contraindication, indication, or exclusion
  unless the stem explicitly states that conclusion. Write such items as facts
  whose applicability must be verified by evidence.
- Build a compact entity tree with explicit hop (0, 1, or 2), parent, relation,
  importance, and evidence need. Include drug background, indication, dose,
  contraindication, interaction, population, guideline, trial, or regulatory
  entities when relevant.
- Hop-0 entities are roots: parent_id must be null even if they are medically
  related. Every hop-1/2 entity must reference a parent exactly one hop lower.
- Keep descriptions brief and avoid repeating option text in separate entities.
- Primary entities are directly decision-critical. Secondary entities are
  plausible supporting or 2-hop bridge entities.
- Describe evidence needs without selecting relations or writing search tasks.
  A separate screening stage will select relations after this tree is built.
- In evidence needs for a vignette, include raw unusual anchors
  from the stem such as procedures, devices, foreign bodies, exposures,
  cultured organisms, timelines, lab/imaging findings, or medication context.
  Do not replace these anchors with only an inferred diagnosis label.
- Include concise candidate-option terms needed to compare plausible choices,
  especially when the stem asks for a first-priority medication, source,
  mechanism, trigger, or predisposing factor.
- When a toxicology, overdose, adverse-effect, or altered-mental-status stem
  includes multiple drugs, substances, or medication classes, preserve the
  competing toxidromes and discriminating signs as parallel evidence needs.
  Do not name the decision goal after one inferred substance class unless the
  stem explicitly establishes it.
- When answer choices are multi-action management bundles, decompose the plan
  into the visible action axes (continue/stop a medication, culture/test,
  start/withhold therapy, monitoring threshold, timing). Do not collapse the
  decision goal to only one lab, biomarker, diagnosis label, or safety concern
  if another action axis distinguishes the options.
- For medication counseling, safety, or management questions that include both
  a symptom/event narrative and a medication list, keep at least two competing
  evidence needs when visible: (1) symptom-to-condition or patient-state clues
  that could change medication eligibility, and (2) drug-drug/drug-product
  safety facts. Do not collapse the plan to the first obvious interaction if
  another visible symptom/medication chain could create a stronger exclusion.
- Preserve wording constraints that change the meaning of the decision. For
  example, if the stem asks "in addition to verifying X" or says X is already
  established, record X as a boundary/exclusion and do not turn one candidate
  option's restatement of X into the decision goal.
- Prefer authoritative primary or governed sources: official labels and
  regulators, clinical guidelines, trial registries, and peer-reviewed papers.
- Keep the candidate tree bounded: at most 8 primary and 12 secondary entities.

Return one JSON object only, using this schema:
{
  "task_type": "treatment|safety|diagnosis|mechanism|dose|regulatory|trial|other",
  "decision_goal": "exact judgment to make",
  "patient_goal": "patient-specific therapeutic or verification goal",
  "patient_anchors": [
    {"fact": "visible fact", "type": "demographic|condition|medication|lab|timeline|preference|constraint", "decision_role": "why it matters"}
  ],
  "constraints": ["decision boundary or exclusion"],
  "entities": [
    {
      "id": "e1",
      "name": "entity name",
      "type": "drug|disease|symptom|lab|population|interaction|dose|guideline|trial|regulatory|mechanism|other",
      "importance": "primary|secondary",
      "hop": 0,
      "parent_id": null,
      "relation": "present_in_stem|candidate_option|indication_for|contraindicated_in|interacts_with|monitored_by|governed_by|mechanism_of|other",
      "source_span": "short verbatim span from the visible input, or empty for a derived hop",
      "why_relevant": "decision-specific reason",
      "search_terms": ["canonical term", "synonym"],
      "evidence_need": "fact that must be verified"
    }
  ]
}
""".strip()


RELATION_SCREENING_SYSTEM_PROMPT = """
You screen an existing medical entity tree for the exact visible decision.
Do not answer the question or prefer an answer choice. Use only the visible
case, goals, constraints, candidate entities and candidate relations supplied.
They are planning data, not established medical evidence.

Select only relations whose verification could resolve a decision-critical
uncertainty. Copy relation_id values from candidate_relations; never invent
nodes or edges. Give a decision-specific reason for every selected relation.
Preserve all patient facts, exclusions and competing answer options as context.
Do not treat a visible symptom or lab as a confirmed diagnosis.

Derive at most 6 evidence requirements from the selected relations. Every
selected relation must be covered by a requirement. A requirement may also
verify a hop-0 entity directly, with relation_ids=[]; do not invent an edge
for a root. Non-root target entities must be children of the selected relations
listed in that requirement. Keep ancestor nodes as context, not automatic
retrieval targets. Queries must preserve the patient anchors and the precise
dose, route, timing, monitoring or eligibility distinction being verified.
Use channel local for patient/setting knowledge, online for external label,
regulatory or current facts, or either when both sources can resolve the gap.
If require_evidence=true, return at least one evidence requirement, including
direct verification of a root if needed. Otherwise, if no external evidence is
needed, both arrays may be empty; that differs from an invalid/failed response.

Return one JSON object only:
{
  "selected_relations": [
    {"relation_id":"e1->e2", "reason":"uncertainty relevant to the goal"}
  ],
  "evidence_requirements": [
    {"id":"need1", "relation_ids":["e1->e2"], "entity_ids":["e2"],
     "fact_to_verify":"one decision-critical fact", "channel":"local|online|either",
     "query":"goal-conditioned query with relevant patient anchors",
     "preferred_source_types":["official_label", "guideline"]}
  ]
}
""".strip()


_TASK_TYPES = {"treatment", "safety", "diagnosis", "mechanism", "dose", "regulatory", "trial", "other"}
_ENTITY_TYPES = {
    "drug",
    "disease",
    "symptom",
    "lab",
    "population",
    "interaction",
    "dose",
    "guideline",
    "trial",
    "regulatory",
    "mechanism",
    "other",
}
_CHANNELS = {"online", "local", "either"}
_OPTION_LABEL_RE = re.compile(r"(?<![A-Za-z])([A-Z])[\.\)]\s*")
_GENERIC_OPTION_TERMS = {
    "all",
    "both",
    "either",
    "neither",
    "none",
    "none of the above",
    "all of the above",
    "no treatment",
    "observation",
}
_DRUGISH_SUFFIXES = (
    "ab",
    "afil",
    "azole",
    "caine",
    "cillin",
    "cycline",
    "dipine",
    "epine",
    "gliflozin",
    "gliptin",
    "mab",
    "nib",
    "olol",
    "oxetine",
    "prazole",
    "pril",
    "sartan",
    "setron",
    "statin",
    "terol",
    "tidine",
    "vir",
    "xaban",
    "zepam",
)
_KNOWN_DRUG_WORDS = {
    "acetaminophen",
    "amitriptyline",
    "amikacin",
    "amoxicillin",
    "ampicillin",
    "aspirin",
    "azithromycin",
    "buprenorphine",
    "cefepime",
    "cephalexin",
    "ceftriaxone",
    "ciprofloxacin",
    "clindamycin",
    "cyclobenzaprine",
    "duloxetine",
    "gentamicin",
    "gabapentin",
    "ibuprofen",
    "levofloxacin",
    "meropenem",
    "metronidazole",
    "naloxone",
    "piperacillin",
    "piperacillin-tazobactam",
    "pregabalin",
    "teicoplanin",
    "tobramycin",
    "tylenol",
    "vancomycin",
    "warfarin",
}


def _text(value: Any, limit: int = 600) -> str:
    value = str(value or "").strip()
    value = re.sub(r"\s+", " ", value)
    return value[:limit]


def _visible_question_body(user_text: str) -> str:
    text = str(user_text or "")
    marker = re.search(r"\bQuestion\s*:\s*", text, flags=re.IGNORECASE)
    if marker:
        text = text[marker.end():]
    text = re.sub(
        r"(?is)^.*?Finish with a new line exactly formatted as:\s*FINAL_ANSWER:\s*<[^>]+>\s*",
        "",
        text,
    )
    return text.strip()


def _split_choices(user_text: str) -> list[dict[str, str]]:
    text = _visible_question_body(user_text)
    marker = re.search(r"\b(?:Answer\s+Choices?|Choices)\s*:\s*", text, flags=re.IGNORECASE)
    if marker:
        text = text[marker.end():]
    starts = list(_OPTION_LABEL_RE.finditer(text))
    choices: list[dict[str, str]] = []
    for index, match in enumerate(starts):
        label = match.group(1)
        if label in {"I"}:
            continue
        end = starts[index + 1].start() if index + 1 < len(starts) else len(text)
        option = _text(text[match.end():end], 300)
        option = re.sub(r"\s*(?:FINAL_ANSWER|Question|Context)\s*:.*$", "", option, flags=re.IGNORECASE)
        if option and len(option) <= 300:
            choices.append({"label": label, "text": option})
    return choices[:10]


def _stem_without_choices(user_text: str) -> str:
    text = _visible_question_body(user_text)
    marker = re.search(r"\b(?:Answer\s+Choices?|Choices)\s*:\s*", text, flags=re.IGNORECASE)
    if marker:
        text = text[: marker.start()]
    return _text(text, 4000)


def visible_wording_constraints(user_text: str) -> list[str]:
    """Return stem-only wording boundaries that should constrain planning/retrieval."""

    stem = _stem_without_choices(user_text)
    constraints: list[str] = []
    for match in re.finditer(r"\bin addition to\s+([^,?.;\n]{3,180})", stem, flags=re.IGNORECASE):
        boundary = _text(match.group(1), 180).rstrip(" .")
        if not boundary:
            continue
        constraints.append(
            "Stem wording boundary: 'in addition to "
            + boundary
            + "' means that this purpose is already assumed or insufficient by itself; "
            "the decision goal must resolve the additional purpose beyond that boundary."
        )
    for match in re.finditer(
        r"\b(?:already|previously)\s+(?:shown|verified|established|confirmed)\s+([^,?.;\n]{3,180})",
        stem,
        flags=re.IGNORECASE,
    ):
        boundary = _text(match.group(1), 180).rstrip(" .")
        if not boundary:
            continue
        constraints.append(
            "Stem wording boundary: already verified/established "
            + boundary
            + "; do not make that already-established fact the sole decision goal."
        )
    deduped: list[str] = []
    seen: set[str] = set()
    for item in constraints:
        key = item.lower()
        if key not in seen:
            seen.add(key)
            deduped.append(item)
    return deduped[:4]


_UNVERIFIED_CONCLUSION_RE = re.compile(
    r"\b("
    r"allerg(?:y|ic)|contraindicat(?:e|ed|es|ion)?|preclud(?:e|es|ed)|avoid(?:s|ed)?|"
    r"rule(?:s)?\s+out|rules?\s+in|diagnostic\s+of|diagnoses|confirms?|establish(?:es|ed)?|"
    r"indicates?|qualif(?:y|ies)|requires?|must|should|preferred|first[- ]line|best|safe|unsafe"
    r")\b",
    flags=re.IGNORECASE,
)


def _normalized_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def _is_visible_fact(fact: str, user_text: str) -> bool:
    fact_norm = _normalized_text(fact)
    stem_norm = _normalized_text(_stem_without_choices(user_text))
    if not fact_norm:
        return False
    if fact_norm in stem_norm:
        return True
    tokens = [
        token
        for token in re.findall(r"[a-z0-9]+", fact_norm)
        if len(token) > 2 and token not in {"patient", "female", "male", "with", "after", "before"}
    ]
    if not tokens:
        return False
    hits = sum(1 for token in tokens if token in stem_norm)
    return hits / max(1, len(tokens)) >= 0.75


def _sanitize_planner_role(value: str) -> str:
    value = _text(value, 300)
    if not value:
        return ""
    if _UNVERIFIED_CONCLUSION_RE.search(value):
        return "Visible fact that may affect the decision; verify its direction and applicability against retrieved evidence."
    return value


def _keep_planner_constraint(value: str, user_text: str) -> bool:
    value = _text(value, 300)
    if not value:
        return False
    if _UNVERIFIED_CONCLUSION_RE.search(value):
        return False
    return _is_visible_fact(value, user_text)


def _decision_goal_from_visible_input(user_text: str) -> str:
    stem = _stem_without_choices(user_text)
    questions = re.findall(r"[^?\n]{12,260}\?", stem)
    if questions:
        return _text(questions[-1], 500)
    for pattern in (
        r"\bWhich of the following[^.\n]{10,260}",
        r"\bWhat is[^.\n]{10,260}",
        r"\bWhich[^.\n]{10,260}",
    ):
        match = re.search(pattern, stem, flags=re.IGNORECASE)
        if match:
            return _text(match.group(0), 500)
    return "Answer the user-visible medical question accurately."


def _anchor_type(text: str) -> str:
    lower = text.lower()
    if re.search(r"\b\d{1,3}\s*(?:year|years|yo|y/o)", lower):
        return "demographic"
    if any(token in lower for token in ("taking", "currently takes", "on ", "ibuprofen", "tylenol", "acetaminophen")):
        return "medication"
    if any(token in lower for token in ("negative", "normal", "elevated", "esr", "test", "lab")):
        return "lab"
    if any(token in lower for token in ("does not", "avoid", "contraindicat", "refuses", "does not want")):
        return "constraint"
    return "condition"


def _visible_patient_anchors(user_text: str) -> list[dict[str, str]]:
    stem = _stem_without_choices(user_text)
    patterns = [
        r"\b\d{1,3}\s*(?:year|years|yo|y/o)[- ]old[^.;?\n]{0,140}",
        r"\b(?:complains of|presents with|presenting with|has|history of|diagnosed with|negative for|does not have|tests show|ESR is|taking|currently takes|on|after|following|underwent|received|managed with|treated with|blood cultures grow|echocardiography reveals)\s+[^.;?\n]{2,180}",
        r"\b(?:normal|elevated|low|high)\s+[A-Za-z0-9 /+-]{2,80}",
        r"\bdoes not want to\s+[^.;?\n]{2,120}",
    ]
    anchors: list[dict[str, str]] = []
    seen: set[str] = set()
    for pattern in patterns:
        for match in re.finditer(pattern, stem, flags=re.IGNORECASE):
            fact = _text(match.group(0), 260)
            key = fact.lower()
            if len(fact) < 4 or key in seen:
                continue
            seen.add(key)
            anchors.append(
                {
                    "fact": fact,
                    "type": _anchor_type(fact),
                    "decision_role": "Visible patient-specific anchor that should constrain retrieval and option comparison.",
                }
            )
            if len(anchors) >= 10:
                return anchors
    return anchors


def _task_type_from_visible_input(user_text: str) -> str:
    lower = user_text.lower()
    if any(token in lower for token in ("contraindicat", "avoid", "safe", "toxicity", "adverse", "interaction")):
        return "safety"
    if any(token in lower for token in ("dose", "dosing", "mg", "renal adjustment", "hepatic adjustment")):
        return "dose"
    if any(token in lower for token in ("mechanism", "receptor", "pathway")):
        return "mechanism"
    if any(token in lower for token in ("diagnosis", "diagnostic")):
        return "diagnosis"
    if any(token in lower for token in ("treatment", "best for", "which of the following", "medication", "drug")):
        return "treatment"
    return "other"


def _looks_drug_like(term: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]", "", term.lower())
    if normalized in _KNOWN_DRUG_WORDS:
        return True
    return any(normalized.endswith(suffix) for suffix in _DRUGISH_SUFFIXES + ("floxacin", "mycin", "penem", "xone"))


def _option_components(option_text: str) -> list[str]:
    parts = re.split(r"\s*(?:\+|/|,|;|\band\b|\bor\b| with )\s*", option_text, flags=re.IGNORECASE)
    components: list[str] = []
    for part in parts:
        value = _text(part, 120)
        value = re.sub(r"^\b(?:add|start|stop|continue|use|give)\b\s+", "", value, flags=re.IGNORECASE)
        if not value or value.lower() in _GENERIC_OPTION_TERMS:
            continue
        if len(value) <= 2 and not value.isupper():
            continue
        if value not in components:
            components.append(value)
    return components[:4]


def _stem_drug_entities(user_text: str) -> list[str]:
    stem = _stem_without_choices(user_text)
    terms: list[str] = []
    for match in re.finditer(r"\b[A-Za-z][A-Za-z0-9-]{3,40}\b", stem):
        value = match.group(0)
        if _looks_drug_like(value) and value.lower() not in {item.lower() for item in terms}:
            terms.append(value)
        if len(terms) >= 6:
            break
    return terms


def visible_fallback_plan(user_text: str, error: str = "") -> dict[str, Any]:
    """Build a transparent retrieval plan from visible text only.

    This is deliberately conservative: it uses no benchmark answers, does not
    rank choices, and exists only to keep goal-conditioned retrieval available
    when the LLM planner emits malformed JSON.
    """

    decision_goal = _decision_goal_from_visible_input(user_text)
    anchors = _visible_patient_anchors(user_text)
    anchor_summary = "; ".join(item["fact"] for item in anchors[:5])
    patient_goal = (
        "Resolve the visible medical decision while preserving patient-specific constraints"
        + (f": {anchor_summary}" if anchor_summary else ".")
    )
    choices = _split_choices(user_text)
    entities: list[dict[str, Any]] = []
    seen_names: set[str] = set()

    def add_entity(
        *,
        name: str,
        entity_type: str,
        importance: str,
        hop: int,
        parent_id: str | None,
        relation: str,
        source_span: str,
        why_relevant: str,
        evidence_need: str,
    ) -> str:
        normalized = re.sub(r"\s+", " ", name.strip().lower())
        if not normalized or normalized in seen_names:
            return ""
        if importance == "primary" and sum(1 for item in entities if item["importance"] == "primary") >= 8:
            importance = "secondary"
        if importance == "secondary" and sum(1 for item in entities if item["importance"] == "secondary") >= 12:
            return ""
        seen_names.add(normalized)
        entity_id = f"e{len(entities) + 1}"
        entities.append(
            {
                "id": entity_id,
                "name": _text(name, 160),
                "type": entity_type if entity_type in _ENTITY_TYPES else "other",
                "importance": importance,
                "hop": max(0, min(2, int(hop))),
                "parent_id": parent_id,
                "relation": relation,
                "source_span": _text(source_span, 350),
                "why_relevant": why_relevant,
                "search_terms": [_text(name, 120)],
                "evidence_need": evidence_need,
            }
        )
        return entity_id

    for choice in choices:
        option_text = choice["text"]
        components = _option_components(option_text)
        option_id = ""
        simple_option = len(option_text) <= 90 and any(_looks_drug_like(part) for part in components)
        if simple_option:
            option_id = add_entity(
                name=option_text,
                entity_type="drug",
                importance="secondary",
                hop=0,
                parent_id=None,
                relation="candidate_option",
                source_span=option_text,
                why_relevant="Visible compact answer-choice candidate that must be compared symmetrically.",
                evidence_need="Evidence needed to decide whether this candidate fits the patient-specific goal.",
            )
        for component in components:
            component_is_drug = _looks_drug_like(component)
            if len(component) > 90 and not component_is_drug:
                continue
            add_entity(
                name=component,
                entity_type="drug" if component_is_drug else "other",
                importance="primary" if component_is_drug and not entities else "secondary",
                hop=1,
                parent_id=option_id or None,
                relation="component_of_candidate",
                source_span=component if component in _visible_question_body(user_text) else "",
                why_relevant="Component or named entity from a visible candidate option.",
                evidence_need="Verify indication, safety, contraindications, or guideline role for this patient context.",
            )

    for term in _stem_drug_entities(user_text):
        add_entity(
            name=term,
            entity_type="drug",
            importance="primary" if not entities else "secondary",
            hop=0,
            parent_id=None,
            relation="present_in_stem",
            source_span=term,
            why_relevant="Visible medication or drug-like entity from the question stem.",
            evidence_need="Verify how this entity affects the patient-specific decision.",
        )

    choice_text = "; ".join(f"{item['label']}. {item['text']}" for item in choices[:4])
    all_choice_terms = "; ".join(f"{item['label']}. {item['text']}" for item in choices[:10])
    entity_ids = [item["id"] for item in entities[:8]]
    entity_terms = "; ".join(item["name"] for item in entities[:10])
    local_query = _text(
        " ".join(
            part
            for part in [
                decision_goal,
                anchor_summary,
                f"key entities: {entity_terms}" if entity_terms else "",
                f"choices: {all_choice_terms or choice_text}" if (all_choice_terms or choice_text) else "",
            ]
            if part
        ),
        700,
    )
    guideline_entities = ", ".join(item["name"] for item in entities[:5])
    online_query = _text(
        " ".join(
            part
            for part in [
                guideline_entities,
                "guideline label evidence",
                anchor_summary,
                decision_goal,
            ]
            if part
        ),
        400,
    )
    search_tasks: list[dict[str, Any]] = []
    if local_query:
        search_tasks.append(
            {
                "channel": "local",
                "query": local_query,
                "goal": _text(patient_goal, 400),
                "entity_ids": entity_ids,
                "preferred_source_types": ["guideline", "peer_reviewed", "official_label"],
            }
        )
    if online_query and online_query != local_query:
        search_tasks.append(
            {
                "channel": "online",
                "query": online_query,
                "goal": _text(f"Find authoritative evidence for: {decision_goal}", 400),
                "entity_ids": entity_ids,
                "preferred_source_types": ["official_label", "guideline", "peer_reviewed"],
            }
        )

    plan = normalize_plan(
        {
            "task_type": _task_type_from_visible_input(user_text),
            "decision_goal": decision_goal,
            "patient_goal": patient_goal,
            "patient_anchors": anchors,
            "constraints": [item["fact"] for item in anchors if item.get("type") == "constraint"][:6],
            "entities": entities,
            "search_tasks": search_tasks,
        },
        user_text=user_text,
    )
    if plan.get("status") == "ok":
        plan["status"] = "heuristic_visible"
        plan["planner_error"] = _text(
            "llm planner output was unusable; deterministic visible-input fallback used"
            + (f": {error}" if error else ""),
            500,
        )
    return plan


def extract_json_object(raw: Any, required_keys: set[str] | tuple[str, ...] | list[str] | None = None) -> dict[str, Any]:
    text = str(raw or "").strip()
    required = {str(item) for item in (required_keys or [])}

    def _accept(value: Any) -> dict[str, Any]:
        if not isinstance(value, dict):
            return {}
        if required and not required.issubset(set(value.keys())):
            return {}
        return value

    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        value = json.loads(text)
        accepted = _accept(value)
        if accepted:
            return accepted
    except Exception:
        pass

    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except Exception:
            continue
        accepted = _accept(value)
        if accepted:
            return accepted
    return {}


def fallback_plan(user_text: str, error: str = "") -> dict[str, Any]:
    return {
        "schema_version": "entity_plan_v1",
        "status": "fallback",
        "task_type": "other",
        "decision_goal": "Answer the user-visible medical question accurately.",
        "patient_goal": "Resolve the requested decision while preserving all patient-specific constraints.",
        "patient_anchors": [],
        "constraints": [],
        "entities": [],
        "search_tasks": [],
        "planner_error": _text(error, 500),
        "visible_input_preview": _text(user_text, 500),
    }


def disabled_plan(reason: str = "planning disabled by ablation config") -> dict[str, Any]:
    """Return an explicit empty state without synthesizing goals or entities."""

    return {
        "schema_version": "entity_plan_v1",
        "status": "disabled",
        "task_type": "",
        "decision_goal": "",
        "patient_goal": "",
        "patient_anchors": [],
        "constraints": [],
        "entities": [],
        "search_tasks": [],
        "planner_error": _text(reason, 500),
        "visible_input_preview": "",
    }


def normalize_plan(payload: Any, user_text: str = "") -> dict[str, Any]:
    if not isinstance(payload, dict):
        return fallback_plan(user_text, "planner output was not a JSON object")

    decision_goal = _text(payload.get("decision_goal"), 500)
    patient_goal = _text(payload.get("patient_goal"), 500)
    if not payload or not decision_goal or not patient_goal:
        missing = []
        if not decision_goal:
            missing.append("decision_goal")
        if not patient_goal:
            missing.append("patient_goal")
        return fallback_plan(
            user_text,
            "planner output missing required field(s): " + ", ".join(missing),
        )

    task_type = _text(payload.get("task_type"), 40).lower()
    if task_type not in _TASK_TYPES:
        task_type = "other"

    anchors: list[dict[str, str]] = []
    for item in list(payload.get("patient_anchors") or [])[:12]:
        if not isinstance(item, dict):
            continue
        fact = _text(item.get("fact"), 300)
        if not fact or not _is_visible_fact(fact, user_text):
            continue
        anchors.append(
            {
                "fact": fact,
                "type": _text(item.get("type"), 60) or "constraint",
                "decision_role": _sanitize_planner_role(item.get("decision_role")),
            }
        )

    constraints: list[str] = []
    for item in list(payload.get("constraints") or [])[:12]:
        value = _text(item, 300)
        if value and _keep_planner_constraint(value, user_text) and value not in constraints:
            constraints.append(value)
    for value in visible_wording_constraints(user_text):
        if value and value not in constraints:
            constraints.append(value)

    raw_entities = list(payload.get("entities") or [])
    visible_normalized = re.sub(r"\s+", " ", str(user_text or "").lower())
    entities: list[dict[str, Any]] = []
    entity_ids: set[str] = set()
    primary_count = 0
    secondary_count = 0
    for item in raw_entities:
        if not isinstance(item, dict):
            continue
        name = _text(item.get("name"), 160)
        if not name:
            continue
        relation = _text(item.get("relation"), 80) or "other"
        importance = _text(item.get("importance"), 20).lower()
        importance = importance if importance in {"primary", "secondary"} else "secondary"
        if importance == "primary":
            if primary_count >= 8:
                continue
            primary_count += 1
        else:
            if secondary_count >= 12:
                continue
            secondary_count += 1
        entity_id = _text(item.get("id"), 40) or f"e{len(entities) + 1}"
        if entity_id in entity_ids:
            entity_id = f"e{len(entities) + 1}"
        entity_ids.add(entity_id)
        entity_type = _text(item.get("type"), 40).lower()
        if entity_type not in _ENTITY_TYPES:
            entity_type = "other"
        try:
            hop = max(0, min(2, int(item.get("hop", 0))))
        except Exception:
            hop = 0
        source_span = _text(item.get("source_span"), 350)
        visible_name = re.sub(r"\s+", " ", name.lower()) in visible_normalized
        visible_span = bool(source_span and re.sub(r"\s+", " ", source_span.lower()) in visible_normalized)
        if visible_span:
            provenance_status = "verified_visible_span"
        elif visible_name:
            provenance_status = "verified_visible_name"
        elif hop > 0:
            provenance_status = "derived_requires_parent"
        else:
            provenance_status = "unverified_visible_provenance"
        entities.append(
            {
                "id": entity_id,
                "name": name,
                "type": entity_type,
                "importance": importance,
                "hop": hop,
                "parent_id": None if hop == 0 or item.get("parent_id") in {None, "", "null"} else _text(item.get("parent_id"), 40),
                "relation": relation,
                "source_span": source_span if visible_span else "",
                "provenance_status": provenance_status,
                "why_relevant": _sanitize_planner_role(item.get("why_relevant")),
                "search_terms": [
                    _text(value, 120)
                    for value in list(item.get("search_terms") or [])[:6]
                    if _text(value, 120)
                ],
                "evidence_need": _text(item.get("evidence_need"), 350),
            }
        )

    valid_entity_ids = {item["id"] for item in entities}
    for item in entities:
        if item["parent_id"] not in valid_entity_ids:
            item["parent_id"] = None
        if item.get("provenance_status") == "derived_requires_parent" and item["parent_id"] is None:
            item["provenance_status"] = "unverified_derived_without_parent"

    search_tasks: list[dict[str, Any]] = []
    for item in list(payload.get("search_tasks") or [])[:6]:
        if not isinstance(item, dict):
            continue
        query = _text(item.get("query"), 400)
        goal = _text(item.get("goal"), 400)
        if not query and not goal:
            continue
        channel = _text(item.get("channel"), 20).lower()
        if channel not in _CHANNELS:
            channel = "either"
        task_entity_ids = [
            _text(value, 40)
            for value in list(item.get("entity_ids") or [])[:8]
            if _text(value, 40) in valid_entity_ids
        ]
        source_types = []
        for value in list(item.get("preferred_source_types") or [])[:6]:
            normalized = _text(value, 80)
            if normalized and normalized not in source_types:
                source_types.append(normalized)
        search_tasks.append(
            {
                "channel": channel,
                "query": query,
                "goal": goal,
                "entity_ids": task_entity_ids,
                "preferred_source_types": source_types,
            }
        )

    return {
        "schema_version": "entity_plan_v1",
        "status": "ok",
        "task_type": task_type,
        "decision_goal": decision_goal,
        "patient_goal": patient_goal,
        "patient_anchors": anchors,
        "constraints": constraints,
        "entities": entities,
        "search_tasks": search_tasks,
        "planner_error": "",
    }


def screening_candidates(plan: dict[str, Any]) -> dict[str, Any]:
    """Expose only well-formed tree paths; root facts do not invent edges."""
    entities = list(plan.get("entities") or [])
    by_id = {item["id"]: item for item in entities}
    if len(by_id) != len(entities):
        raise ValueError("candidate tree contains duplicate entity IDs")

    def valid_path(entity_id: str, visited: set[str]) -> bool:
        if entity_id in visited or entity_id not in by_id:
            return False
        node = by_id[entity_id]
        parent_id = node.get("parent_id")
        hop = node.get("hop", 0)
        if hop == 0:
            return parent_id is None
        return (
            hop in {1, 2}
            and parent_id in by_id
            and by_id[parent_id].get("hop", 0) == hop - 1
            and valid_path(parent_id, visited | {entity_id})
        )

    valid = [item for item in entities if valid_path(item["id"], set())]
    relations = [
        {
            "relation_id": f"{item['parent_id']}->{item['id']}",
            "parent_id": item["parent_id"],
            "child_id": item["id"],
            "relation": item.get("relation", "other"),
        }
        for item in valid if item.get("parent_id") is not None
    ]
    return {"candidate_entities": valid, "candidate_relations": relations}


def screening_failure_plan(plan: dict[str, Any], error: str) -> dict[str, Any]:
    """Keep patient context but do not reuse unscreened retrieval tasks."""
    return {
        **plan,
        "entities": [dict(item) for item in plan.get("entities", []) if item.get("hop") == 0],
        "search_tasks": [],
        "selected_relations": [],
        "evidence_requirements": [],
        "screening_status": "failed",
        "screening_error": _text(error, 1000),
    }


def normalize_relation_screening(payload: Any, plan: dict[str, Any]) -> dict[str, Any]:
    """Validate selected edges and derive all search tasks from evidence needs."""
    if not isinstance(payload, dict):
        raise ValueError("relation screening must return a JSON object")

    def array(obj: dict[str, Any], key: str) -> list[Any]:
        value = obj.get(key)
        if not isinstance(value, list):
            raise ValueError(f"{key} must be an array")
        return value

    candidates = screening_candidates(plan)
    by_id = {item["id"]: item for item in candidates["candidate_entities"]}
    edges = {item["relation_id"]: item for item in candidates["candidate_relations"]}
    selected: dict[str, dict[str, Any]] = {}
    for item in array(payload, "selected_relations"):
        if not isinstance(item, dict):
            raise ValueError("selected relation must be an object")
        relation_id = _text(item.get("relation_id"), 100)
        reason = _text(item.get("reason"), 600)
        if relation_id not in edges or relation_id in selected or not reason:
            raise ValueError(f"invalid, duplicate or unexplained relation: {relation_id}")
        selected[relation_id] = {**edges[relation_id], "reason": reason}

    raw_requirements = array(payload, "evidence_requirements")
    if len(raw_requirements) > 6:
        raise ValueError("at most 6 evidence requirements are allowed")
    requirements: list[dict[str, Any]] = []
    tasks: list[dict[str, Any]] = []
    covered: set[str] = set()
    need_ids: set[str] = set()
    target_ids: set[str] = set()
    for item in raw_requirements:
        if not isinstance(item, dict):
            raise ValueError("evidence requirement must be an object")
        need_id = _text(item.get("id"), 40)
        relation_ids = array(item, "relation_ids")
        entity_ids = array(item, "entity_ids")
        if not need_id or need_id in need_ids:
            raise ValueError("evidence requirement IDs must be nonempty and unique")
        if any(not isinstance(value, str) or value not in selected for value in relation_ids):
            raise ValueError(f"{need_id} references an unselected relation")
        if not entity_ids or any(not isinstance(value, str) or value not in by_id for value in entity_ids):
            raise ValueError(f"{need_id} references an invalid target entity")
        if any(
            by_id[entity_id].get("hop", 0) > 0
            and not any(selected[edge]["child_id"] == entity_id for edge in relation_ids)
            for entity_id in entity_ids
        ):
            raise ValueError(f"{need_id} targets a derived entity without its selected relation")
        if any(selected[edge]["child_id"] not in entity_ids for edge in relation_ids):
            raise ValueError(f"{need_id} must target the child of every referenced relation")
        fact = _text(item.get("fact_to_verify"), 600)
        query = _text(item.get("query"), 1200)
        channel = _text(item.get("channel"), 20).lower()
        if not fact or not query or channel not in _CHANNELS:
            raise ValueError(f"{need_id} needs a fact, query and valid channel")
        source_types = array(item, "preferred_source_types")
        if any(not isinstance(value, str) for value in source_types):
            raise ValueError("preferred_source_types must contain strings")
        relation_ids = list(dict.fromkeys(relation_ids))
        entity_ids = list(dict.fromkeys(entity_ids))
        sources = list(dict.fromkeys(_text(value, 80) for value in source_types if value.strip()))[:6]
        requirements.append({
            "id": need_id, "relation_ids": relation_ids, "entity_ids": entity_ids,
            "fact_to_verify": fact, "channel": channel, "query": query,
            "preferred_source_types": sources,
        })
        tasks.append({
            "requirement_id": need_id, "relation_ids": relation_ids,
            "entity_ids": entity_ids, "channel": channel, "query": query,
            "goal": fact, "preferred_source_types": sources,
        })
        need_ids.add(need_id)
        covered.update(relation_ids)
        target_ids.update(entity_ids)
    if covered != set(selected):
        raise ValueError("every selected relation must have an evidence requirement")

    # Retain roots and ancestor paths as context, without retrieving them merely
    # because they connect a selected relation. The full candidate tree is audited
    # separately by the agent and is not fed back into the active retrieval plan.
    active_ids = {entity_id for entity_id, item in by_id.items() if item.get("hop") == 0}
    for entity_id in target_ids:
        while entity_id is not None:
            active_ids.add(entity_id)
            entity_id = by_id[entity_id].get("parent_id")
    return {
        **plan,
        "schema_version": "screened_entity_plan_v1",
        "entities": [dict(item) for item in candidates["candidate_entities"] if item["id"] in active_ids],
        "search_tasks": tasks,
        "selected_relations": list(selected.values()),
        "evidence_requirements": requirements,
        "screening_status": "ok",
        "screening_error": "",
    }


def render_plan_for_model(plan: dict[str, Any]) -> str:
    return (
        "CURRENT PATIENT- AND GOAL-CONDITIONED ENTITY PLAN\n"
        "This plan is retrieval guidance, not an answer. Preserve its patient goal and anchors in every search query.\n"
        "When screening_status is ok, search_tasks are derived from the selected relations and evidence requirements. "
        "Other root and ancestor nodes are case context, not automatic retrieval targets. "
        "When screening_status is failed, relation screening was not completed; any visible-case retrieval is fallback.\n"
        + json.dumps(plan, ensure_ascii=False, indent=2)
    )


def plan_goal_context(plan: dict[str, Any]) -> str:
    parts = [
        _text(plan.get("decision_goal"), 300),
        _text(plan.get("patient_goal"), 300),
    ]
    constraints = [
        _text(item, 220)
        for item in list(plan.get("constraints") or [])[:4]
        if _text(item, 220)
    ]
    if constraints:
        parts.append("Decision boundaries: " + "; ".join(constraints))
    anchors = [
        _text(item.get("fact"), 160)
        for item in list(plan.get("patient_anchors") or [])[:6]
        if isinstance(item, dict)
    ]
    if anchors:
        parts.append("Patient anchors: " + "; ".join(item for item in anchors if item))
    return " | ".join(item for item in parts if item)


def plan_entity_names(plan: dict[str, Any], max_items: int = 12) -> list[str]:
    target_ids = None
    if "screening_status" in plan:
        target_ids = {
            entity_id
            for need in plan.get("evidence_requirements", [])
            for entity_id in need.get("entity_ids", [])
        }
    ranked = sorted(
        [item for item in list(plan.get("entities") or [])
         if isinstance(item, dict) and (target_ids is None or item.get("id") in target_ids)],
        key=lambda item: (0 if item.get("importance") == "primary" else 1, int(item.get("hop", 0) or 0)),
    )
    names: list[str] = []
    for item in ranked:
        name = _text(item.get("name"), 160)
        if name and name not in names:
            names.append(name)
        if len(names) >= max(1, int(max_items)):
            break
    return names
