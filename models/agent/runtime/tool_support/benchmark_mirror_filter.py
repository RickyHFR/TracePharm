from __future__ import annotations

from dataclasses import dataclass
import hashlib
import html
import json
import re
from typing import Any, Mapping, Sequence
from urllib.parse import urlparse
import unicodedata


FILTER_VERSION = "visible-question-overlap-v1"
_SHINGLE_SIZE = 5
_MAX_REFERENCE_TOKENS = 2_000
_MAX_CANDIDATE_TOKENS = 60_000
_TAG_RE = re.compile(r"<[^>]+>")
_TOKEN_RE = re.compile(
    r"[\u3400-\u4dbf\u4e00-\u9fff]|[^\W_]+(?:['\u2019-][^\W_]+)*",
    flags=re.IGNORECASE,
)

# These are generic distribution/answer-site signals, not benchmark names.
# They can lower a content-overlap threshold, but never block a page alone.
_MIRROR_HOST_RULES: tuple[tuple[str, str], ...] = (
    ("huggingface.co", "dataset_host"),
    ("kaggle.com", "dataset_host"),
    ("raw.githubusercontent.com", "raw_repository_host"),
    ("gist.githubusercontent.com", "raw_repository_host"),
    ("quizlet.com", "answer_site_host"),
    ("chegg.com", "answer_site_host"),
    ("coursehero.com", "answer_site_host"),
    ("studocu.com", "answer_site_host"),
    ("brainly.com", "answer_site_host"),
    ("docsity.com", "answer_site_host"),
)
_MIRROR_TEXT_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\b(?:benchmark|dataset|test[ -]?set|evaluation[ -]?set)\b", re.I), "dataset_title"),
    (re.compile(r"\b(?:answer[ -]?key|worked solutions?|quiz answers?)\b", re.I), "answer_title"),
    (re.compile(r"\b(?:multiple[ -]?choice questions?|mcq|exam questions?)\b", re.I), "question_collection_title"),
    (re.compile(r"(?:/datasets?/|\.(?:jsonl?|csv|parquet)(?:$|[?#]))", re.I), "dataset_path"),
)


@dataclass(frozen=True, slots=True)
class VisibleExampleReference:
    """Only fields shown to the evaluated model may be retained here."""

    question_tokens: tuple[str, ...]
    question_compact: str
    choice_tokens: tuple[tuple[str, ...], ...]
    choice_compact: tuple[str, ...]
    question_sha256: str

    @property
    def question_token_count(self) -> int:
        return len(self.question_tokens)

    @property
    def choice_count(self) -> int:
        return len(self.choice_tokens)


def _normalize_text(value: Any) -> str:
    text = html.unescape(str(value or ""))
    text = _TAG_RE.sub(" ", text)
    text = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(text.split())


def _tokens(value: Any, *, maximum: int) -> tuple[str, ...]:
    normalized = _normalize_text(value)
    return tuple(match.group(0).casefold() for match in _TOKEN_RE.finditer(normalized))[:maximum]


def _compact(value: Any, *, maximum: int = 20_000) -> str:
    normalized = _normalize_text(value)
    return "".join(char for char in normalized if char.isalnum())[:maximum]


def _choice_text(value: Any) -> str:
    if isinstance(value, Mapping):
        for key in ("text", "value", "content", "option", "choice"):
            if key in value and str(value.get(key, "") or "").strip():
                return str(value.get(key, "") or "").strip()
        return " ".join(str(item or "").strip() for item in value.values() if str(item or "").strip())
    return str(value or "").strip()


def _visible_choice_texts(raw_choices: Any) -> list[str]:
    choices = raw_choices
    if isinstance(choices, str):
        text = choices.strip()
        try:
            parsed = json.loads(text)
        except Exception:
            parsed = None
        if isinstance(parsed, (list, tuple, dict)):
            choices = parsed
        else:
            lines = [
                re.sub(r"^\s*(?:[A-Z]|\d{1,2})[.):]\s*", "", line).strip()
                for line in text.splitlines()
                if line.strip()
            ]
            return [line for line in lines if line]

    if isinstance(choices, Mapping):
        values = list(choices.values())
    elif isinstance(choices, Sequence) and not isinstance(choices, (str, bytes, bytearray)):
        values = list(choices)
    else:
        values = []
    return [text for text in (_choice_text(item) for item in values) if text]


def build_visible_example_reference(context: Any) -> VisibleExampleReference | None:
    """Build a reference without inspecting answer, gold, rationale, or metadata.

    The explicit key access is intentional: callers may pass richer execution
    context, but this integrity filter is allowed to see only the already
    visible question and choices.
    """

    if not isinstance(context, Mapping):
        return None
    current_example = context.get("current_example")
    if not isinstance(current_example, Mapping):
        return None
    question = str(current_example.get("question", "") or "").strip()
    if not question:
        return None

    question_tokens = _tokens(question, maximum=_MAX_REFERENCE_TOKENS)
    question_compact = _compact(question)
    if len(question_tokens) < 6 and len(question_compact) < 40:
        return None

    choice_tokens: list[tuple[str, ...]] = []
    choice_compact: list[str] = []
    for choice in _visible_choice_texts(current_example.get("choices")):
        tokens = _tokens(choice, maximum=200)
        compact = _compact(choice, maximum=2_000)
        if not tokens and not compact:
            continue
        choice_tokens.append(tokens)
        choice_compact.append(compact)

    canonical_question = " ".join(question_tokens)
    return VisibleExampleReference(
        question_tokens=question_tokens,
        question_compact=question_compact,
        choice_tokens=tuple(choice_tokens),
        choice_compact=tuple(choice_compact),
        question_sha256=hashlib.sha256(canonical_question.encode("utf-8")).hexdigest(),
    )


def _sequence_overlap(reference: tuple[str, ...], candidate: tuple[str, ...], n: int = _SHINGLE_SIZE) -> dict[str, Any]:
    if len(reference) < n or len(candidate) < n:
        return {"coverage": 0.0, "matched_tokens": 0, "longest_span_tokens": 0}

    reference_index: dict[tuple[str, ...], list[int]] = {}
    for index in range(len(reference) - n + 1):
        key = reference[index : index + n]
        positions = reference_index.setdefault(key, [])
        # Repeated boilerplate must not make runtime unbounded.
        if len(positions) < 24:
            positions.append(index)

    covered = bytearray(len(reference))
    diagonal_last: dict[int, int] = {}
    diagonal_run: dict[int, int] = {}
    longest = 0
    for candidate_index in range(len(candidate) - n + 1):
        positions = reference_index.get(candidate[candidate_index : candidate_index + n])
        if not positions:
            continue
        for reference_index_value in positions:
            for offset in range(reference_index_value, min(len(reference), reference_index_value + n)):
                covered[offset] = 1
            diagonal = candidate_index - reference_index_value
            if diagonal_last.get(diagonal) == candidate_index - 1:
                run = diagonal_run.get(diagonal, n) + 1
            else:
                run = n
            diagonal_last[diagonal] = candidate_index
            diagonal_run[diagonal] = run
            longest = max(longest, run)

    matched = int(sum(covered))
    return {
        "coverage": matched / max(1, len(reference)),
        "matched_tokens": matched,
        "longest_span_tokens": longest,
    }


def _choice_matches(reference: VisibleExampleReference, candidate_tokens: tuple[str, ...], candidate_compact: str) -> int:
    matched = 0
    for tokens, compact in zip(reference.choice_tokens, reference.choice_compact, strict=False):
        # One-word drug names and short common phrases are deliberately not
        # distinctive enough to count as mirror evidence.
        if len(tokens) >= 4:
            overlap = _sequence_overlap(tokens, candidate_tokens, n=min(4, len(tokens)))
            is_match = bool(
                overlap["coverage"] >= 0.75
                and overlap["matched_tokens"] >= 4
                and overlap["longest_span_tokens"] >= 4
            )
        elif len(tokens) >= 2 and len(compact) >= 16:
            is_match = compact in candidate_compact
        else:
            is_match = False
        if is_match:
            matched += 1
    return matched


def _mirror_signals(url: str, title: str) -> list[str]:
    signals: list[str] = []
    parsed = urlparse(str(url or ""))
    hostname = str(parsed.hostname or "").casefold().rstrip(".")
    for host, reason in _MIRROR_HOST_RULES:
        if hostname == host or hostname.endswith("." + host):
            signals.append(reason)
    title_and_path = f"{str(title or '')[:1_500]} {parsed.path} {parsed.query}"
    for pattern, reason in _MIRROR_TEXT_PATTERNS:
        if pattern.search(title_and_path):
            signals.append(reason)
    return sorted(set(signals))


def inspect_candidate(
    content: Any,
    reference: VisibleExampleReference | None,
    *,
    url: str = "",
    title: str = "",
    candidate_type: str,
) -> dict[str, Any]:
    raw_content = str(content or "")
    content_sha256 = hashlib.sha256(raw_content.encode("utf-8", errors="ignore")).hexdigest()
    if reference is None:
        return {
            "blocked": False,
            "content_sha256": content_sha256,
            "reasons": [],
            "signals": {},
            "candidate_type": candidate_type,
        }

    candidate_tokens = _tokens(raw_content, maximum=_MAX_CANDIDATE_TOKENS)
    candidate_compact = _compact(raw_content)
    question_overlap = _sequence_overlap(reference.question_tokens, candidate_tokens)
    choice_match_count = _choice_matches(reference, candidate_tokens, candidate_compact)
    mirror_signals = _mirror_signals(url=url, title=title)

    question_tokens = reference.question_token_count
    coverage = float(question_overlap["coverage"])
    matched_tokens = int(question_overlap["matched_tokens"])
    longest = int(question_overlap["longest_span_tokens"])
    reasons: list[str] = []

    # High coverage of a sufficiently long complete question is the core
    # label-free mirror signal.  Ordinary pages sharing short clinical phrases
    # do not approach this combination of coverage and token count.
    if question_tokens >= 12 and coverage >= 0.72 and matched_tokens >= 12:
        reasons.append("high_complete_question_sequence_overlap")

    # A copied question accompanied by multiple substantial options is even
    # more characteristic of a benchmark/quiz mirror.  The question component
    # remains mandatory so a page listing several drugs is not blocked.
    if question_tokens >= 8 and coverage >= 0.42 and matched_tokens >= 8 and choice_match_count >= 2:
        reasons.append("question_plus_multiple_choice_overlap")
    elif question_tokens >= 10 and coverage >= 0.62 and matched_tokens >= 10 and choice_match_count >= 1:
        reasons.append("question_plus_choice_overlap")

    # Generic dataset/answer-site signals can support a moderately truncated
    # copy, but cannot trigger a block without a long question sequence.
    if mirror_signals and question_tokens >= 10 and matched_tokens >= 10 and (coverage >= 0.38 or longest >= 12):
        reasons.append("mirror_distribution_signal_with_long_question_overlap")

    if (
        6 <= question_tokens < 12
        and coverage >= 0.85
        and matched_tokens >= 6
        and (choice_match_count >= 2 or bool(mirror_signals))
    ):
        reasons.append("short_complete_question_with_supporting_mirror_signal")

    # For punctuation-poor CJK text, token shingles already operate at the
    # character level.  This compact check only handles short, fully copied
    # questions and still requires either options or a mirror-distribution cue.
    if (
        question_tokens < 12
        and len(reference.question_compact) >= 40
        and reference.question_compact in candidate_compact
        and (choice_match_count >= 2 or bool(mirror_signals))
    ):
        if "short_complete_question_with_supporting_mirror_signal" not in reasons:
            reasons.append("short_complete_question_with_supporting_mirror_signal")

    blocked = bool(reasons)
    return {
        "blocked": blocked,
        "content_sha256": content_sha256,
        "reasons": reasons,
        "signals": {
            "question_coverage": round(coverage, 4),
            "matched_question_tokens": matched_tokens,
            "longest_question_span_tokens": longest,
            "matched_substantial_choices": choice_match_count,
            "mirror_distribution_signals": mirror_signals,
        },
        "candidate_type": candidate_type,
    }


def new_filter_audit(reference: VisibleExampleReference | None) -> dict[str, Any]:
    return {
        "version": FILTER_VERSION,
        "enabled": reference is not None,
        "context_available": reference is not None,
        "visible_reference_fields": ["question", "choices"] if reference is not None else [],
        "reference": (
            {
                "question_sha256": reference.question_sha256,
                "question_token_count": reference.question_token_count,
                "choice_count": reference.choice_count,
            }
            if reference is not None
            else {}
        ),
        "assessed_count": 0,
        "filtered_count": 0,
        "filtered_items": [],
    }


def add_audit_decision(audit: dict[str, Any], decision: Mapping[str, Any]) -> None:
    if not audit.get("enabled"):
        return
    audit["assessed_count"] = int(audit.get("assessed_count", 0) or 0) + 1
    if not bool(decision.get("blocked", False)):
        return
    audit["filtered_count"] = int(audit.get("filtered_count", 0) or 0) + 1
    audit.setdefault("filtered_items", []).append(
        {
            "content_sha256": str(decision.get("content_sha256", "") or ""),
            "candidate_type": str(decision.get("candidate_type", "") or ""),
            "reasons": [str(item) for item in list(decision.get("reasons") or [])],
            "signals": dict(decision.get("signals") or {}),
        }
    )


def merge_filter_audits(reference: VisibleExampleReference | None, audits: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    merged = new_filter_audit(reference)
    for audit in audits:
        if not isinstance(audit, Mapping):
            continue
        merged["assessed_count"] += int(audit.get("assessed_count", 0) or 0)
        merged["filtered_count"] += int(audit.get("filtered_count", 0) or 0)
        for item in list(audit.get("filtered_items") or []):
            if isinstance(item, Mapping):
                merged["filtered_items"].append(dict(item))
    return merged
