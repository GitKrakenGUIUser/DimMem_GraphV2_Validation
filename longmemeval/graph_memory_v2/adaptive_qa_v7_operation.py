from __future__ import annotations

import argparse
import json
import math
import os
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .adaptive_qa_v5 import (
    _base_candidate_from_prediction,
    _dense_scores,
    _hypotheses,
    _merge_sources,
    _source_output_row,
    _unique,
)
from .adaptive_qa_v6_lite import (
    _json,
    _norm,
    _rank,
    discover_samples,
    infer_contract_v2,
    slot_balanced_rrf,
)
from .io_utils import normalize_question_type, normalize_text, read_json, write_json
from .prompts_p9_operation import (
    ADJUDICATE_SYSTEM_P9,
    ADJUDICATE_TEMPLATE_P9,
    EXPAND_SYSTEM_P9,
    EXPAND_TEMPLATE_P9,
    SPECIALIST_SYSTEM_P9,
    SPECIALIST_TEMPLATE_P9,
)
from .source_qa_v4 import (
    ABSTENTION,
    CallMeta,
    _idf_scores,
    _parse_clock,
    _parse_date,
    _parse_number,
    _question_and_gold,
    _render,
    _source_index,
    _source_lexical_score,
    _truncate,
    copy_retrieval_variant,
    is_abstention,
    judge_prediction,
    make_client,
    memory_evidence,
    postprocess_candidate,
)


SUPERLATIVE_RE = re.compile(
    r"\b(?:most|least|highest|lowest|largest|smallest|earliest|latest)\b",
    re.IGNORECASE,
)
HABITUAL_RE = re.compile(
    r"\b(?:usually|normally|typically|generally|routine|regularly)\b",
    re.IGNORECASE,
)
DIRECT_LOCATION_RE = re.compile(
    r"^\s*where\s+(?:do|did|does|was|were|is|are)\b",
    re.IGNORECASE,
)
SINGULAR_DIRECT_RE = re.compile(
    r"^\s*(?:what|which|where|who)\b",
    re.IGNORECASE,
)
MULTI_ANSWER_RE = re.compile(
    r"\b(?:and|as well as|along with)\b|[,;].+[,;]",
    re.IGNORECASE,
)
NUMBER_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
}


def infer_contract_v3(question: str, parsed: Dict[str, Any]) -> str:
    """Generic operation contract with ordering fixes and direct audits."""
    q = _norm(question)

    # Keep the stronger P8 classifications for assistant recall and preference.
    base = infer_contract_v2(question, parsed)
    if base in {"assistant_recall", "preference_criteria"}:
        return base

    # Duration must precede generic "how many".
    if re.search(
        r"\b(?:how long|how many days|how many weeks|how many months|"
        r"what time .* arrive|what time .* get there)\b",
        q,
    ):
        return "duration"

    if re.search(
        r"\b(?:which|what)\b.*\b(?:spent|cost|paid|amount|price|distance|"
        r"time|score)\b.*\b(?:most|least|highest|lowest|largest|smallest)\b",
        q,
    ) or re.search(
        r"\b(?:most|least|highest|lowest|largest|smallest)\b.*"
        r"\b(?:money|amount|price|cost|distance|time|score)\b",
        q,
    ):
        return "argmax_or_argmin"

    if HABITUAL_RE.search(question):
        return "habitual_state"

    if re.search(
        r"\b(?:who|which|what)\b.*\b(?:first|earlier|later|before|after)\b",
        q,
    ):
        return "comparison"

    if re.search(
        r"\b(?:earliest to latest|latest to earliest|chronological|"
        r"put .* in order|order of|order the)\b",
        q,
    ):
        return "timeline"

    if re.search(
        r"\b(?:total|combined|altogether|in total|sum of|how much .* altogether)\b",
        q,
    ):
        return "sum"

    if re.search(r"\b(?:how many|number of)\b", q):
        return "enumerate_count"

    if DIRECT_LOCATION_RE.search(question):
        return "direct_relation"

    if SUPERLATIVE_RE.search(question):
        return "argmax_or_argmin"

    return base


def requested_operation(contract: str, question: str) -> str:
    q = _norm(question)
    if contract == "enumerate_count":
        return "count_distinct"
    if contract == "sum":
        return "sum_by_group"
    if contract == "timeline":
        return "timeline"
    if contract == "duration":
        return "duration"
    if contract == "comparison":
        if re.search(r"\b(?:first|earlier|before)\b", q):
            return "compare_dates"
        if re.search(r"\b(?:later|after)\b", q):
            return "compare_dates"
        return "compare_numbers"
    if contract == "argmax_or_argmin":
        return "argmin" if re.search(
            r"\b(?:least|lowest|smallest|earliest)\b",
            q,
        ) else "argmax"
    if contract in {"habitual_state", "state_resolution"}:
        return "state_resolution"
    if contract == "preference_criteria":
        return "preference_profile"
    return "direct"


def needs_p9_audit(
    p5: Dict[str, Any],
    p7: Dict[str, Any],
    p8: Dict[str, Any],
    question: str,
    parsed: Dict[str, Any],
) -> Tuple[bool, List[str]]:
    contract = infer_contract_v3(question, parsed)
    answer = str(p8.get("prediction") or p8.get("answer") or "")
    reasons: List[str] = []

    if is_abstention(answer):
        reasons.append("p8_abstained")
    if p8.get("unfilled_slots"):
        reasons.append("p8_unfilled_slots")
    if not p8.get("support_memory_ids") and not p8.get("support_source_ids"):
        reasons.append("p8_no_support")

    answers = {
        _norm(p5.get("prediction") or p5.get("answer")),
        _norm(p7.get("prediction") or p7.get("answer")),
        _norm(answer),
    }
    answers.discard("")
    if len(answers) > 1:
        reasons.append("candidate_disagreement")

    if contract in {
        "enumerate_count",
        "sum",
        "timeline",
        "duration",
        "comparison",
        "argmax_or_argmin",
        "habitual_state",
        "state_resolution",
        "preference_criteria",
        "direct_relation",
    }:
        reasons.append(f"audit_contract:{contract}")

    # Catch over-complete answers to singular direct questions without auditing
    # every ordinary direct fact.
    if (
        SINGULAR_DIRECT_RE.search(question)
        and not re.search(r"\b(?:all|both|which items|what items|how many)\b", _norm(question))
        and MULTI_ANSWER_RE.search(answer)
    ):
        reasons.append("singular_question_multi_answer")

    reasoning = str(p8.get("reasoning_summary") or "")
    if (
        answer
        and re.search(r"\b(?:earlier|later|less|more|first|last)\b", reasoning, re.I)
        and re.search(r"\b(?:earlier|later|less|more|first|last)\b", answer, re.I)
    ):
        reasons.append("comparison_final_consistency_audit")

    return bool(reasons), reasons


def _parse_question_date(value: str) -> Optional[datetime]:
    match = re.search(r"(\d{4})[/-](\d{2})[/-](\d{2})", str(value or ""))
    if not match:
        return None
    try:
        return datetime(
            int(match.group(1)),
            int(match.group(2)),
            int(match.group(3)),
        )
    except ValueError:
        return None


def _relative_window(
    question: str,
    question_date: str,
) -> Tuple[Optional[datetime], Optional[datetime]]:
    end = _parse_question_date(question_date)
    if end is None:
        return None, None
    q = _norm(question)

    match = re.search(
        r"\b(?:past|last)\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten|"
        r"eleven|twelve)\s+(day|week|month)s?\b",
        q,
    )
    if match:
        raw_n = match.group(1)
        n = NUMBER_WORDS.get(raw_n, int(raw_n) if raw_n.isdigit() else 1)
        unit = match.group(2)
        if unit == "day":
            return end - timedelta(days=n), end
        if unit == "week":
            return end - timedelta(days=7 * n), end
        if unit == "month":
            return end - timedelta(days=31 * n), end

    if re.search(r"\blast month\b", q):
        return end - timedelta(days=31), end
    if re.search(r"\blast week\b", q):
        return end - timedelta(days=7), end
    return None, end


def _memory_date(memory: Dict[str, Any]) -> Optional[datetime]:
    dimension = memory.get("dimension") or {}
    time_data = dimension.get("time") or {}
    for key in ("event_start", "valid_from", "event_end", "raw"):
        parsed = _parse_date(time_data.get(key))
        if parsed is not None:
            return parsed
    provenance = memory.get("provenance") or {}
    for value in provenance.get("source_times") or []:
        parsed = _parse_date(value)
        if parsed is not None:
            return parsed
    return None


def _memory_text(memory: Dict[str, Any]) -> str:
    dimension = memory.get("dimension") or {}
    relation = dimension.get("relation") or {}
    preference = dimension.get("preference") or {}
    entities = dimension.get("entities") or []
    entity_text = " ".join(
        str(entity.get("name") or "")
        for entity in entities
        if isinstance(entity, dict)
    )
    return " ".join(
        [
            str(memory.get("content") or ""),
            str(memory.get("evidence_span") or ""),
            str(dimension.get("topic") or ""),
            str(relation.get("subject") or ""),
            str(relation.get("predicate") or ""),
            str(relation.get("object") or ""),
            str(dimension.get("state_key") or ""),
            str(dimension.get("state_value") or ""),
            str(preference.get("target") or ""),
            str(preference.get("polarity") or ""),
            entity_text,
            " ".join(str(value) for value in dimension.get("keywords") or []),
        ]
    )


def _query_expansion(
    client: Any,
    *,
    question: str,
    question_date: str,
    parsed: Dict[str, Any],
    contract: str,
    meta: CallMeta,
) -> Dict[str, Any]:
    payload, result = client.json(
        _render(
            EXPAND_TEMPLATE_P9,
            question=question,
            question_date=question_date,
            query_json=parsed,
            contract=contract,
        ),
        system=EXPAND_SYSTEM_P9,
        max_tokens=int(os.environ.get("QA_P9_EXPAND_TOKENS", "650")),
    )
    meta.add_result(result)
    if not isinstance(payload, dict):
        payload = {}
    queries = _unique(
        [
            question,
            *(str(value) for value in payload.get("queries") or []),
        ]
    )[: int(os.environ.get("QA_P9_MAX_QUERIES", "6"))]
    return {
        "queries": queries,
        "relation_verbs": _unique(
            str(value) for value in payload.get("relation_verbs") or []
        )[:8],
        "entity_hints": _unique(
            str(value) for value in payload.get("entity_hints") or []
        )[:8],
        "time_scope": payload.get("time_scope") or {},
    }


def v2_evidence_census(
    question: str,
    question_date: str,
    expansion: Dict[str, Any],
    memories: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Rank the complete V2 bank, not only P2 top records."""
    if not memories:
        return []

    queries = expansion.get("queries") or [question]
    scoring_rows = [
        {
            "content": _memory_text(memory),
            "assistant_reply": "",
        }
        for memory in memories
    ]
    idf = _idf_scores(scoring_rows)
    dense_enabled = bool(os.environ.get("QA_V5_EMBEDDING_MODEL", "").strip())
    rrf_k = float(os.environ.get("QA_P9_RRF_K", "60"))
    per_query_k = int(os.environ.get("QA_P9_V2_PER_QUERY_K", "5"))
    global_k = int(os.environ.get("QA_P9_V2_GLOBAL_K", "18"))

    fused: Dict[int, float] = defaultdict(float)
    quota: Set[int] = set()
    reasons: Dict[int, Set[str]] = defaultdict(set)

    for query_index, query in enumerate(queries):
        lexical = [
            _source_lexical_score(query, row, idf)
            for row in scoring_rows
        ]
        dense = (
            _dense_scores(
                [query],
                scoring_rows,
                include_assistant=False,
            )
            if dense_enabled
            else [0.0] * len(scoring_rows)
        )
        lex_rank = _rank(lexical)
        dense_rank = _rank(dense)
        local: List[Tuple[float, int]] = []
        for index in range(len(memories)):
            score = 1.0 / (rrf_k + lex_rank[index])
            if dense_enabled:
                score += 1.0 / (rrf_k + dense_rank[index])
            fused[index] += score
            local.append((score, index))
            reasons[index].add(f"q{query_index}:lex={lex_rank[index]}")
            if dense_enabled:
                reasons[index].add(f"q{query_index}:dense={dense_rank[index]}")
        for _, index in sorted(local, reverse=True)[:per_query_k]:
            quota.add(index)

    start, end = _relative_window(question, question_date)
    candidate_indices = quota.union(
        index
        for index, _ in sorted(
            fused.items(),
            key=lambda item: (-item[1], item[0]),
        )[:global_k]
    )

    output: List[Dict[str, Any]] = []
    for index in sorted(
        candidate_indices,
        key=lambda value: (-fused.get(value, 0.0), value),
    ):
        memory = memories[index]
        event_date = _memory_date(memory)
        in_window: Optional[bool] = None
        if event_date is not None and end is not None:
            in_window = event_date <= end and (
                start is None or event_date >= start
            )
        dimension = memory.get("dimension") or {}
        output.append(
            {
                "memory_id": memory.get("memory_id"),
                "content": _truncate(
                    memory.get("content"),
                    int(os.environ.get("QA_P9_V2_CONTENT_CHARS", "700")),
                ),
                "memory_type": dimension.get("memory_type"),
                "topic": dimension.get("topic"),
                "relation": dimension.get("relation"),
                "state_key": dimension.get("state_key"),
                "state_value": dimension.get("state_value"),
                "state_status": dimension.get("state_status"),
                "preference": dimension.get("preference"),
                "modality": dimension.get("modality"),
                "time": dimension.get("time"),
                "provenance": memory.get("provenance"),
                "event_date": (
                    event_date.strftime("%Y-%m-%d")
                    if event_date is not None
                    else ""
                ),
                "in_requested_window": in_window,
                "rrf_score": fused.get(index, 0.0),
                "selection_reasons": sorted(reasons[index]),
            }
        )

    return output[: int(os.environ.get("QA_P9_V2_CENSUS_K", "22"))]


def preference_or_state_view(
    question: str,
    contract: str,
    memories: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    query_tokens = {
        token
        for token in re.findall(r"[a-z0-9]+", _norm(question))
        if len(token) > 2
    }
    rows: List[Tuple[float, Dict[str, Any]]] = []
    for memory in memories:
        dimension = memory.get("dimension") or {}
        preference = dimension.get("preference") or {}
        state_key = str(dimension.get("state_key") or "")
        text = _memory_text(memory)
        overlap = len(query_tokens.intersection(
            set(re.findall(r"[a-z0-9]+", _norm(text)))
        ))
        score = float(overlap)

        if contract == "preference_criteria":
            if preference.get("target"):
                score += 8.0
            if str(dimension.get("memory_type") or "").casefold() == "profile":
                score += 5.0
        elif contract in {"habitual_state", "state_resolution", "direct_relation"}:
            if state_key:
                score += 6.0
            relation = dimension.get("relation") or {}
            if relation.get("predicate"):
                score += 2.0

        if score <= 0:
            continue
        rows.append(
            (
                score,
                {
                    "memory_id": memory.get("memory_id"),
                    "content": _truncate(memory.get("content"), 700),
                    "state_key": state_key,
                    "state_value": dimension.get("state_value"),
                    "state_status": dimension.get("state_status"),
                    "preference": preference,
                    "relation": dimension.get("relation"),
                    "modality": dimension.get("modality"),
                    "time": dimension.get("time"),
                    "provenance": memory.get("provenance"),
                },
            )
        )
    rows.sort(
        key=lambda item: (
            -item[0],
            str(item[1].get("memory_id") or ""),
        )
    )
    return [
        row for _, row in rows[: int(os.environ.get("QA_P9_PROFILE_K", "12"))]
    ]



def build_operation_plan(
    question: str,
    contract: str,
    expansion: Dict[str, Any],
) -> Dict[str, Any]:
    queries = expansion.get("queries") or [question]
    operation = requested_operation(contract, question)

    if contract == "preference_criteria":
        slots = [
            {
                "name": "positive_preference_criteria",
                "query": queries[0],
                "semantic_type": "preference",
                "required": True,
            },
            {
                "name": "negative_or_avoidance_criteria",
                "query": queries[1] if len(queries) > 1 else question,
                "semantic_type": "preference",
                "required": False,
            },
            {
                "name": "relevant_past_experiences",
                "query": queries[2] if len(queries) > 2 else question,
                "semantic_type": "event",
                "required": False,
            },
        ]
    elif contract in {"enumerate_count", "timeline"}:
        slots = [
            {
                "name": "complete_candidate_event_set",
                "query": query,
                "semantic_type": "event",
                "required": True,
            }
            for query in queries[:4]
        ]
    elif contract in {"argmax_or_argmin", "sum"}:
        slots = [
            {
                "name": "candidate_entities_or_groups",
                "query": queries[0],
                "semantic_type": "entity",
                "required": True,
            },
            {
                "name": "numeric_values_for_each_candidate",
                "query": queries[1] if len(queries) > 1 else question,
                "semantic_type": "number",
                "required": True,
            },
        ]
    elif contract == "comparison":
        slots = [
            {
                "name": "first_comparison_side",
                "query": queries[0],
                "semantic_type": "event",
                "required": True,
            },
            {
                "name": "second_comparison_side",
                "query": queries[1] if len(queries) > 1 else question,
                "semantic_type": "event",
                "required": True,
            },
        ]
    elif contract == "duration":
        slots = [
            {
                "name": "start_time_or_date",
                "query": queries[0],
                "semantic_type": "time",
                "required": True,
            },
            {
                "name": "end_time_or_date",
                "query": queries[1] if len(queries) > 1 else question,
                "semantic_type": "time",
                "required": True,
            },
        ]
    elif contract in {"habitual_state", "state_resolution"}:
        slots = [
            {
                "name": "state_history",
                "query": queries[0],
                "semantic_type": "state",
                "required": True,
            },
            {
                "name": "latest_valid_state",
                "query": queries[1] if len(queries) > 1 else question,
                "semantic_type": "state",
                "required": True,
            },
        ]
    else:
        slots = [
            {
                "name": "exact_requested_relation",
                "query": queries[0],
                "semantic_type": "other",
                "required": True,
            }
        ]

    return {
        "answer_contract": contract,
        "operation": operation,
        "slots": slots,
        "retrieval_queries": queries,
        "relation_verbs": expansion.get("relation_verbs") or [],
        "entity_hints": expansion.get("entity_hints") or [],
        "time_scope": expansion.get("time_scope") or {},
    }


def _specialist_candidate(
    client: Any,
    *,
    question: str,
    question_date: str,
    contract: str,
    parsed: Dict[str, Any],
    prompt_memories: Sequence[Dict[str, Any]],
    validation_memories: Sequence[Dict[str, Any]],
    v2_census: Sequence[Dict[str, Any]],
    source_census: Sequence[Dict[str, Any]],
    profile: Sequence[Dict[str, Any]],
    candidates: Dict[str, Any],
    meta: CallMeta,
) -> Dict[str, Any]:
    payload, result = client.json(
        _render(
            SPECIALIST_TEMPLATE_P9,
            question=question,
            question_date=question_date,
            contract=contract,
            memory_json=list(prompt_memories),
            v2_census_json=list(v2_census),
            source_census_json=list(source_census),
            profile_json=list(profile),
            candidate_json=candidates,
        ),
        system=SPECIALIST_SYSTEM_P9,
        max_tokens=int(os.environ.get("QA_P9_SPECIALIST_TOKENS", "1900")),
    )
    meta.add_result(result)
    if not isinstance(payload, dict):
        payload = {
            "answer": ABSTENTION,
            "answerable": False,
            "operation_data": {
                "type": requested_operation(contract, question),
                "items": [],
                "result": "",
            },
        }

    # Validate cited IDs while avoiding V4's unsafe arithmetic executor.
    calculation = payload.pop("calculation", None)
    parsed_for_grounding = parsed
    if contract != "assistant_recall":
        parsed_for_grounding = {
            **parsed,
            "hypotheses": [
                {**row, "need_assistant_context": False}
                for row in _hypotheses(parsed)
            ],
        }
    grounded = postprocess_candidate(
        payload,
        parsed=parsed_for_grounding,
        question=question,
        memories=validation_memories,
        sources=source_census,
    )
    if calculation is not None:
        grounded["calculation"] = calculation
    grounded.setdefault("operation_data", payload.get("operation_data") or {})
    grounded.setdefault("coverage", payload.get("coverage") or {})
    return grounded


def _supported_items(
    operation_data: Dict[str, Any],
) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    for item in operation_data.get("items") or []:
        if not isinstance(item, dict):
            continue
        if not (
            item.get("support_memory_ids")
            or item.get("support_source_ids")
        ):
            continue
        if str(item.get("status") or "").casefold() in {
            "negated",
            "uncertain",
        }:
            continue
        output.append(dict(item))
    return output


def _number(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)):
        return float(value)
    return _parse_number(value)


def _format_number(value: float) -> str:
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.6f}".rstrip("0").rstrip(".")


def _requested_duration_unit(question: str) -> str:
    q = _norm(question)
    if "month" in q:
        return "months"
    if "week" in q:
        return "weeks"
    if "hour" in q:
        return "hours"
    return "days"


def _duration_result(
    items: Sequence[Dict[str, Any]],
    question: str,
) -> Optional[str]:
    dates: List[datetime] = []
    for item in items:
        parsed = _parse_date(item.get("date") or item.get("value"))
        if parsed is not None:
            dates.append(parsed)
    if len(dates) < 2:
        return None
    days = abs((dates[1].date() - dates[0].date()).days)
    unit = _requested_duration_unit(question)
    if unit == "weeks":
        value = days / 7.0
        return f"{_format_number(value)} weeks"
    if unit == "months":
        first, second = sorted(dates[:2])
        months = (second.year - first.year) * 12 + second.month - first.month
        if second.day < first.day:
            months -= 1
        return f"{max(0, months)} months"
    if unit == "hours":
        return f"{_format_number(days * 24.0)} hours"
    return f"{days} days"


def deterministic_operation_answer(
    candidate: Dict[str, Any],
    *,
    contract: str,
    question: str,
) -> Dict[str, Any]:
    """Force final answer to agree with labeled operation operands."""
    result = dict(candidate)
    operation = dict(result.get("operation_data") or {})
    operation_type = str(
        operation.get("type")
        or requested_operation(contract, question)
    )
    items = _supported_items(operation)
    answer: Optional[str] = None

    if operation_type == "count_distinct":
        labels = {
            _norm(item.get("label"))
            for item in items
            if str(item.get("label") or "").strip()
            and str(item.get("status") or "").casefold() != "planned"
        }
        if labels:
            answer = str(len(labels))

    elif operation_type in {"sum_by_group", "argmax", "argmin"}:
        grouped: Dict[str, float] = defaultdict(float)
        original_label: Dict[str, str] = {}
        for item in items:
            label = str(item.get("label") or "").strip()
            value = _number(item.get("value"))
            if not label or value is None:
                continue
            key = _norm(label)
            grouped[key] += value
            original_label.setdefault(key, label)
        if grouped:
            if operation_type == "sum_by_group" and contract == "sum":
                total = sum(grouped.values())
                unit = str(operation.get("requested_unit") or "").strip()
                answer = f"{_format_number(total)} {unit}".strip()
            else:
                chooser = min if operation_type == "argmin" else max
                selected = chooser(grouped, key=grouped.get)
                answer = original_label[selected]

    elif operation_type == "compare_dates":
        dated: List[Tuple[datetime, str]] = []
        for item in items:
            label = str(item.get("label") or "").strip()
            date_value = _parse_date(item.get("date") or item.get("value"))
            if label and date_value is not None:
                dated.append((date_value, label))
        # Both sides are mandatory; absence means insufficient information.
        if len(dated) >= 2:
            dated.sort(key=lambda pair: pair[0])
            if re.search(r"\b(?:later|after|last)\b", _norm(question)):
                answer = dated[-1][1]
            else:
                answer = dated[0][1]
        else:
            answer = ABSTENTION
            result["answerable"] = False

    elif operation_type == "compare_numbers":
        valued: List[Tuple[float, str]] = []
        for item in items:
            label = str(item.get("label") or "").strip()
            value = _number(item.get("value"))
            if label and value is not None:
                valued.append((value, label))
        if len(valued) >= 2:
            if re.search(r"\b(?:less|lower|smaller|decrease)\b", _norm(question)):
                answer = min(valued)[1]
            else:
                answer = max(valued)[1]
        else:
            answer = ABSTENTION
            result["answerable"] = False

    elif operation_type == "timeline":
        dated = []
        for item in items:
            label = str(item.get("label") or "").strip()
            date_value = _parse_date(item.get("date") or item.get("value"))
            if label and date_value is not None:
                dated.append((date_value, label))
        if dated:
            dated.sort(key=lambda pair: pair[0])
            if re.search(r"\blatest to earliest\b", _norm(question)):
                dated.reverse()
            answer = " → ".join(label for _, label in dated)

    elif operation_type == "duration":
        answer = _duration_result(items, question)

    elif operation_type in {
        "state_resolution",
        "preference_profile",
        "direct",
    }:
        operation_result = str(operation.get("result") or "").strip()
        if operation_result:
            answer = operation_result

    if answer is not None:
        result["answer"] = answer
        result["operation_data"] = {
            **operation,
            "type": operation_type,
            "result": answer,
        }
        result["deterministic_finalizer_applied"] = True
        if answer != ABSTENTION:
            result["answerable"] = True

    return result


def _audit_evidence(
    v2_census: Sequence[Dict[str, Any]],
    source_census: Sequence[Dict[str, Any]],
    profile: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "v2_census": list(v2_census)[:18],
        "source_census": list(source_census)[:12],
        "state_or_preference_view": list(profile)[:10],
    }


def _adjudicate(
    client: Any,
    *,
    question: str,
    contract: str,
    candidates: Dict[str, Any],
    evidence: Dict[str, Any],
    parsed: Dict[str, Any],
    memories: Sequence[Dict[str, Any]],
    sources: Sequence[Dict[str, Any]],
    meta: CallMeta,
) -> Dict[str, Any]:
    payload, result = client.json(
        _render(
            ADJUDICATE_TEMPLATE_P9,
            question=question,
            contract=contract,
            candidate_json=candidates,
            evidence_json=evidence,
        ),
        system=ADJUDICATE_SYSTEM_P9,
        max_tokens=int(os.environ.get("QA_P9_ADJUDICATE_TOKENS", "1050")),
    )
    meta.add_result(result)
    if not isinstance(payload, dict):
        return candidates["p8"]

    parsed_for_grounding = parsed
    if contract != "assistant_recall":
        parsed_for_grounding = {
            **parsed,
            "hypotheses": [
                {**row, "need_assistant_context": False}
                for row in _hypotheses(parsed)
            ],
        }
    grounded = postprocess_candidate(
        payload,
        parsed=parsed_for_grounding,
        question=question,
        memories=memories,
        sources=sources,
    )
    grounded.setdefault("operation_data", payload.get("operation_data") or {})
    return deterministic_operation_answer(
        grounded,
        contract=contract,
        question=question,
    )


def _duration_value_days(value: str) -> Optional[float]:
    text = _norm(value)
    match = re.search(
        r"\b(\d+(?:\.\d+)?|zero|one|two|three|four|five|six|seven|eight|"
        r"nine|ten|eleven|twelve)\s+(day|week)s?\b",
        text,
    )
    if not match:
        return None
    raw = match.group(1)
    number = float(NUMBER_WORDS.get(raw, raw))
    return number * (7.0 if match.group(2) == "week" else 1.0)


def judge_prediction_p9(
    client: Any,
    *,
    question_type: str,
    question: str,
    gold_answer: str,
    prediction: str,
    meta: CallMeta,
) -> Tuple[bool, str, str]:
    gold_days = _duration_value_days(gold_answer)
    prediction_days = _duration_value_days(prediction)
    if (
        gold_days is not None
        and prediction_days is not None
        and abs(gold_days - prediction_days) < 1e-9
    ):
        return (
            True,
            "Deterministic duration equivalence.",
            "deterministic_duration_equivalence",
        )
    return judge_prediction(
        client,
        question_type=question_type,
        question=question,
        gold_answer=gold_answer,
        prediction=prediction,
        meta=meta,
    )


def run_sample(
    sample_dir: Path,
    *,
    retrieval_name: str,
    p5_name: str,
    p7_name: str,
    p8_name: str,
    output_name: str,
    force: bool,
) -> Dict[str, Any]:
    qa_path = sample_dir / "qa_v2" / output_name / "prediction.json"
    judge_path = sample_dir / "judge_v2" / output_name / "judge.json"
    if qa_path.exists() and judge_path.exists() and not force:
        return {
            "status": "existing",
            "sample_dir": str(sample_dir),
            "correct": bool(read_json(judge_path).get("correct")),
        }

    p5_path = sample_dir / "qa_v2" / p5_name / "prediction.json"
    p7_path = sample_dir / "qa_v2" / p7_name / "prediction.json"
    p8_path = sample_dir / "qa_v2" / p8_name / "prediction.json"
    compact_path = sample_dir / "qa_v2" / p5_name / "source_evidence.json"
    for required_path in (p5_path, p7_path, p8_path, compact_path):
        if not required_path.exists():
            raise FileNotFoundError(required_path)

    copy_retrieval_variant(sample_dir, retrieval_name, output_name)
    item = read_json(sample_dir / "input_item.json")
    question, question_date, gold = _question_and_gold(item)
    parsed = read_json(sample_dir / "query_v2" / "parsed_query.json")
    p5_raw = read_json(p5_path)
    p7_raw = read_json(p7_path)
    p8_raw = read_json(p8_path)
    p5 = _base_candidate_from_prediction(p5_raw)
    p7 = _base_candidate_from_prediction(p7_raw)
    p8 = _base_candidate_from_prediction(p8_raw)
    compact = read_json(compact_path)
    source_turns = read_json(sample_dir / "source_turns.json")
    top_records = read_json(
        sample_dir / "retrieval_v2" / retrieval_name / "top_records.json"
    )
    memories = memory_evidence(
        top_records,
        limit=int(os.environ.get("QA_P9_MEMORY_K", "12")),
    )
    all_v2_path = sample_dir / "memory_v2" / "all_memories.json"
    all_v2 = read_json(all_v2_path) if all_v2_path.exists() else []
    validation_memories = (
        memory_evidence(all_v2, limit=len(all_v2))
        if all_v2
        else memories
    )

    audit, risk_reasons = needs_p9_audit(
        p5_raw,
        p7_raw,
        p8_raw,
        question,
        parsed,
    )
    contract = infer_contract_v3(question, parsed)
    final = p8
    specialist: Dict[str, Any] = {}
    adjudicated: Dict[str, Any] = {}
    expansion: Dict[str, Any] = {}
    plan: Dict[str, Any] = {}
    v2_census: List[Dict[str, Any]] = []
    source_census: List[Dict[str, Any]] = []
    profile: List[Dict[str, Any]] = []
    meta = CallMeta()
    choice = "p8"

    if audit:
        client = make_client()
        expansion = _query_expansion(
            client,
            question=question,
            question_date=question_date,
            parsed=parsed,
            contract=contract,
            meta=meta,
        )
        plan = build_operation_plan(question, contract, expansion)
        v2_census = v2_evidence_census(
            question,
            question_date,
            expansion,
            all_v2,
        )
        profile = preference_or_state_view(
            question,
            contract,
            all_v2,
        )
        source_census = slot_balanced_rrf(
            question,
            parsed,
            plan,
            source_turns,
            compact,
        )
        source_context = _merge_sources(
            compact[: int(os.environ.get("QA_P9_COMPACT_K", "8"))],
            source_census,
        )

        specialist = _specialist_candidate(
            client,
            question=question,
            question_date=question_date,
            contract=contract,
            parsed=parsed,
            prompt_memories=memories,
            validation_memories=validation_memories,
            v2_census=v2_census,
            source_census=source_context,
            profile=profile,
            candidates={"p5": p5, "p7": p7, "p8": p8},
            meta=meta,
        )
        specialist = deterministic_operation_answer(
            specialist,
            contract=contract,
            question=question,
        )

        candidate_answers = {
            _norm(candidate.get("answer"))
            for candidate in (p5, p7, p8, specialist)
            if candidate.get("answer")
        }
        if len(candidate_answers) == 1:
            final = specialist if specialist.get("support_source_ids") else p8
            choice = "specialist" if final is specialist else "p8"
        else:
            merged_sources = source_context
            adjudicated = _adjudicate(
                client,
                question=question,
                contract=contract,
                candidates={
                    "p5": p5,
                    "p7": p7,
                    "p8": p8,
                    "specialist": specialist,
                },
                evidence=_audit_evidence(
                    v2_census,
                    source_census,
                    profile,
                ),
                parsed=parsed,
                memories=validation_memories,
                sources=merged_sources,
                meta=meta,
            )
            final = adjudicated
            choice = "adjudicated"

    qa_output = {
        "question_id": item.get("question_id"),
        "question_type": normalize_question_type(
            item.get("question_type") or sample_dir.parent.name
        ),
        "question": question,
        "question_date": question_date,
        "gold_answer": gold,
        "prediction": str(final.get("answer") or ABSTENTION),
        "support_memory_ids": final.get("support_memory_ids") or [],
        "support_source_ids": final.get("support_source_ids") or [],
        "confidence": final.get("confidence") or 0.0,
        "reasoning_summary": (
            final.get("reasoning_summary")
            or final.get("reason")
            or ""
        ),
        "required_slots": final.get("required_slots") or [],
        "unfilled_slots": final.get("unfilled_slots") or [],
        "operation": (
            (final.get("operation_data") or {}).get("type")
            or final.get("operation")
            or "none"
        ),
        "operation_data": final.get("operation_data") or {},
        "answerable": bool(final.get("answerable")),
        "retrieval_name": output_name,
        "source_retrieval_name": retrieval_name,
        "p5_name": p5_name,
        "p7_name": p7_name,
        "p8_name": p8_name,
        "p9_choice": choice,
        "risk_reasons": risk_reasons,
        "contract": contract,
        "meta": {
            "mode": "p9_operation_census",
            **meta.to_dict(),
        },
    }
    qa_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(qa_path, qa_output)
    write_json(
        qa_path.parent / "p9_trace.json",
        {
            "audited": audit,
            "risk_reasons": risk_reasons,
            "contract": contract,
            "expansion": expansion,
            "plan": plan,
            "p5_candidate": p5,
            "p7_candidate": p7,
            "p8_candidate": p8,
            "specialist_candidate": specialist,
            "adjudicated_candidate": adjudicated,
            "choice": choice,
            "v2_census_memory_ids": [
                row.get("memory_id") for row in v2_census
            ],
            "source_census_ids": [
                row.get("source_id") for row in source_census
            ],
            "profile_memory_ids": [
                row.get("memory_id") for row in profile
            ],
        },
    )
    write_json(qa_path.parent / "v2_evidence_census.json", v2_census)
    write_json(qa_path.parent / "source_evidence_census.json", source_census)
    write_json(qa_path.parent / "state_preference_view.json", profile)

    client = make_client()
    judge_meta = CallMeta()
    correct, reason, judge_mode = judge_prediction_p9(
        client,
        question_type=qa_output["question_type"],
        question=question,
        gold_answer=gold,
        prediction=qa_output["prediction"],
        meta=judge_meta,
    )
    judge_output = {
        **qa_output,
        "correct": bool(correct),
        "judge_reason": reason,
        "judge_meta": {
            "mode": judge_mode,
            **judge_meta.to_dict(),
        },
    }
    judge_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(judge_path, judge_output)

    return {
        "status": "ok",
        "question_id": item.get("question_id"),
        "sample_dir": str(sample_dir),
        "correct": bool(correct),
        "audited": audit,
        "choice": choice,
        "contract": contract,
        "prompt_tokens": meta.prompt_tokens,
        "completion_tokens": meta.completion_tokens,
        "v2_census_count": len(v2_census),
        "source_census_count": len(source_census),
    }


def run_all(args: argparse.Namespace) -> Dict[str, Any]:
    run_root = Path(args.run_root)
    samples = discover_samples(run_root)
    workers = len(samples) if args.workers == 0 else max(1, args.workers)
    rows: List[Dict[str, Any]] = []

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                run_sample,
                sample,
                retrieval_name=args.retrieval_name,
                p5_name=args.p5_name,
                p7_name=args.p7_name,
                p8_name=args.p8_name,
                output_name=args.output_name,
                force=args.force,
            ): sample
            for sample in samples
        }
        for index, future in enumerate(as_completed(futures), start=1):
            sample = futures[future]
            try:
                row = future.result()
            except Exception as exc:
                if args.fail_fast:
                    raise
                row = {
                    "status": "failed",
                    "sample_dir": str(sample),
                    "error": repr(exc),
                    "correct": False,
                }
            rows.append(row)
            print(
                f"[p9-operation] {index}/{len(samples)} | {sample.name} | "
                f"{row.get('status')} | contract={row.get('contract', '-')} | "
                f"choice={row.get('choice', '-')}",
                flush=True,
            )

    correct = sum(bool(row.get("correct")) for row in rows)
    summary = {
        "retrieval_name": args.retrieval_name,
        "p5_name": args.p5_name,
        "p7_name": args.p7_name,
        "p8_name": args.p8_name,
        "output_name": args.output_name,
        "n": len(rows),
        "correct": correct,
        "accuracy": correct / len(rows) if rows else 0.0,
        "workers": workers,
        "rows": rows,
    }
    write_json(
        run_root / f"p9_manifest_{args.output_name}.json",
        summary,
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument(
        "--retrieval-name",
        default="graph_active_p2_gate",
    )
    parser.add_argument(
        "--p5-name",
        default="graph_active_p5_source_qa",
    )
    parser.add_argument(
        "--p7-name",
        default="graph_active_p7_adaptive_qa",
    )
    parser.add_argument(
        "--p8-name",
        default="graph_active_p8_lite_generic",
    )
    parser.add_argument(
        "--output-name",
        default="graph_active_p9_operation_census",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()

    summary = run_all(args)
    print(_json(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
