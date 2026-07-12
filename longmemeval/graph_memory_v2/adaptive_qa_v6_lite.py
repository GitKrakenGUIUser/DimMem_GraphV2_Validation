from __future__ import annotations

import argparse
import json
import math
import os
import re
import threading
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
from .io_utils import normalize_question_type, normalize_text, read_json, write_json
from .prompts_p8_lite import (
    PLAN_SYSTEM_P8,
    PLAN_TEMPLATE_P8,
    SPECIALIST_SYSTEM_P8,
    SPECIALIST_TEMPLATE_P8,
    VERIFY_SYSTEM_P8,
    VERIFY_TEMPLATE_P8,
)
from .source_qa_v4 import (
    ABSTENTION,
    CallMeta,
    _adjacent_rows,
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
    question_needs_assistant,
)


HARD_CONTRACTS = {
    "enumerate_count",
    "sum",
    "timeline",
    "duration",
    "state_resolution",
    "comparison",
    "preference_criteria",
    "assistant_recall",
    "insufficient_information_sensitive",
}

EXPLICIT_ASSISTANT_RECALL = re.compile(
    r"\b(?:what|which|how)\s+(?:did|have)\s+you\s+"
    r"(?:say|suggest|recommend|advise|tell|explain|list)\b|"
    r"\b(?:your|the assistant'?s)\s+(?:previous|earlier)\s+"
    r"(?:answer|response|reply|recommendation|advice)\b|"
    r"\bremind me what you\b",
    re.IGNORECASE,
)

FORWARD_ADVICE = re.compile(
    r"\b(?:recommend|suggest|tips?|advice|ideas?|what should i|"
    r"help me (?:choose|decide|improve|find)|do you have any)\b",
    re.IGNORECASE,
)

SUSPICIOUS_REASONING = re.compile(
    r"\b(?:not mentioned|no evidence|could not find|cannot confirm|"
    r"no later information|assuming|must have|likely)\b",
    re.IGNORECASE,
)


def _norm(value: Any) -> str:
    return normalize_text(str(value or ""))


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)


def _answer_dims(parsed: Dict[str, Any]) -> Set[str]:
    return {
        str(row.get("answer_dim") or "").casefold()
        for row in _hypotheses(parsed)
        if row.get("answer_dim")
    }


def _memory_types(parsed: Dict[str, Any]) -> Set[str]:
    return {
        str(value).casefold()
        for row in _hypotheses(parsed)
        for value in row.get("target_memory_types") or []
    }


def _state_keys(parsed: Dict[str, Any]) -> List[str]:
    return _unique(
        str(value)
        for row in _hypotheses(parsed)
        for value in row.get("state_keys") or []
    )


def infer_contract_v2(question: str, parsed: Dict[str, Any]) -> str:
    """Infer generic answer operation without benchmark-category branches."""
    q = _norm(question)
    dims = _answer_dims(parsed)
    memory_types = _memory_types(parsed)

    # Forward-looking advice must not be mistaken for recalling an old assistant
    # message merely because the parser set need_assistant_context.
    if EXPLICIT_ASSISTANT_RECALL.search(question):
        return "assistant_recall"

    if FORWARD_ADVICE.search(question) and (
        "preference" in dims
        or "profile" in memory_types
        or any(
            token in q
            for token in (
                "my preferences",
                "based on what i like",
                "personalized",
                "for me",
            )
        )
    ):
        return "preference_criteria"

    if re.search(
        r"\b(?:total|combined|altogether|in total|sum of|how much .* altogether)\b",
        q,
    ):
        return "sum"

    if re.search(r"\b(?:how many|number of)\b", q):
        # A single product metric is a stated value; autobiographical plural
        # event/item questions require enumeration.
        if re.search(
            r"\b(?:copies|pages|employees|members|miles|minutes|hours|"
            r"ounces|dollars|percent|percentage)\b",
            q,
        ) and not re.search(r"\b(?:did i|have i|i attended|i acquired|i bought|i received)\b", q):
            return "direct_numeric"
        return "enumerate_count"

    if re.search(
        r"\b(?:earliest to latest|latest to earliest|chronological|"
        r"put .* in order|order the)\b",
        q,
    ):
        return "timeline"

    if re.search(
        r"\b(?:how long|how many days|how many weeks|how many months|"
        r"what time .* arrive|what time .* get there)\b",
        q,
    ):
        return "duration"

    if re.search(
        r"\b(?:more or less|increase or decrease|higher or lower|"
        r"before or after|which .* first|who .* first)\b",
        q,
    ):
        return "comparison"

    if _state_keys(parsed) and re.search(
        r"\b(?:current|currently|now|latest|still|where is|what is my)\b",
        q,
    ):
        return "state_resolution"

    if re.search(
        r"\b(?:is there enough information|can this be determined|"
        r"who .* first|which .* first)\b",
        q,
    ):
        return "insufficient_information_sensitive"

    if question_needs_assistant(parsed, question) and EXPLICIT_ASSISTANT_RECALL.search(question):
        return "assistant_recall"

    return "direct_value"


def operator_for_contract(contract: str, question: str) -> str:
    q = _norm(question)
    if contract == "enumerate_count":
        return "count_distinct"
    if contract == "sum":
        return "sum"
    if contract == "timeline":
        return "timeline"
    if contract == "state_resolution":
        return "state_resolution"
    if contract == "comparison":
        return "comparison"
    if contract == "duration":
        if "month" in q:
            return "date_difference_months"
        if "week" in q:
            return "date_difference_weeks"
        if "what time" in q:
            return "time_add_minutes"
        return "date_difference_days"
    return "none"


def reconcile_plan(
    question: str,
    parsed: Dict[str, Any],
    payload: Dict[str, Any],
) -> Dict[str, Any]:
    result = dict(payload or {})
    hint = infer_contract_v2(question, parsed)
    llm_contract = str(result.get("answer_contract") or "")
    if hint in HARD_CONTRACTS:
        result["answer_contract"] = hint
    elif not llm_contract:
        result["answer_contract"] = hint

    expected_operator = operator_for_contract(
        str(result.get("answer_contract") or hint),
        question,
    )
    if expected_operator != "none":
        result["operator"] = expected_operator
    else:
        result.setdefault("operator", "none")

    slots = [
        dict(slot)
        for slot in result.get("slots") or []
        if isinstance(slot, dict)
    ]
    if not slots:
        parser_slots = _unique(
            str(value)
            for row in _hypotheses(parsed)
            for value in row.get("missing_slots") or []
        )
        if parser_slots:
            slots = [
                {
                    "name": slot,
                    "query": f"{question} {slot}",
                    "semantic_type": "other",
                    "required": True,
                }
                for slot in parser_slots
            ]
        else:
            slots = [
                {
                    "name": "answer evidence",
                    "query": question,
                    "semantic_type": "other",
                    "required": True,
                }
            ]

    for slot in slots:
        slot.setdefault("name", "answer evidence")
        slot.setdefault("query", question)
        slot.setdefault("semantic_type", "other")
        slot.setdefault("required", True)
    result["slots"] = slots
    result["retrieval_queries"] = _unique(
        [
            *(str(value) for value in result.get("retrieval_queries") or []),
            *(str(slot.get("query") or "") for slot in slots),
        ]
    )
    result["deterministic_contract_hint"] = hint
    result["contract_overridden"] = (
        hint in HARD_CONTRACTS and hint != llm_contract
    )
    return result


def needs_escalation(
    p5: Dict[str, Any],
    p7: Dict[str, Any],
    question: str,
    parsed: Dict[str, Any],
) -> Tuple[bool, List[str]]:
    contract = infer_contract_v2(question, parsed)
    reasons: List[str] = []
    p7_answer = p7.get("prediction") or p7.get("answer")

    if is_abstention(p7_answer):
        reasons.append("p7_abstained")
    if p7.get("unfilled_slots"):
        reasons.append("p7_unfilled_slots")
    if not p7.get("support_memory_ids") and not p7.get("support_source_ids"):
        reasons.append("p7_no_support")
    if _norm(p5.get("prediction")) != _norm(p7_answer):
        reasons.append("p5_p7_disagree")
    if contract in {
        "enumerate_count",
        "sum",
        "timeline",
        "duration",
        "state_resolution",
        "comparison",
        "preference_criteria",
        "insufficient_information_sensitive",
    }:
        reasons.append(f"audit_contract:{contract}")
    if contract == "assistant_recall":
        # The 100-case report shows this category is already strong; only audit
        # if the candidate itself is weak.
        if is_abstention(p7_answer) or not p7.get("support_source_ids"):
            reasons.append("weak_assistant_recall")
    if (
        p7_answer
        and not is_abstention(p7_answer)
        and SUSPICIOUS_REASONING.search(str(p7.get("reasoning_summary") or ""))
    ):
        reasons.append("concrete_answer_with_uncertain_reasoning")

    return bool(reasons), reasons


def _rank(values: Sequence[float]) -> List[int]:
    ordered = sorted(
        range(len(values)),
        key=lambda index: (-float(values[index]), index),
    )
    ranks = [0] * len(values)
    for rank, index in enumerate(ordered, start=1):
        ranks[index] = rank
    return ranks


def slot_balanced_rrf(
    question: str,
    parsed: Dict[str, Any],
    plan: Dict[str, Any],
    source_turns: Sequence[Dict[str, Any]],
    compact_sources: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Per-slot lexical+dense retrieval with rank fusion and session caps."""
    include_assistant = infer_contract_v2(question, parsed) == "assistant_recall"
    scoring_rows = (
        list(source_turns)
        if include_assistant
        else [{**row, "assistant_reply": ""} for row in source_turns]
    )
    idf = _idf_scores(scoring_rows)

    query_groups: List[Tuple[str, str]] = []
    for index, slot in enumerate(plan.get("slots") or []):
        if isinstance(slot, dict):
            query_groups.append(
                (
                    f"slot_{index}:{slot.get('name', '')}",
                    str(slot.get("query") or question),
                )
            )
    query_groups.extend(
        (f"plan_{index}", str(query))
        for index, query in enumerate(plan.get("retrieval_queries") or [])
        if str(query).strip()
    )
    query_groups.append(("question", question))

    deduped: List[Tuple[str, str]] = []
    seen: Set[str] = set()
    for label, query in query_groups:
        marker = _norm(query)
        if marker and marker not in seen:
            seen.add(marker)
            deduped.append((label, query))
    query_groups = deduped

    rrf_k = float(os.environ.get("QA_P8_RRF_K", "60"))
    per_slot_k = int(os.environ.get("QA_P8_PER_SLOT_K", "3"))
    global_k = int(os.environ.get("QA_P8_GLOBAL_K", "6"))
    fused: Dict[int, float] = defaultdict(float)
    reasons: Dict[int, Set[str]] = defaultdict(set)
    quota_ids: Set[int] = set()

    dense_enabled = bool(os.environ.get("QA_V5_EMBEDDING_MODEL", "").strip())
    for label, query in query_groups:
        lexical_scores = [
            _source_lexical_score(query, row, idf)
            for row in scoring_rows
        ]
        dense_scores = (
            _dense_scores(
                [query],
                scoring_rows,
                include_assistant=include_assistant,
            )
            if dense_enabled
            else [0.0] * len(scoring_rows)
        )
        lexical_ranks = _rank(lexical_scores)
        dense_ranks = _rank(dense_scores)

        local: List[Tuple[float, int]] = []
        for index, row in enumerate(source_turns):
            source_id = int(row.get("source_id") or 0)
            score = 1.0 / (rrf_k + lexical_ranks[index])
            if dense_enabled:
                score += 1.0 / (rrf_k + dense_ranks[index])
            fused[source_id] += score
            local.append((score, source_id))
            reasons[source_id].add(
                f"{label}:lex={lexical_ranks[index]}"
            )
            if dense_enabled:
                reasons[source_id].add(
                    f"{label}:dense={dense_ranks[index]}"
                )

        for _, source_id in sorted(local, reverse=True)[:per_slot_k]:
            quota_ids.add(source_id)
            reasons[source_id].add(f"{label}:quota")

    global_ids = [
        source_id
        for source_id, _ in sorted(
            fused.items(),
            key=lambda item: (-item[1], item[0]),
        )[:global_k]
    ]
    candidate_ids = quota_ids.union(global_ids)
    by_id, by_session = _source_index(source_turns)

    session_cap = int(os.environ.get("QA_P8_SESSION_CAP", "4"))
    per_session: Dict[str, int] = defaultdict(int)
    selected: List[int] = []
    for source_id in sorted(
        candidate_ids,
        key=lambda value: (-fused.get(value, 0.0), value),
    ):
        row = by_id.get(source_id)
        if row is None:
            continue
        session_id = str(row.get("session_id") or "")
        if per_session[session_id] >= session_cap:
            continue
        selected.append(source_id)
        per_session[session_id] += 1

    adjacent_top = int(os.environ.get("QA_P8_ADJACENT_TOP", "3"))
    adjacent_radius = int(os.environ.get("QA_P8_ADJACENT_RADIUS", "1"))
    expanded: Dict[int, Dict[str, Any]] = {}
    for rank, source_id in enumerate(selected, start=1):
        row = by_id[source_id]
        expanded[source_id] = row
        reasons[source_id].add(f"rrf_rank_{rank}")
        if rank <= adjacent_top:
            for adjacent in _adjacent_rows(
                row,
                by_session,
                adjacent_radius,
            ):
                adjacent_id = int(adjacent.get("source_id") or 0)
                expanded[adjacent_id] = adjacent
                reasons[adjacent_id].add(f"adjacent_to_{rank}")

    compact_ids = {
        int(row.get("source_id") or 0)
        for row in compact_sources
        if row.get("source_id") is not None
    }
    output: List[Dict[str, Any]] = []
    for source_id, row in expanded.items():
        if source_id in compact_ids:
            reasons[source_id].add("already_compact")
        item = _source_output_row(
            row,
            sorted(reasons[source_id]),
            needs_assistant=include_assistant,
        )
        # Tight source limits control prompt size.
        item["user_content"] = _truncate(
            item.get("user_content"),
            int(os.environ.get("QA_P8_SOURCE_CHARS", "1100")),
        )
        if item.get("assistant_reply"):
            item["assistant_reply"] = _truncate(
                item["assistant_reply"],
                int(os.environ.get("QA_P8_ASSISTANT_CHARS", "1400")),
            )
        item["rrf_score"] = fused.get(source_id, 0.0)
        output.append(item)

    output.sort(
        key=lambda row: (
            -float(row.get("rrf_score") or 0.0),
            int(row.get("source_id") or 0),
        )
    )
    return output[: int(os.environ.get("QA_P8_MAX_TARGET_ROWS", "14"))]


def _memory_time(memory: Dict[str, Any]) -> str:
    dimension = memory.get("dimension") or {}
    time_data = dimension.get("time") or {}
    return str(
        time_data.get("valid_from")
        or time_data.get("event_start")
        or time_data.get("raw")
        or ""
    )


def v2_timeline(
    question: str,
    parsed: Dict[str, Any],
    memories: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    contract = infer_contract_v2(question, parsed)
    if contract not in {"state_resolution", "preference_criteria", "comparison"}:
        return []

    keys = {_norm(value) for value in _state_keys(parsed)}
    query_tokens = {
        token
        for token in re.findall(r"[a-z0-9]+", _norm(question))
        if len(token) > 2
    }
    rows: List[Tuple[float, Dict[str, Any]]] = []
    for memory in memories:
        dimension = memory.get("dimension") or {}
        state_key = _norm(dimension.get("state_key"))
        content = str(memory.get("content") or "")
        content_tokens = set(re.findall(r"[a-z0-9]+", _norm(content)))
        overlap = len(query_tokens.intersection(content_tokens))
        score = float(overlap)
        if keys and state_key in keys:
            score += 10.0
        if contract == "preference_criteria" and dimension.get("preference"):
            score += 5.0
        if score <= 0:
            continue
        rows.append(
            (
                score,
                {
                    "memory_id": memory.get("memory_id"),
                    "content": content,
                    "state_key": dimension.get("state_key"),
                    "state_value": dimension.get("state_value"),
                    "state_status": dimension.get("state_status"),
                    "modality": dimension.get("modality"),
                    "time": dimension.get("time"),
                    "preference": dimension.get("preference"),
                    "provenance": memory.get("provenance"),
                },
            )
        )

    rows.sort(
        key=lambda item: (
            -item[0],
            _memory_time(item[1]),
            str(item[1].get("memory_id") or ""),
        )
    )
    return [
        row
        for _, row in rows[: int(os.environ.get("QA_P8_V2_TIMELINE_K", "8"))]
    ]


def _completed_months(first: datetime, second: datetime) -> int:
    if second < first:
        first, second = second, first
    months = (second.year - first.year) * 12 + second.month - first.month
    if second.day < first.day:
        months -= 1
    return max(0, months)


def safe_calculation(
    calculation: Dict[str, Any],
    contract: str,
) -> Optional[str]:
    operator = str(calculation.get("operator") or "none")
    operands = calculation.get("operands") or []
    unit = str(calculation.get("unit") or "").strip()

    expected = {
        "enumerate_count": {"count_distinct"},
        "sum": {"sum"},
        "duration": {
            "date_difference_days",
            "date_difference_weeks",
            "date_difference_months",
            "time_add_minutes",
        },
        "direct_numeric": {"stated_count", "none"},
    }
    if contract in expected and operator not in expected[contract]:
        return None

    if operator == "stated_count" and operands:
        number = _parse_number(operands[0])
        if number is not None:
            return str(int(number) if float(number).is_integer() else number)

    if operator == "count_distinct" and operands:
        # Do not turn ["7 short stories"] into 1. Atomic operands must not be
        # aggregate phrases containing a leading count.
        if len(operands) == 1:
            text = str(operands[0])
            number = _parse_number(text)
            if number is not None and len(re.findall(r"[a-zA-Z]+", text)) >= 1:
                return None
        distinct = {
            _norm(value)
            for value in operands
            if str(value or "").strip()
        }
        return str(len(distinct)) if distinct else None

    if operator == "sum" and operands:
        numbers = [_parse_number(value) for value in operands]
        if all(value is not None for value in numbers):
            total = sum(float(value) for value in numbers if value is not None)
            rendered = str(int(total)) if total.is_integer() else f"{total:g}"
            return f"{rendered} {unit}".strip()

    if operator.startswith("date_difference_") and len(operands) >= 2:
        first = _parse_date(operands[0])
        second = _parse_date(operands[1])
        if not first or not second:
            return None
        days = abs((second.date() - first.date()).days)
        if operator == "date_difference_days":
            return f"{days} days"
        if operator == "date_difference_weeks":
            weeks = days / 7.0
            rendered = str(int(weeks)) if weeks.is_integer() else f"{weeks:g}"
            return f"{rendered} weeks"
        if operator == "date_difference_months":
            return f"{_completed_months(first, second)} months"

    if operator == "time_add_minutes" and len(operands) >= 2:
        start = _parse_clock(operands[0])
        minutes = _parse_number(operands[1])
        if start is not None and minutes is not None:
            return (start + timedelta(minutes=float(minutes))).strftime("%-I:%M %p")

    return None


def safe_postprocess(
    candidate: Dict[str, Any],
    *,
    contract: str,
    parsed: Dict[str, Any],
    question: str,
    memories: Sequence[Dict[str, Any]],
    sources: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    result = dict(candidate or {})
    calculation = dict(result.get("calculation") or {})
    # Reuse support-ID validation from V4, but disable its unsafe executor.
    result["calculation"] = {"operator": "none", "operands": []}
    parsed_for_grounding = parsed
    if contract != "assistant_recall":
        parsed_for_grounding = {
            **parsed,
            "hypotheses": [
                {
                    **row,
                    "need_assistant_context": False,
                }
                for row in _hypotheses(parsed)
            ],
        }

    result = postprocess_candidate(
        result,
        parsed=parsed_for_grounding,
        question=question,
        memories=memories,
        sources=sources,
    )
    result["calculation"] = calculation

    calculated = safe_calculation(calculation, contract)
    if calculated is not None:
        result["answer"] = calculated
        result["answerable"] = True
        result["calculation"]["result"] = calculated
        result["safe_executor_applied"] = True

    # If an aggregate operator failed validation, preserve a supported natural
    # answer rather than overwriting it with a wrong number.
    return result


def _planner(
    client: Any,
    *,
    question: str,
    question_date: str,
    parsed: Dict[str, Any],
    candidates: Dict[str, Any],
    meta: CallMeta,
) -> Dict[str, Any]:
    hint = infer_contract_v2(question, parsed)
    payload, result = client.json(
        _render(
            PLAN_TEMPLATE_P8,
            question=question,
            question_date=question_date,
            query_json=parsed,
            contract_hint=hint,
            candidate_json=candidates,
        ),
        system=PLAN_SYSTEM_P8,
        max_tokens=int(os.environ.get("QA_P8_PLAN_TOKENS", "900")),
    )
    meta.add_result(result)
    return reconcile_plan(
        question,
        parsed,
        payload if isinstance(payload, dict) else {},
    )


def _specialist(
    client: Any,
    *,
    question: str,
    question_date: str,
    parsed: Dict[str, Any],
    plan: Dict[str, Any],
    memories: Sequence[Dict[str, Any]],
    compact: Sequence[Dict[str, Any]],
    targeted: Sequence[Dict[str, Any]],
    timeline: Sequence[Dict[str, Any]],
    meta: CallMeta,
) -> Dict[str, Any]:
    payload, result = client.json(
        _render(
            SPECIALIST_TEMPLATE_P8,
            question=question,
            question_date=question_date,
            plan_json=plan,
            memory_json=list(memories),
            compact_source_json=list(compact),
            targeted_source_json=list(targeted),
            v2_timeline_json=list(timeline),
        ),
        system=SPECIALIST_SYSTEM_P8,
        max_tokens=int(os.environ.get("QA_P8_SPECIALIST_TOKENS", "2200")),
    )
    meta.add_result(result)
    if not isinstance(payload, dict):
        payload = {"answer": ABSTENTION, "answerable": False}
    return safe_postprocess(
        payload,
        contract=str(plan.get("answer_contract") or "direct_value"),
        parsed=parsed,
        question=question,
        memories=memories,
        sources=_merge_sources(compact, targeted),
    )


def _same_answer(candidates: Dict[str, Dict[str, Any]]) -> bool:
    values = {
        _norm(candidate.get("answer") or candidate.get("prediction"))
        for candidate in candidates.values()
        if candidate
    }
    return len(values) <= 1


def _verification_evidence(
    compact: Sequence[Dict[str, Any]],
    targeted: Sequence[Dict[str, Any]],
    timeline: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    # Evidence passed to verifier is intentionally compact.
    merged = _merge_sources(compact[:8], targeted[:8])
    return {
        "sources": merged[:12],
        "v2_timeline": list(timeline)[:6],
    }


def _verify(
    client: Any,
    *,
    question: str,
    plan: Dict[str, Any],
    candidates: Dict[str, Any],
    evidence: Dict[str, Any],
    meta: CallMeta,
) -> Tuple[str, Dict[str, Any]]:
    payload, result = client.json(
        _render(
            VERIFY_TEMPLATE_P8,
            question=question,
            plan_json=plan,
            candidate_json=candidates,
            evidence_json=evidence,
        ),
        system=VERIFY_SYSTEM_P8,
        max_tokens=int(os.environ.get("QA_P8_VERIFY_TOKENS", "900")),
    )
    meta.add_result(result)
    if not isinstance(payload, dict):
        return "p7", {}
    choice = str(payload.get("choice") or "p7").casefold()
    if choice not in {"p5", "p7", "specialist", "abstain"}:
        choice = "p7"
    return choice, payload


def discover_samples(run_root: Path) -> List[Path]:
    """Use the holdout ID file, not a potentially stale run_manifest."""
    qid_file = run_root / "holdout100_question_ids.txt"
    allowed: Optional[Set[str]] = None
    if qid_file.exists():
        allowed = {
            line.strip()
            for line in qid_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }

    found: List[Tuple[str, Path]] = []
    for input_path in run_root.glob("*/*/input_item.json"):
        item = read_json(input_path)
        question_id = str(item.get("question_id") or "")
        if allowed is not None and question_id not in allowed:
            continue
        found.append((question_id, input_path.parent))

    found.sort(key=lambda item: item[0])
    if allowed is not None and len(found) != len(allowed):
        missing = sorted(allowed.difference(question_id for question_id, _ in found))
        raise RuntimeError(
            f"Expected {len(allowed)} holdout samples, found {len(found)}; "
            f"missing={missing[:10]}"
        )
    return [path for _, path in found]


def run_sample(
    sample_dir: Path,
    *,
    retrieval_name: str,
    p5_name: str,
    p7_name: str,
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
    compact_path = sample_dir / "qa_v2" / p5_name / "source_evidence.json"
    for path in (p5_path, p7_path, compact_path):
        if not path.exists():
            raise FileNotFoundError(path)

    copy_retrieval_variant(sample_dir, retrieval_name, output_name)
    item = read_json(sample_dir / "input_item.json")
    question, question_date, gold = _question_and_gold(item)
    parsed = read_json(sample_dir / "query_v2" / "parsed_query.json")
    p5_raw = read_json(p5_path)
    p7_raw = read_json(p7_path)
    p5 = _base_candidate_from_prediction(p5_raw)
    p7 = _base_candidate_from_prediction(p7_raw)
    compact = read_json(compact_path)
    source_turns = read_json(sample_dir / "source_turns.json")
    top_records = read_json(
        sample_dir / "retrieval_v2" / retrieval_name / "top_records.json"
    )
    memories = memory_evidence(
        top_records,
        limit=int(os.environ.get("QA_P8_MEMORY_K", "12")),
    )
    all_v2_path = sample_dir / "memory_v2" / "all_memories.json"
    all_v2 = read_json(all_v2_path) if all_v2_path.exists() else []

    escalate, risk_reasons = needs_escalation(
        p5_raw,
        p7_raw,
        question,
        parsed,
    )
    meta = CallMeta()
    plan = {
        "answer_contract": infer_contract_v2(question, parsed),
        "operator": operator_for_contract(
            infer_contract_v2(question, parsed),
            question,
        ),
        "slots": [],
        "retrieval_queries": [],
    }
    specialist: Dict[str, Any] = {}
    targeted: List[Dict[str, Any]] = []
    timeline = v2_timeline(question, parsed, all_v2)
    choice = "p7"
    verification: Dict[str, Any] = {}
    final = p7

    if escalate:
        client = make_client()
        plan = _planner(
            client,
            question=question,
            question_date=question_date,
            parsed=parsed,
            candidates={"p5": p5, "p7": p7},
            meta=meta,
        )
        targeted = slot_balanced_rrf(
            question,
            parsed,
            plan,
            source_turns,
            compact,
        )
        specialist = _specialist(
            client,
            question=question,
            question_date=question_date,
            parsed=parsed,
            plan=plan,
            memories=memories,
            compact=compact[: int(os.environ.get("QA_P8_COMPACT_K", "10"))],
            targeted=targeted,
            timeline=timeline,
            meta=meta,
        )
        candidates = {
            "p5": p5,
            "p7": p7,
            "specialist": specialist,
        }

        if _same_answer(candidates):
            # Avoid verifier tokens when all views agree.
            if specialist.get("support_source_ids"):
                choice = "specialist"
                final = specialist
            else:
                choice = "p7"
                final = p7
        else:
            choice, verification = _verify(
                client,
                question=question,
                plan=plan,
                candidates=candidates,
                evidence=_verification_evidence(
                    compact,
                    targeted,
                    timeline,
                ),
                meta=meta,
            )
            if choice == "p5":
                final = p5
            elif choice == "specialist":
                final = specialist
            elif choice == "abstain":
                final = {
                    "answer": ABSTENTION,
                    "answerable": False,
                    "support_memory_ids": [],
                    "support_source_ids": [],
                    "required_slots": [],
                    "unfilled_slots": ["verified evidence insufficient"],
                    "operation": "none",
                    "calculation": {"operator": "none", "operands": []},
                    "confidence": 0.1,
                    "reasoning_summary": "Evidence verifier found no supported candidate.",
                }
            else:
                choice = "p7"
                final = p7

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
        "reasoning_summary": final.get("reasoning_summary") or "",
        "required_slots": final.get("required_slots") or [],
        "unfilled_slots": final.get("unfilled_slots") or [],
        "operation": final.get("operation") or "none",
        "calculation": final.get("calculation") or {},
        "answerable": bool(final.get("answerable")),
        "retrieval_name": output_name,
        "source_retrieval_name": retrieval_name,
        "p5_name": p5_name,
        "p7_name": p7_name,
        "p8_choice": choice,
        "risk_reasons": risk_reasons,
        "plan": plan,
        "verification": verification,
        "meta": {
            "mode": "p8_lite_generic",
            **meta.to_dict(),
        },
    }
    qa_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(qa_path, qa_output)
    write_json(
        qa_path.parent / "p8_trace.json",
        {
            "escalated": escalate,
            "risk_reasons": risk_reasons,
            "plan": plan,
            "p5_candidate": p5,
            "p7_candidate": p7,
            "specialist_candidate": specialist,
            "choice": choice,
            "verification": verification,
            "targeted_source_ids": [
                row.get("source_id") for row in targeted
            ],
            "v2_timeline_memory_ids": [
                row.get("memory_id") for row in timeline
            ],
        },
    )
    write_json(
        qa_path.parent / "targeted_source_evidence.json",
        targeted,
    )
    write_json(
        qa_path.parent / "v2_timeline.json",
        timeline,
    )

    client = make_client()
    judge_meta = CallMeta()
    correct, reason, mode = judge_prediction(
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
        "judge_meta": {"mode": mode, **judge_meta.to_dict()},
    }
    judge_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(judge_path, judge_output)

    return {
        "status": "ok",
        "question_id": item.get("question_id"),
        "sample_dir": str(sample_dir),
        "correct": bool(correct),
        "escalated": escalate,
        "choice": choice,
        "prompt_tokens": meta.prompt_tokens,
        "completion_tokens": meta.completion_tokens,
        "targeted_source_count": len(targeted),
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
                f"[p8-lite] {index}/{len(samples)} | {sample.name} | "
                f"{row.get('status')} | choice={row.get('choice', '-')}",
                flush=True,
            )

    correct = sum(bool(row.get("correct")) for row in rows)
    summary = {
        "retrieval_name": args.retrieval_name,
        "p5_name": args.p5_name,
        "p7_name": args.p7_name,
        "output_name": args.output_name,
        "n": len(rows),
        "correct": correct,
        "accuracy": correct / len(rows) if rows else 0.0,
        "workers": workers,
        "rows": rows,
    }
    write_json(
        run_root / f"p8_lite_manifest_{args.output_name}.json",
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
        "--output-name",
        default="graph_active_p8_lite_generic",
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
