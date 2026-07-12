from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import statistics
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .io_utils import normalize_question_type, normalize_text, read_json, write_json
from .llm_client import OpenAICompatibleClient
from .prompts_source_v4 import (
    JUDGE_SYSTEM_V4,
    JUDGE_TEMPLATE_V4,
    QA_SYSTEM_V4,
    QA_TEMPLATE_V4,
    VERIFY_SYSTEM_V4,
    VERIFY_TEMPLATE_V4,
)


ABSTENTION = "Cannot be determined from the conversation."

_ABSTENTION_PATTERNS = [
    r"\bcannot be determined\b",
    r"\bcan(?:not|'t) determine\b",
    r"\binsufficient information\b",
    r"\bnot enough information\b",
    r"\binformation (?:provided )?is not enough\b",
    r"\bnot enough (?:information|evidence|details)\b",
    r"\bno information (?:is |was )?provided\b",
    r"\bno evidence\b",
    r"\bwas not mentioned\b",
    r"\bwere not mentioned\b",
    r"\bdid not mention\b",
    r"\bnever mentioned\b",
]

_ASSISTANT_QUERY_RE = re.compile(
    r"\b(?:you|assistant)\s+(?:said|suggested|recommended|explained|"
    r"advised|told|listed|called|referred)\b|"
    r"\bprevious (?:answer|response|reply|chat)\b",
    flags=re.IGNORECASE,
)


def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)


def _render(template: str, **values: Any) -> str:
    result = template
    for key, value in values.items():
        if isinstance(value, (dict, list)):
            replacement = _json_text(value)
        else:
            replacement = str(value if value is not None else "")
        result = result.replace("{" + key + "}", replacement)
    return result


def _tokens(text: Any) -> List[str]:
    return [
        token
        for token in re.findall(r"[a-z0-9]+", str(text or "").casefold())
        if len(token) > 1
    ]


def _truncate(text: Any, limit: int) -> str:
    value = str(text or "")
    if len(value) <= limit:
        return value
    return value[: max(0, limit - 20)] + " …[truncated]"


def is_abstention(text: Any) -> bool:
    normalized = normalize_text(str(text or ""))
    if not normalized:
        return True
    return any(re.search(pattern, normalized) for pattern in _ABSTENTION_PATTERNS)


def question_needs_assistant(parsed: Dict[str, Any], question: str) -> bool:
    hypotheses = parsed.get("hypotheses") or []
    if any(
        bool(hypothesis.get("need_assistant_context"))
        for hypothesis in hypotheses
        if isinstance(hypothesis, dict)
    ):
        return True
    return bool(_ASSISTANT_QUERY_RE.search(question))


def _question_and_gold(item: Dict[str, Any]) -> Tuple[str, str, str]:
    question = str(item.get("question") or item.get("query") or "").strip()
    question_date = str(
        item.get("question_date")
        or item.get("query_date")
        or item.get("question_timestamp")
        or ""
    ).strip()
    answer = item.get("answer")
    if answer is None:
        answer = item.get("gold_answer")
    if isinstance(answer, list):
        answer = answer[0] if answer else ""
    return question, question_date, str(answer or "").strip()


def _source_index(
    source_turns: Sequence[Dict[str, Any]],
) -> Tuple[Dict[int, Dict[str, Any]], Dict[str, List[Dict[str, Any]]]]:
    by_id: Dict[int, Dict[str, Any]] = {}
    by_session: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in source_turns:
        try:
            source_id = int(row.get("source_id"))
        except (TypeError, ValueError):
            continue
        by_id[source_id] = row
        by_session[str(row.get("session_id") or "")].append(row)
    for rows in by_session.values():
        rows.sort(key=lambda row: int(row.get("source_id") or 0))
    return by_id, by_session


def _idf_scores(
    source_turns: Sequence[Dict[str, Any]],
) -> Dict[str, float]:
    documents = [
        set(_tokens(
            str(row.get("content") or "")
            + " "
            + str(row.get("assistant_reply") or "")
        ))
        for row in source_turns
    ]
    document_frequency: Counter[str] = Counter()
    for document in documents:
        document_frequency.update(document)
    n = max(1, len(documents))
    return {
        token: math.log((n + 1) / (frequency + 1)) + 1.0
        for token, frequency in document_frequency.items()
    }


def _quoted_phrases(question: str) -> List[str]:
    return [
        phrase.casefold().strip()
        for phrase in re.findall(r"['\"]([^'\"]{3,120})['\"]", question)
        if phrase.strip()
    ]


def _source_lexical_score(
    question: str,
    row: Dict[str, Any],
    idf: Dict[str, float],
) -> float:
    query_tokens = _tokens(question)
    if not query_tokens:
        return 0.0

    text = (
        str(row.get("content") or "")
        + " "
        + str(row.get("assistant_reply") or "")
    ).casefold()
    doc_tokens = Counter(_tokens(text))
    score = sum(
        idf.get(token, 1.0) * min(3, doc_tokens.get(token, 0))
        for token in set(query_tokens)
    )

    for phrase in _quoted_phrases(question):
        if phrase in text:
            score += 12.0 + 0.5 * len(_tokens(phrase))

    query_bigrams = {
        " ".join(query_tokens[index:index + 2])
        for index in range(max(0, len(query_tokens) - 1))
    }
    score += 2.5 * sum(1 for bigram in query_bigrams if bigram in text)
    return score


def _adjacent_rows(
    target: Dict[str, Any],
    by_session: Dict[str, List[Dict[str, Any]]],
    radius: int,
) -> List[Dict[str, Any]]:
    session_id = str(target.get("session_id") or "")
    rows = by_session.get(session_id) or []
    source_id = int(target.get("source_id") or 0)
    position = next(
        (
            index
            for index, row in enumerate(rows)
            if int(row.get("source_id") or 0) == source_id
        ),
        None,
    )
    if position is None:
        return []
    start = max(0, position - radius)
    end = min(len(rows), position + radius + 1)
    return rows[start:end]


def reconstruct_source_evidence(
    question: str,
    parsed: Dict[str, Any],
    memory_rows: Sequence[Dict[str, Any]],
    source_turns: Sequence[Dict[str, Any]],
    *,
    direct_memory_k: int = 12,
    adjacent_memory_k: int = 5,
    adjacent_radius: int = 1,
    lexical_k: int = 8,
    max_source_rows: int = 24,
    user_char_limit: int = 1600,
    assistant_char_limit: int = 2200,
) -> List[Dict[str, Any]]:
    by_id, by_session = _source_index(source_turns)
    selected: Dict[int, Dict[str, Any]] = {}
    reasons: Dict[int, Set[str]] = defaultdict(set)

    for rank, memory in enumerate(memory_rows[:direct_memory_k], start=1):
        provenance = memory.get("provenance") or {}
        for raw_id in provenance.get("source_ids") or []:
            try:
                source_id = int(raw_id)
            except (TypeError, ValueError):
                continue
            if source_id not in by_id:
                continue
            selected[source_id] = by_id[source_id]
            reasons[source_id].add(f"memory_rank_{rank}")
            if rank <= adjacent_memory_k:
                for adjacent in _adjacent_rows(
                    by_id[source_id],
                    by_session,
                    adjacent_radius,
                ):
                    adjacent_id = int(adjacent.get("source_id") or 0)
                    selected[adjacent_id] = adjacent
                    reasons[adjacent_id].add(
                        f"adjacent_to_memory_rank_{rank}"
                    )

    idf = _idf_scores(source_turns)
    lexical_rows = sorted(
        source_turns,
        key=lambda row: (
            -_source_lexical_score(question, row, idf),
            int(row.get("source_id") or 0),
        ),
    )[:lexical_k]
    for row in lexical_rows:
        source_id = int(row.get("source_id") or 0)
        score = _source_lexical_score(question, row, idf)
        if score <= 0:
            continue
        selected[source_id] = row
        reasons[source_id].add(f"source_lexical_{score:.3f}")

    needs_assistant = question_needs_assistant(parsed, question)
    ranked = sorted(
        selected.values(),
        key=lambda row: (
            0 if any(
                reason.startswith("memory_rank_")
                for reason in reasons[int(row.get("source_id") or 0)]
            ) else 1,
            -_source_lexical_score(question, row, idf),
            int(row.get("source_id") or 0),
        ),
    )[:max_source_rows]

    output: List[Dict[str, Any]] = []
    for row in ranked:
        source_id = int(row.get("source_id") or 0)
        item = {
            "source_id": source_id,
            "source_uid": row.get("source_uid"),
            "session_id": row.get("session_id"),
            "timestamp": row.get("timestamp"),
            "user_content": _truncate(
                row.get("content"),
                user_char_limit,
            ),
            "selection_reasons": sorted(reasons[source_id]),
        }
        assistant_reply = str(row.get("assistant_reply") or "")
        if assistant_reply and (
            needs_assistant
            or _source_lexical_score(question, row, idf) >= 4.0
        ):
            item["assistant_reply"] = _truncate(
                assistant_reply,
                assistant_char_limit,
            )
        output.append(item)
    return output


def memory_evidence(
    rows: Sequence[Dict[str, Any]],
    *,
    limit: int = 18,
) -> List[Dict[str, Any]]:
    return [
        {
            "memory_id": row.get("memory_id"),
            "content": row.get("content"),
            "dimension": row.get("dimension"),
            "provenance": row.get("provenance"),
            "assistant_replies": row.get("assistant_replies") or [],
            "score": row.get("score"),
            "rank": row.get("rank"),
        }
        for row in rows[:limit]
    ]


def _parse_number(value: Any) -> Optional[float]:
    match = re.search(
        r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?",
        str(value or ""),
    )
    if not match:
        return None
    try:
        return float(match.group(0).replace(",", ""))
    except ValueError:
        return None


def _parse_date(value: Any) -> Optional[datetime]:
    text = str(value or "").strip()
    patterns = [
        r"\d{4}-\d{2}-\d{2}",
        r"\d{4}/\d{2}/\d{2}",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if not match:
            continue
        candidate = match.group(0).replace("/", "-")
        try:
            return datetime.strptime(candidate, "%Y-%m-%d")
        except ValueError:
            continue
    return None


def _parse_clock(value: Any) -> Optional[datetime]:
    text = str(value or "").strip()
    for fmt in ("%I:%M %p", "%I %p", "%H:%M"):
        try:
            return datetime.strptime(text.upper(), fmt)
        except ValueError:
            continue
    match = re.search(r"\b(\d{1,2})(?::(\d{2}))?\s*([AP]M)\b", text, re.I)
    if match:
        hour = int(match.group(1))
        minute = int(match.group(2) or 0)
        meridiem = match.group(3).upper()
        if meridiem == "PM" and hour != 12:
            hour += 12
        if meridiem == "AM" and hour == 12:
            hour = 0
        return datetime(2000, 1, 1, hour, minute)
    return None


def _format_number(number: float) -> str:
    if abs(number - round(number)) < 1e-9:
        return str(int(round(number)))
    return f"{number:.6f}".rstrip("0").rstrip(".")


def deterministic_calculation(
    calculation: Dict[str, Any],
) -> Optional[str]:
    operator = str(calculation.get("operator") or "none")
    operands = calculation.get("operands") or []
    unit = str(calculation.get("unit") or "").strip()

    if operator == "date_difference_days" and len(operands) >= 2:
        first = _parse_date(operands[0])
        second = _parse_date(operands[1])
        if first and second:
            days = abs((second.date() - first.date()).days)
            return f"{days} days"

    if operator == "time_add_minutes" and len(operands) >= 2:
        start = _parse_clock(operands[0])
        minutes = _parse_number(operands[1])
        if start is not None and minutes is not None:
            result = start + timedelta(minutes=minutes)
            return result.strftime("%-I:%M %p")

    if operator == "sum" and operands:
        numbers = [_parse_number(value) for value in operands]
        if all(number is not None for number in numbers):
            total = sum(number for number in numbers if number is not None)
            rendered = _format_number(total)
            return f"{rendered} {unit}".strip()

    if operator == "count_distinct" and operands:
        distinct = {
            normalize_text(str(value))
            for value in operands
            if str(value or "").strip()
        }
        if distinct:
            return str(len(distinct))

    return None


def _valid_support_ids(
    candidate: Dict[str, Any],
    memories: Sequence[Dict[str, Any]],
    sources: Sequence[Dict[str, Any]],
) -> Tuple[List[str], List[int]]:
    valid_memory_ids = {
        str(row.get("memory_id"))
        for row in memories
        if row.get("memory_id")
    }
    valid_source_ids = {
        int(row.get("source_id"))
        for row in sources
        if row.get("source_id") is not None
    }
    memory_ids = [
        str(value)
        for value in candidate.get("support_memory_ids") or []
        if str(value) in valid_memory_ids
    ]
    source_ids: List[int] = []
    for value in candidate.get("support_source_ids") or []:
        try:
            source_id = int(value)
        except (TypeError, ValueError):
            continue
        if source_id in valid_source_ids:
            source_ids.append(source_id)
    return list(dict.fromkeys(memory_ids)), list(dict.fromkeys(source_ids))


def postprocess_candidate(
    candidate: Dict[str, Any],
    *,
    parsed: Dict[str, Any],
    question: str,
    memories: Sequence[Dict[str, Any]],
    sources: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    result = dict(candidate)
    memory_ids, source_ids = _valid_support_ids(
        result,
        memories,
        sources,
    )
    result["support_memory_ids"] = memory_ids
    result["support_source_ids"] = source_ids
    result.setdefault("required_slots", [])
    result.setdefault("unfilled_slots", [])
    result.setdefault("operation", "none")
    result.setdefault("calculation", {"operator": "none", "operands": []})
    result.setdefault("reasoning_summary", "")
    result.setdefault("confidence", 0.0)

    calculated = deterministic_calculation(
        result.get("calculation") or {}
    )
    if calculated is not None:
        result["answer"] = calculated
        result["answerable"] = True
        result["calculation"]["result"] = calculated
        result["postprocess"] = {
            "deterministic_calculation_applied": True
        }

    answer = str(result.get("answer") or "").strip()
    answerable = bool(result.get("answerable", bool(answer)))

    if (
        str((result.get("calculation") or {}).get("operator"))
        == "count_distinct"
        and not (result.get("calculation") or {}).get("operands")
    ):
        answerable = False

    if question_needs_assistant(parsed, question):
        supported_assistant_source = any(
            int(row.get("source_id") or -1) in set(source_ids)
            and bool(row.get("assistant_reply"))
            for row in sources
        )
        if not supported_assistant_source:
            answerable = False
            result["grounding_failure"] = (
                "assistant answer lacks a supported assistant reply"
            )

    if answerable and not memory_ids and not source_ids:
        answerable = False
        result["grounding_failure"] = (
            "concrete answer has no valid support IDs"
        )

    if not answerable:
        result["answer"] = ABSTENTION
        result["answerable"] = False
        result["confidence"] = min(
            float(result.get("confidence") or 0.0),
            0.2,
        )
    else:
        result["answerable"] = True
        result["answer"] = answer or ABSTENTION

    return result


def candidate_score(candidate: Dict[str, Any]) -> float:
    score = 0.0
    if candidate.get("answerable"):
        score += 2.0
    score += 0.35 * len(candidate.get("support_memory_ids") or [])
    score += 0.45 * len(candidate.get("support_source_ids") or [])
    score -= 0.7 * len(candidate.get("unfilled_slots") or [])
    if (
        candidate.get("postprocess", {})
        .get("deterministic_calculation_applied")
    ):
        score += 1.5
    if candidate.get("grounding_failure"):
        score -= 3.0
    score += 0.2 * float(candidate.get("confidence") or 0.0)
    return score


@dataclass
class CallMeta:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    elapsed_seconds: float = 0.0

    def add_result(self, result: Any) -> None:
        self.prompt_tokens += int(result.prompt_tokens or 0)
        self.completion_tokens += int(result.completion_tokens or 0)
        self.elapsed_seconds += float(result.elapsed_seconds or 0.0)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "elapsed_seconds": self.elapsed_seconds,
        }


def make_client() -> OpenAICompatibleClient:
    missing = [
        name
        for name in ("OPENAI_BASE_URL", "OPENAI_API_KEY", "MODEL_NAME")
        if not os.environ.get(name)
    ]
    if missing:
        raise RuntimeError(
            "Missing LLM environment variables: " + ", ".join(missing)
        )
    return OpenAICompatibleClient(
        base_url=os.environ["OPENAI_BASE_URL"],
        api_key=os.environ["OPENAI_API_KEY"],
        model_name=os.environ["MODEL_NAME"],
        timeout=int(os.environ.get("QA_V4_TIMEOUT", "600")),
        retries=int(os.environ.get("QA_V4_RETRIES", "5")),
    )


def generate_candidate(
    client: OpenAICompatibleClient,
    *,
    question: str,
    question_date: str,
    parsed: Dict[str, Any],
    memories: Sequence[Dict[str, Any]],
    sources: Sequence[Dict[str, Any]],
    meta: CallMeta,
) -> Dict[str, Any]:
    payload, result = client.json(
        _render(
            QA_TEMPLATE_V4,
            question=question,
            question_date=question_date,
            query_json=parsed,
            memory_evidence_json=list(memories),
            source_evidence_json=list(sources),
        ),
        system=QA_SYSTEM_V4,
        max_tokens=int(os.environ.get("QA_V4_MAX_TOKENS", "3200")),
    )
    meta.add_result(result)
    if not isinstance(payload, dict):
        payload = {"answer": str(payload)}
    return postprocess_candidate(
        payload,
        parsed=parsed,
        question=question,
        memories=memories,
        sources=sources,
    )


def verify_candidate(
    client: OpenAICompatibleClient,
    *,
    question: str,
    candidate: Dict[str, Any],
    parsed: Dict[str, Any],
    memories: Sequence[Dict[str, Any]],
    sources: Sequence[Dict[str, Any]],
    meta: CallMeta,
) -> Dict[str, Any]:
    payload, result = client.json(
        _render(
            VERIFY_TEMPLATE_V4,
            question=question,
            candidate_json=candidate,
            memory_evidence_json=list(memories),
            source_evidence_json=list(sources),
        ),
        system=VERIFY_SYSTEM_V4,
        max_tokens=int(os.environ.get("QA_V4_VERIFY_TOKENS", "2200")),
    )
    meta.add_result(result)
    if not isinstance(payload, dict):
        return candidate
    if bool(payload.get("accept", False)):
        return candidate
    corrected = payload.get("corrected")
    if not isinstance(corrected, dict):
        return candidate
    corrected = postprocess_candidate(
        corrected,
        parsed=parsed,
        question=question,
        memories=memories,
        sources=sources,
    )
    if candidate_score(corrected) >= candidate_score(candidate) - 0.25:
        corrected["verifier_reason"] = payload.get("reason") or ""
        return corrected
    return candidate


def judge_prediction(
    client: OpenAICompatibleClient,
    *,
    question_type: str,
    question: str,
    gold_answer: str,
    prediction: str,
    meta: CallMeta,
) -> Tuple[bool, str, str]:
    gold_abstains = is_abstention(gold_answer)
    prediction_abstains = is_abstention(prediction)

    if gold_abstains and prediction_abstains:
        return (
            True,
            "Both reference and prediction indicate insufficient information.",
            "deterministic_abstention_equivalence",
        )
    if gold_abstains and not prediction_abstains:
        return (
            False,
            "Reference indicates insufficient information but prediction is concrete.",
            "deterministic_abstention_guard",
        )
    if prediction_abstains and not gold_abstains:
        return (
            False,
            "Prediction abstained while reference contains a concrete answer.",
            "deterministic_abstention_guard",
        )

    payload, result = client.json(
        _render(
            JUDGE_TEMPLATE_V4,
            question_type=question_type,
            question=question,
            gold_answer=gold_answer,
            prediction=prediction,
        ),
        system=JUDGE_SYSTEM_V4,
        max_tokens=800,
    )
    meta.add_result(result)
    label = (
        str(payload.get("label") or "").upper()
        if isinstance(payload, dict)
        else ""
    )
    reason = (
        str(payload.get("reason") or "")
        if isinstance(payload, dict)
        else ""
    )
    return label == "CORRECT", reason, "llm"


def copy_retrieval_variant(
    sample_dir: Path,
    source_name: str,
    output_name: str,
) -> None:
    source = sample_dir / "retrieval_v2" / source_name
    target = sample_dir / "retrieval_v2" / output_name
    target.mkdir(parents=True, exist_ok=True)
    for filename in (
        "top_records.json",
        "top_records_compat.json",
        "retrieval_trace.json",
        "retrieval_meta.json",
    ):
        source_path = source / filename
        target_path = target / filename
        if source_path.exists():
            shutil.copy2(source_path, target_path)
    write_json(
        target / "source_qa_v4_meta.json",
        {
            "source_retrieval_name": source_name,
            "output_name": output_name,
        },
    )


def run_sample(
    sample_dir: Path,
    *,
    source_retrieval_name: str,
    output_name: str,
    qa_votes: int,
    verify: bool,
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

    copy_retrieval_variant(
        sample_dir,
        source_retrieval_name,
        output_name,
    )

    item = read_json(sample_dir / "input_item.json")
    question, question_date, gold_answer = _question_and_gold(item)
    parsed = read_json(sample_dir / "query_v2" / "parsed_query.json")
    top_records = read_json(
        sample_dir
        / "retrieval_v2"
        / source_retrieval_name
        / "top_records.json"
    )
    source_turns = read_json(sample_dir / "source_turns.json")

    memories = memory_evidence(
        top_records,
        limit=int(os.environ.get("QA_V4_MEMORY_K", "18")),
    )
    sources = reconstruct_source_evidence(
        question,
        parsed,
        top_records,
        source_turns,
        direct_memory_k=int(os.environ.get("QA_V4_DIRECT_MEMORY_K", "12")),
        adjacent_memory_k=int(os.environ.get("QA_V4_ADJACENT_MEMORY_K", "5")),
        adjacent_radius=int(os.environ.get("QA_V4_ADJACENT_RADIUS", "1")),
        lexical_k=int(os.environ.get("QA_V4_SOURCE_LEXICAL_K", "8")),
        max_source_rows=int(os.environ.get("QA_V4_MAX_SOURCE_ROWS", "24")),
    )

    client = make_client()
    qa_meta = CallMeta()
    candidates = [
        generate_candidate(
            client,
            question=question,
            question_date=question_date,
            parsed=parsed,
            memories=memories,
            sources=sources,
            meta=qa_meta,
        )
        for _ in range(max(1, qa_votes))
    ]
    candidate = max(candidates, key=candidate_score)
    if verify:
        candidate = verify_candidate(
            client,
            question=question,
            candidate=candidate,
            parsed=parsed,
            memories=memories,
            sources=sources,
            meta=qa_meta,
        )

    qa_output = {
        "question_id": item.get("question_id"),
        "question_type": normalize_question_type(
            item.get("question_type") or sample_dir.parent.name
        ),
        "question": question,
        "question_date": question_date,
        "gold_answer": gold_answer,
        "prediction": str(candidate.get("answer") or ABSTENTION),
        "support_memory_ids": candidate.get("support_memory_ids") or [],
        "support_source_ids": candidate.get("support_source_ids") or [],
        "confidence": candidate.get("confidence", 0.0),
        "reasoning_summary": candidate.get("reasoning_summary") or "",
        "required_slots": candidate.get("required_slots") or [],
        "unfilled_slots": candidate.get("unfilled_slots") or [],
        "operation": candidate.get("operation") or "none",
        "calculation": candidate.get("calculation") or {},
        "answerable": bool(candidate.get("answerable")),
        "retrieval_name": output_name,
        "source_retrieval_name": source_retrieval_name,
        "source_evidence_ids": [
            row.get("source_id") for row in sources
        ],
        "candidate_count": len(candidates),
        "candidate_scores": [
            candidate_score(row) for row in candidates
        ],
        "meta": {
            "mode": "source_qa_v4",
            **qa_meta.to_dict(),
        },
    }
    qa_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(qa_path, qa_output)
    write_json(
        qa_path.parent / "source_evidence.json",
        sources,
    )
    write_json(
        qa_path.parent / "qa_candidates.json",
        candidates,
    )

    judge_meta = CallMeta()
    correct, reason, judge_mode = judge_prediction(
        client,
        question_type=qa_output["question_type"],
        question=question,
        gold_answer=gold_answer,
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
        "sample_dir": str(sample_dir),
        "question_id": item.get("question_id"),
        "correct": bool(correct),
        "qa_prompt_tokens": qa_meta.prompt_tokens,
        "qa_completion_tokens": qa_meta.completion_tokens,
        "judge_prompt_tokens": judge_meta.prompt_tokens,
        "judge_completion_tokens": judge_meta.completion_tokens,
        "source_evidence_count": len(sources),
    }


def selected_samples(
    run_root: Path,
    question_types: Sequence[str],
) -> List[Path]:
    manifest = read_json(run_root / "run_manifest.json")
    allowed = {
        normalize_question_type(value)
        for value in question_types
        if value.strip()
    }
    output: List[Path] = []
    for sample in manifest.get("samples") or []:
        if allowed and sample.get("question_type") not in allowed:
            continue
        output.append(Path(sample["sample_dir"]))
    return output


def run_all(args: argparse.Namespace) -> Dict[str, Any]:
    run_root = Path(args.run_root)
    samples = selected_samples(
        run_root,
        args.question_types.split(",") if args.question_types else [],
    )
    workers = len(samples) if args.workers == 0 else max(1, args.workers)
    rows: List[Dict[str, Any]] = []

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                run_sample,
                sample_dir,
                source_retrieval_name=args.source_retrieval_name,
                output_name=args.output_name,
                qa_votes=args.qa_votes,
                verify=not args.no_verify,
                force=args.force,
            ): sample_dir
            for sample_dir in samples
        }
        completed = 0
        for future in as_completed(futures):
            completed += 1
            sample_dir = futures[future]
            try:
                row = future.result()
            except Exception as exc:
                if args.fail_fast:
                    raise
                row = {
                    "status": "failed",
                    "sample_dir": str(sample_dir),
                    "error": repr(exc),
                    "correct": False,
                }
            rows.append(row)
            print(
                f"[source-qa-v4] {completed}/{len(samples)} | "
                f"{sample_dir.name} | {row.get('status')}",
                flush=True,
            )

    correct = sum(1 for row in rows if row.get("correct"))
    summary = {
        "source_retrieval_name": args.source_retrieval_name,
        "output_name": args.output_name,
        "n": len(rows),
        "correct": correct,
        "accuracy": correct / len(rows) if rows else 0.0,
        "qa_votes": args.qa_votes,
        "verify": not args.no_verify,
        "workers": workers,
        "rows": rows,
    }
    write_json(
        run_root / f"source_qa_v4_manifest_{args.output_name}.json",
        summary,
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run source-backed, self-consistent QA and corrected judging "
            "on an existing retrieval variant."
        )
    )
    parser.add_argument("--run-root", required=True)
    parser.add_argument(
        "--source-retrieval-name",
        default="graph_active_p2_gate",
    )
    parser.add_argument(
        "--output-name",
        default="graph_active_p5_source_qa",
    )
    parser.add_argument("--qa-votes", type=int, default=2)
    parser.add_argument("--no-verify", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--question-types", default="")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    summary = run_all(args)
    print(_json_text(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
