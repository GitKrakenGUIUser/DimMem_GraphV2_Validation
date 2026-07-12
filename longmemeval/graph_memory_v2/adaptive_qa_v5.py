from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

from .io_utils import normalize_question_type, normalize_text, read_json, write_json
from .prompts_adaptive_v5 import (
    ARBITER_SYSTEM_V5,
    ARBITER_TEMPLATE_V5,
    PLAN_SYSTEM_V5,
    PLAN_TEMPLATE_V5,
    SPECIALIST_SYSTEM_V5,
    SPECIALIST_TEMPLATE_V5,
)
from .source_qa_v4 import (
    ABSTENTION,
    CallMeta,
    _adjacent_rows,
    _idf_scores,
    _question_and_gold,
    _render,
    _source_index,
    _source_lexical_score,
    _truncate,
    candidate_score,
    copy_retrieval_variant,
    is_abstention,
    judge_prediction,
    make_client,
    memory_evidence,
    postprocess_candidate,
    question_needs_assistant,
)


_MODEL = None
_MODEL_PATH = None
_MODEL_LOAD_LOCK = threading.Lock()
_MODEL_ENCODE_LOCK = threading.Lock()


def _unique(values: Iterable[str]) -> List[str]:
    output: List[str] = []
    seen: Set[str] = set()
    for value in values:
        text = str(value or "").strip()
        marker = text.casefold()
        if not text or marker in seen:
            continue
        seen.add(marker)
        output.append(text)
    return output


def _hypotheses(parsed: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [
        row
        for row in parsed.get("hypotheses") or []
        if isinstance(row, dict)
    ]


def _answer_dims(parsed: Dict[str, Any]) -> Set[str]:
    return {
        str(row.get("answer_dim") or "").casefold()
        for row in _hypotheses(parsed)
        if row.get("answer_dim")
    }


def _target_memory_types(parsed: Dict[str, Any]) -> Set[str]:
    return {
        str(value).casefold()
        for row in _hypotheses(parsed)
        for value in row.get("target_memory_types") or []
    }


def _state_keys(parsed: Dict[str, Any]) -> List[str]:
    return _unique(
        value
        for row in _hypotheses(parsed)
        for value in row.get("state_keys") or []
    )


def infer_contract_hint(
    question: str,
    parsed: Dict[str, Any],
) -> str:
    q = normalize_text(question)
    dims = _answer_dims(parsed)
    memory_types = _target_memory_types(parsed)

    if question_needs_assistant(parsed, question):
        return "assistant_recall"
    if "preference" in dims or "profile" in memory_types:
        if re.search(
            r"\b(?:suggest|recommend|prefer|should i|what do you think|"
            r"decide whether)\b",
            q,
        ):
            return "preference_criteria"
    if _state_keys(parsed) and re.search(
        r"\b(?:current|currently|now|latest|still)\b",
        q,
    ):
        return "state_resolution"
    if re.search(r"\b(?:how many|number of)\b", q):
        return "count"
    if re.search(r"\b(?:total|combined|altogether|sum of)\b", q):
        return "sum"
    if re.search(
        r"\b(?:order|earliest to latest|latest to earliest|chronological)\b",
        q,
    ):
        return "timeline"
    if re.search(r"\b(?:what time|how long|how many days)\b", q):
        return "temporal"
    return "direct_value"


def needs_specialist(
    base: Dict[str, Any],
    question: str,
    parsed: Dict[str, Any],
) -> Tuple[bool, List[str]]:
    reasons: List[str] = []
    contract = infer_contract_hint(question, parsed)

    if is_abstention(base.get("prediction") or base.get("answer")):
        reasons.append("base_abstained")
    if base.get("unfilled_slots"):
        reasons.append("base_has_unfilled_slots")
    if contract in {
        "count",
        "sum",
        "timeline",
        "state_resolution",
        "preference_criteria",
        "temporal",
        "assistant_recall",
    }:
        reasons.append(f"specialist_contract:{contract}")

    operation = str(base.get("operation") or "none")
    if contract in {"count", "sum", "timeline", "temporal"} and operation == "none":
        reasons.append("expected_operation_missing")

    return bool(reasons), reasons


def _planner_payload(
    client: Any,
    *,
    question: str,
    question_date: str,
    parsed: Dict[str, Any],
    base: Dict[str, Any],
    meta: CallMeta,
) -> Dict[str, Any]:
    payload, result = client.json(
        _render(
            PLAN_TEMPLATE_V5,
            question=question,
            question_date=question_date,
            query_json=parsed,
            base_candidate_json=base,
        ),
        system=PLAN_SYSTEM_V5,
        max_tokens=int(os.environ.get("QA_V5_PLAN_TOKENS", "1400")),
    )
    meta.add_result(result)
    if not isinstance(payload, dict):
        payload = {}
    payload.setdefault(
        "answer_contract",
        infer_contract_hint(question, parsed),
    )
    payload.setdefault("operator", "none")
    payload.setdefault("target_count", 0)
    payload.setdefault("slots", [])
    payload.setdefault("retrieval_queries", [])
    payload.setdefault("output_constraints", [])
    return payload


def _load_dense_model() -> Any:
    global _MODEL, _MODEL_PATH
    model_path = os.environ.get("QA_V5_EMBEDDING_MODEL", "").strip()
    if not model_path:
        return None

    with _MODEL_LOAD_LOCK:
        if _MODEL is not None and _MODEL_PATH == model_path:
            return _MODEL
        from sentence_transformers import SentenceTransformer

        device = os.environ.get("QA_V5_EMBEDDING_DEVICE", "cpu")
        _MODEL = SentenceTransformer(model_path, device=device)
        _MODEL_PATH = model_path
        return _MODEL


def _dense_scores(
    queries: Sequence[str],
    source_turns: Sequence[Dict[str, Any]],
    *,
    include_assistant: bool,
) -> List[float]:
    model = _load_dense_model()
    if model is None or not queries or not source_turns:
        return [0.0] * len(source_turns)

    documents = [
        (
            str(row.get("content") or "")
            + (
                "\n" + str(row.get("assistant_reply") or "")
                if include_assistant
                else ""
            )
        )
        for row in source_turns
    ]
    with _MODEL_ENCODE_LOCK:
        query_vectors = model.encode(
            list(queries),
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        document_vectors = model.encode(
            documents,
            normalize_embeddings=True,
            convert_to_numpy=True,
            batch_size=int(os.environ.get("QA_V5_EMBED_BATCH", "64")),
        )

    matrix = np.asarray(query_vectors) @ np.asarray(document_vectors).T
    return matrix.max(axis=0).tolist()


def _source_output_row(
    row: Dict[str, Any],
    reasons: Sequence[str],
    *,
    needs_assistant: bool,
) -> Dict[str, Any]:
    item = {
        "source_id": int(row.get("source_id") or 0),
        "source_uid": row.get("source_uid"),
        "session_id": row.get("session_id"),
        "timestamp": row.get("timestamp"),
        "user_content": _truncate(
            row.get("content"),
            int(os.environ.get("QA_V5_USER_CHAR_LIMIT", "1600")),
        ),
        "selection_reasons": list(reasons),
    }
    assistant_reply = str(row.get("assistant_reply") or "")
    if assistant_reply and needs_assistant:
        item["assistant_reply"] = _truncate(
            assistant_reply,
            int(os.environ.get("QA_V5_ASSISTANT_CHAR_LIMIT", "2200")),
        )
    return item


def targeted_source_evidence(
    question: str,
    parsed: Dict[str, Any],
    plan: Dict[str, Any],
    source_turns: Sequence[Dict[str, Any]],
    compact_sources: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    queries = _unique(
        [
            question,
            *(
                str(value)
                for value in plan.get("retrieval_queries") or []
            ),
            *(
                str(slot.get("query") or "")
                for slot in plan.get("slots") or []
                if isinstance(slot, dict)
            ),
            *(
                str(row.get("query_anchor") or "")
                for row in _hypotheses(parsed)
            ),
        ]
    )

    needs_assistant = question_needs_assistant(parsed, question)
    scoring_rows = (
        list(source_turns)
        if needs_assistant
        else [
            {
                **row,
                "assistant_reply": "",
            }
            for row in source_turns
        ]
    )
    idf = _idf_scores(scoring_rows)
    lexical_scores = [
        max(
            [
                _source_lexical_score(query, row, idf)
                for query in queries
            ]
            or [0.0]
        )
        for row in scoring_rows
    ]
    dense_scores = _dense_scores(
        queries,
        scoring_rows,
        include_assistant=needs_assistant,
    )

    ranked_indices = sorted(
        range(len(source_turns)),
        key=lambda index: (
            -(
                float(lexical_scores[index])
                + float(dense_scores[index])
                * float(os.environ.get("QA_V5_DENSE_WEIGHT", "18.0"))
            ),
            int(source_turns[index].get("source_id") or 0),
        ),
    )

    compact_ids = {
        int(row.get("source_id") or 0)
        for row in compact_sources
        if row.get("source_id") is not None
    }
    selected: Dict[int, Tuple[Dict[str, Any], Set[str]]] = {}
    top_k = int(os.environ.get("QA_V5_TARGET_SOURCE_K", "12"))
    adjacent_top = int(os.environ.get("QA_V5_TARGET_ADJACENT_TOP", "4"))
    adjacent_radius = int(os.environ.get("QA_V5_TARGET_ADJACENT_RADIUS", "1"))

    by_id, by_session = _source_index(source_turns)
    for rank, index in enumerate(ranked_indices[:top_k], start=1):
        row = source_turns[index]
        source_id = int(row.get("source_id") or 0)
        reasons = selected.setdefault(source_id, (row, set()))[1]
        reasons.add(
            f"target_rank_{rank}"
        )
        reasons.add(
            f"lexical_{lexical_scores[index]:.3f}"
        )
        reasons.add(
            f"dense_{dense_scores[index]:.3f}"
        )

        if rank <= adjacent_top:
            for adjacent in _adjacent_rows(
                row,
                by_session,
                adjacent_radius,
            ):
                adjacent_id = int(adjacent.get("source_id") or 0)
                adjacent_reasons = selected.setdefault(
                    adjacent_id,
                    (adjacent, set()),
                )[1]
                adjacent_reasons.add(
                    f"adjacent_to_target_rank_{rank}"
                )

    output: List[Dict[str, Any]] = []
    for source_id, (row, reasons) in selected.items():
        if source_id in compact_ids:
            reasons.add("already_in_compact")
        output.append(
            _source_output_row(
                row,
                sorted(reasons),
                needs_assistant=needs_assistant,
            )
        )

    output.sort(
        key=lambda row: (
            0
            if any(
                reason.startswith("target_rank_")
                for reason in row["selection_reasons"]
            )
            else 1,
            int(row.get("source_id") or 0),
        )
    )
    return output[: int(os.environ.get("QA_V5_MAX_TARGET_ROWS", "18"))]


def _merge_sources(
    compact: Sequence[Dict[str, Any]],
    targeted: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    merged: Dict[int, Dict[str, Any]] = {}
    for row in [*compact, *targeted]:
        source_id = int(row.get("source_id") or 0)
        if source_id not in merged:
            merged[source_id] = dict(row)
            continue
        existing = merged[source_id]
        existing["selection_reasons"] = _unique(
            [
                *(existing.get("selection_reasons") or []),
                *(row.get("selection_reasons") or []),
            ]
        )
        if row.get("assistant_reply") and not existing.get("assistant_reply"):
            existing["assistant_reply"] = row["assistant_reply"]
    return list(merged.values())


def _specialist_candidate(
    client: Any,
    *,
    question: str,
    question_date: str,
    plan: Dict[str, Any],
    parsed: Dict[str, Any],
    memories: Sequence[Dict[str, Any]],
    compact_sources: Sequence[Dict[str, Any]],
    targeted_sources: Sequence[Dict[str, Any]],
    meta: CallMeta,
) -> Dict[str, Any]:
    payload, result = client.json(
        _render(
            SPECIALIST_TEMPLATE_V5,
            question=question,
            question_date=question_date,
            plan_json=plan,
            memory_evidence_json=list(memories),
            compact_source_json=list(compact_sources),
            targeted_source_json=list(targeted_sources),
        ),
        system=SPECIALIST_SYSTEM_V5,
        max_tokens=int(os.environ.get("QA_V5_SPECIALIST_TOKENS", "3600")),
    )
    meta.add_result(result)
    if not isinstance(payload, dict):
        payload = {"answer": str(payload)}
    all_sources = _merge_sources(compact_sources, targeted_sources)
    return postprocess_candidate(
        payload,
        parsed=parsed,
        question=question,
        memories=memories,
        sources=all_sources,
    )


def _base_candidate_from_prediction(base: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "answer": base.get("prediction") or ABSTENTION,
        "answerable": bool(base.get("answerable")),
        "support_memory_ids": base.get("support_memory_ids") or [],
        "support_source_ids": base.get("support_source_ids") or [],
        "required_slots": base.get("required_slots") or [],
        "unfilled_slots": base.get("unfilled_slots") or [],
        "operation": base.get("operation") or "none",
        "calculation": base.get("calculation") or {
            "operator": "none",
            "operands": [],
        },
        "confidence": base.get("confidence") or 0.0,
        "reasoning_summary": base.get("reasoning_summary") or "",
    }


def _has_deterministic_result(candidate: Dict[str, Any]) -> bool:
    calculation = candidate.get("calculation") or {}
    return (
        str(calculation.get("operator") or "none") != "none"
        and bool(calculation.get("result"))
    )


def choose_candidate_safely(
    base: Dict[str, Any],
    specialist: Dict[str, Any],
    plan: Dict[str, Any],
) -> Tuple[Optional[str], str]:
    base_abstains = is_abstention(base.get("answer"))
    specialist_abstains = is_abstention(specialist.get("answer"))

    if not base_abstains and specialist_abstains:
        return "base", "monotonic guard: expanded evidence only caused abstention"
    if base_abstains and not specialist_abstains:
        return "specialist", "specialist filled an abstained base answer"

    contract = str(plan.get("answer_contract") or "")
    if contract in {"preference_criteria", "state_resolution"}:
        if specialist.get("answerable") and specialist.get("support_source_ids"):
            return "specialist", f"specialist satisfies {contract} contract"

    if _has_deterministic_result(base) and not _has_deterministic_result(specialist):
        return "base", "base has deterministic calculation"
    if _has_deterministic_result(specialist) and not _has_deterministic_result(base):
        return "specialist", "specialist has deterministic calculation"

    if normalize_text(base.get("answer")) == normalize_text(specialist.get("answer")):
        if candidate_score(specialist) > candidate_score(base):
            return "specialist", "same answer with stronger specialist support"
        return "base", "same answer; preserve compact candidate"

    return None, "requires arbiter"


def _arbiter_choice(
    client: Any,
    *,
    question: str,
    plan: Dict[str, Any],
    base: Dict[str, Any],
    specialist: Dict[str, Any],
    meta: CallMeta,
) -> Tuple[str, str]:
    safe_choice, reason = choose_candidate_safely(
        base,
        specialist,
        plan,
    )
    if safe_choice is not None:
        return safe_choice, reason

    payload, result = client.json(
        _render(
            ARBITER_TEMPLATE_V5,
            question=question,
            plan_json=plan,
            base_candidate_json=base,
            specialist_candidate_json=specialist,
        ),
        system=ARBITER_SYSTEM_V5,
        max_tokens=int(os.environ.get("QA_V5_ARBITER_TOKENS", "900")),
    )
    meta.add_result(result)
    choice = (
        str(payload.get("choice") or "base").casefold()
        if isinstance(payload, dict)
        else "base"
    )
    if choice not in {"base", "specialist"}:
        choice = "base"
    arbiter_reason = (
        str(payload.get("reason") or "")
        if isinstance(payload, dict)
        else ""
    )
    return choice, arbiter_reason or reason


def run_sample(
    sample_dir: Path,
    *,
    source_retrieval_name: str,
    base_output_name: str,
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

    base_path = (
        sample_dir
        / "qa_v2"
        / base_output_name
        / "prediction.json"
    )
    compact_source_path = (
        sample_dir
        / "qa_v2"
        / base_output_name
        / "source_evidence.json"
    )
    if not base_path.exists() or not compact_source_path.exists():
        raise FileNotFoundError(
            f"Run the compact base QA first: {base_output_name}"
        )

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
    base_prediction = read_json(base_path)
    compact_sources = read_json(compact_source_path)
    base = _base_candidate_from_prediction(base_prediction)

    memories = memory_evidence(
        top_records,
        limit=int(os.environ.get("QA_V5_MEMORY_K", "18")),
    )

    client = make_client()
    meta = CallMeta()
    specialist_needed, risk_reasons = needs_specialist(
        base_prediction,
        question,
        parsed,
    )

    trace: Dict[str, Any] = {
        "base_output_name": base_output_name,
        "source_retrieval_name": source_retrieval_name,
        "risk_reasons": risk_reasons,
        "specialist_needed": specialist_needed,
    }

    final = base
    plan: Dict[str, Any] = {
        "answer_contract": infer_contract_hint(question, parsed),
        "operator": "none",
        "slots": [],
        "retrieval_queries": [],
    }
    targeted_sources: List[Dict[str, Any]] = []
    specialist: Optional[Dict[str, Any]] = None
    choice = "base"
    choice_reason = "base candidate passed risk gate"

    if specialist_needed:
        plan = _planner_payload(
            client,
            question=question,
            question_date=question_date,
            parsed=parsed,
            base=base,
            meta=meta,
        )
        targeted_sources = targeted_source_evidence(
            question,
            parsed,
            plan,
            source_turns,
            compact_sources,
        )
        specialist = _specialist_candidate(
            client,
            question=question,
            question_date=question_date,
            plan=plan,
            parsed=parsed,
            memories=memories,
            compact_sources=compact_sources,
            targeted_sources=targeted_sources,
            meta=meta,
        )
        choice, choice_reason = _arbiter_choice(
            client,
            question=question,
            plan=plan,
            base=base,
            specialist=specialist,
            meta=meta,
        )
        final = specialist if choice == "specialist" else base

    qa_output = {
        "question_id": item.get("question_id"),
        "question_type": normalize_question_type(
            item.get("question_type") or sample_dir.parent.name
        ),
        "question": question,
        "question_date": question_date,
        "gold_answer": gold_answer,
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
        "source_retrieval_name": source_retrieval_name,
        "base_output_name": base_output_name,
        "adaptive_choice": choice,
        "adaptive_choice_reason": choice_reason,
        "risk_reasons": risk_reasons,
        "plan": plan,
        "meta": {
            "mode": "adaptive_qa_v5",
            **meta.to_dict(),
        },
    }

    qa_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(qa_path, qa_output)
    write_json(
        qa_path.parent / "adaptive_trace.json",
        {
            **trace,
            "plan": plan,
            "base_candidate": base,
            "specialist_candidate": specialist,
            "choice": choice,
            "choice_reason": choice_reason,
            "compact_source_ids": [
                row.get("source_id") for row in compact_sources
            ],
            "targeted_source_ids": [
                row.get("source_id") for row in targeted_sources
            ],
        },
    )
    write_json(
        qa_path.parent / "targeted_source_evidence.json",
        targeted_sources,
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
        "specialist_needed": specialist_needed,
        "choice": choice,
        "prompt_tokens": meta.prompt_tokens,
        "completion_tokens": meta.completion_tokens,
        "targeted_source_count": len(targeted_sources),
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
                base_output_name=args.base_output_name,
                output_name=args.output_name,
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
                f"[adaptive-qa-v5] {completed}/{len(samples)} | "
                f"{sample_dir.name} | {row.get('status')} | "
                f"choice={row.get('choice', '-')}",
                flush=True,
            )

    correct = sum(1 for row in rows if row.get("correct"))
    summary = {
        "source_retrieval_name": args.source_retrieval_name,
        "base_output_name": args.base_output_name,
        "output_name": args.output_name,
        "n": len(rows),
        "correct": correct,
        "accuracy": correct / len(rows) if rows else 0.0,
        "workers": workers,
        "rows": rows,
    }
    write_json(
        run_root / f"adaptive_qa_v5_manifest_{args.output_name}.json",
        summary,
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run compact-first adaptive QA with specialist escalation "
            "and targeted dense source retrieval."
        )
    )
    parser.add_argument("--run-root", required=True)
    parser.add_argument(
        "--source-retrieval-name",
        default="graph_active_p2_gate",
    )
    parser.add_argument(
        "--base-output-name",
        default="graph_active_p5_source_qa",
    )
    parser.add_argument(
        "--output-name",
        default="graph_active_p7_adaptive_qa",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--question-types", default="")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    summary = run_all(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
