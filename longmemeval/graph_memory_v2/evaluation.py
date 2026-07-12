from __future__ import annotations

import json
import math
import random
import re
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, Iterable, List, Optional, Sequence, Tuple

from .active_agent import ActiveReconstructionAgent
from .embedding import HashEmbedder, SentenceTransformerEmbedder
from .graph_store import GraphBuilder, GraphStore
from .io_utils import normalize_question_type, normalize_text, read_json, write_json
from .llm_client import OpenAICompatibleClient
from .parallel import run_parallel, should_checkpoint
from .prompts import JUDGE_SYSTEM, JUDGE_TEMPLATE, QA_SYSTEM, QA_TEMPLATE, render
from .retrieval import StaticRetriever
from .schemas import ParsedQueryV2, RetrievalRecord, memories_from_payload


VALID_MODES = {"dimmem_v1", "dimension_v2", "graph_static", "graph_active"}


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


def _make_embedder(kind: str, model_name: str = "", device: str = "cpu"):
    if kind == "sentence-transformers":
        return SentenceTransformerEmbedder(model_name or "sentence-transformers/all-MiniLM-L6-v2", device=device)
    return HashEmbedder()


def run_retrieval_sample(
    sample_dir: Path,
    *,
    mode: str,
    output_name: str,
    route_k: int = 20,
    initial_k: int = 12,
    final_k: int = 15,
    max_rounds: int = 3,
    active_client: Optional[OpenAICompatibleClient] = None,
    router_mode: str = "llm",
    embedder_kind: str = "hash",
    embedding_model: str = "",
    device: str = "cpu",
    force: bool = False,
    embedder: Optional[Any] = None,
    route_workers: int = 0,
    tool_workers: int = 0,
) -> Dict[str, Any]:
    if mode not in VALID_MODES:
        raise ValueError(f"unknown mode {mode}; expected one of {sorted(VALID_MODES)}")
    output_dir = sample_dir / "retrieval_v2" / output_name
    output_path = output_dir / "top_records.json"
    if output_path.exists() and not force:
        rows = read_json(output_path)
        return {"status": "existing", "record_count": len(rows), "path": str(output_path)}

    parsed = ParsedQueryV2.from_dict(read_json(sample_dir / "query_v2" / "parsed_query.json"))
    original = mode == "dimmem_v1"
    if original:
        v1_path = sample_dir / "memory_v1" / "all_memories.json"
        if not v1_path.exists():
            raise FileNotFoundError(f"run V1 extraction first: {v1_path}")
        store = GraphBuilder(add_weak_same_event_edges=False).build(memories_from_payload(read_json(v1_path)))
    else:
        store = GraphStore.load(sample_dir / "graph_v2")
    active_embedder = embedder or _make_embedder(embedder_kind, embedding_model, device)
    retriever = StaticRetriever(
        store,
        embedder=active_embedder,
        original_projection=original,
        route_workers=route_workers,
    )

    started = time.time()
    if mode == "graph_active":
        agent = ActiveReconstructionAgent(
            store=store,
            retriever=retriever,
            client=active_client,
            max_rounds=max_rounds,
            final_k=final_k,
            router_mode=router_mode,
            tool_workers=tool_workers,
        )
        records, trace = agent.retrieve(parsed, route_k=route_k, initial_k=initial_k)
    else:
        records = retriever.retrieve(
            parsed,
            route_k=route_k,
            final_k=final_k,
            graph_expand=mode == "graph_static",
            graph_expand_k=max(final_k, 20),
        )
        trace = {
            "mode": mode,
            "query": parsed.to_dict(),
            "final": [record.to_dict() for record in records],
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [record.to_dict() for record in records]
    write_json(output_path, rows)
    write_json(output_dir / "retrieval_trace.json", trace)
    write_json(
        output_dir / "retrieval_meta.json",
        {
            "mode": mode,
            "route_k": route_k,
            "initial_k": initial_k,
            "final_k": final_k,
            "max_rounds": max_rounds,
            "router_mode": router_mode,
            "embedder_kind": embedder_kind,
            "embedding_model": embedding_model,
            "route_workers": route_workers,
            "tool_workers": tool_workers,
            "elapsed_seconds": time.time() - started,
        },
    )
    # Simple compatibility view for scripts expecting memory content plus dimensions.
    write_json(
        output_dir / "top_records_compat.json",
        [
            {
                "memory_id": row["memory_id"],
                "content": row["content"],
                "memory_type": row["dimension"]["memory_type"],
                "time": row["dimension"]["time"],
                "location": row["dimension"]["locations"],
                "reason": row["dimension"]["reason"],
                "purpose": row["dimension"]["purpose"],
                "keywords": row["dimension"]["keywords"],
                "score": row["score"],
            }
            for row in rows
        ],
    )
    return {"status": "ok", "record_count": len(rows), "path": str(output_path)}


def run_retrieval_run(
    run_root: str,
    *,
    mode: str,
    output_name: str,
    route_k: int = 20,
    initial_k: int = 12,
    final_k: int = 15,
    max_rounds: int = 3,
    active_client: Optional[OpenAICompatibleClient] = None,
    router_mode: str = "llm",
    embedder_kind: str = "hash",
    embedding_model: str = "",
    device: str = "cpu",
    force: bool = False,
    question_types: Optional[Iterable[str]] = None,
    workers: int = 0,
    fail_fast: bool = False,
    shared_embedder: Optional[Any] = None,
    route_workers: int = 0,
    tool_workers: int = 0,
) -> Dict[str, Any]:
    root = Path(run_root)
    manifest = read_json(root / "run_manifest.json")
    allowed = {normalize_question_type(value) for value in question_types or []}
    samples = [
        sample for sample in (manifest.get("samples") or [])
        if not allowed or sample.get("question_type") in allowed
    ]
    # Load the embedding model once. Case workers share it; the sentence-transformer
    # wrapper serializes encode() safely to avoid one GPU model per case.
    embedder = shared_embedder or _make_embedder(embedder_kind, embedding_model, device)
    manifest_path = root / f"retrieval_manifest_{output_name}.json"
    checkpoint_rows: List[Optional[Dict[str, Any]]] = [None] * len(samples)

    def worker(sample: Dict[str, Any]) -> Dict[str, Any]:
        return run_retrieval_sample(
            Path(sample["sample_dir"]),
            mode=mode,
            output_name=output_name,
            route_k=route_k,
            initial_k=initial_k,
            final_k=final_k,
            max_rounds=max_rounds,
            active_client=active_client,
            router_mode=router_mode,
            embedder_kind=embedder_kind,
            embedding_model=embedding_model,
            device=device,
            force=force,
            embedder=embedder,
            route_workers=route_workers,
            tool_workers=tool_workers,
        )

    def on_result(index: int, result: Dict[str, Any], completed: int, total: int) -> None:
        checkpoint_rows[index] = result
        if not should_checkpoint(completed, total):
            return
        write_json(manifest_path, {
            "mode": mode,
            "parallel": {"requested_workers": workers, "completed": completed, "total": total},
            "samples": [row for row in checkpoint_rows if row is not None],
        })

    results, stats = run_parallel(
        samples,
        worker,
        workers=workers,
        stage=f"retrieve-{output_name}",
        merge_input=True,
        fail_fast=fail_fast,
        on_result=on_result,
    )
    write_json(manifest_path, {"mode": mode, "parallel": stats.to_dict(), "samples": results})
    return {"samples": results, "parallel": stats.to_dict()}

def _evidence_for_qa(rows: Sequence[Dict[str, Any]], need_assistant: bool) -> List[Dict[str, Any]]:
    evidence: List[Dict[str, Any]] = []
    for row in rows:
        item = {
            "memory_id": row.get("memory_id"),
            "content": row.get("content"),
            "dimension": row.get("dimension"),
            "provenance": row.get("provenance"),
        }
        if need_assistant:
            replies = row.get("assistant_replies") or []
            if replies:
                item["assistant_replies"] = replies
        evidence.append(item)
    return evidence


def heuristic_answer(question: str, evidence: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not evidence:
        return {"answer": "Cannot be determined from the conversation.", "support_memory_ids": [], "confidence": 0.0, "reasoning_summary": "No evidence."}
    first = evidence[0]
    return {
        "answer": str(first.get("content") or "Cannot be determined from the conversation."),
        "support_memory_ids": [first.get("memory_id")],
        "confidence": 0.1,
        "reasoning_summary": "Smoke-test answer copied from the highest-ranked memory.",
    }


def run_qa_sample(
    sample_dir: Path,
    *,
    retrieval_name: str,
    client: Optional[OpenAICompatibleClient],
    heuristic: bool = False,
    force: bool = False,
) -> Dict[str, Any]:
    output_dir = sample_dir / "qa_v2" / retrieval_name
    output_path = output_dir / "prediction.json"
    if output_path.exists() and not force:
        return {"status": "existing", "path": str(output_path)}
    item = read_json(sample_dir / "input_item.json")
    question, question_date, gold_answer = _question_and_gold(item)
    parsed = ParsedQueryV2.from_dict(read_json(sample_dir / "query_v2" / "parsed_query.json"))
    rows = read_json(sample_dir / "retrieval_v2" / retrieval_name / "top_records.json")
    evidence = _evidence_for_qa(rows, parsed.primary().need_assistant_context)
    if heuristic:
        payload = heuristic_answer(question, evidence)
        meta = {"mode": "heuristic", "prompt_tokens": 0, "completion_tokens": 0}
    else:
        if client is None:
            raise ValueError("QA client required unless heuristic=True")
        payload, result = client.json(
            render(
                QA_TEMPLATE,
                question=question,
                question_date=question_date,
                evidence_json=evidence,
            ),
            system=QA_SYSTEM,
            max_tokens=1600,
        )
        if not isinstance(payload, dict):
            payload = {"answer": str(payload)}
        meta = {
            "mode": "llm",
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "elapsed_seconds": result.elapsed_seconds,
        }
    output = {
        "question_id": item.get("question_id"),
        "question_type": normalize_question_type(item.get("question_type") or sample_dir.parent.name),
        "question": question,
        "question_date": question_date,
        "gold_answer": gold_answer,
        "prediction": str(payload.get("answer") or "").strip(),
        "support_memory_ids": payload.get("support_memory_ids") or [],
        "confidence": payload.get("confidence", 0.0),
        "reasoning_summary": payload.get("reasoning_summary") or "",
        "retrieval_name": retrieval_name,
        "meta": meta,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_path, output)
    return {"status": "ok", "path": str(output_path), **meta}


def exact_judge(gold: str, prediction: str) -> Tuple[bool, str]:
    gold_norm = normalize_text(gold)
    pred_norm = normalize_text(prediction)
    if not gold_norm:
        return False, "Empty gold answer."
    correct = gold_norm == pred_norm or gold_norm in pred_norm or pred_norm in gold_norm
    return correct, "normalized exact/containment match" if correct else "normalized text mismatch"


def run_judge_sample(
    sample_dir: Path,
    *,
    retrieval_name: str,
    client: Optional[OpenAICompatibleClient],
    exact: bool = False,
    force: bool = False,
) -> Dict[str, Any]:
    qa_path = sample_dir / "qa_v2" / retrieval_name / "prediction.json"
    output_path = sample_dir / "judge_v2" / retrieval_name / "judge.json"
    if output_path.exists() and not force:
        return {"status": "existing", "path": str(output_path)}
    qa = read_json(qa_path)
    if exact or client is None:
        correct, reason = exact_judge(qa.get("gold_answer", ""), qa.get("prediction", ""))
        meta = {"mode": "exact"}
    else:
        payload, result = client.json(
            render(
                JUDGE_TEMPLATE,
                question_type=qa.get("question_type"),
                question=qa.get("question"),
                gold_answer=qa.get("gold_answer"),
                prediction=qa.get("prediction"),
            ),
            system=JUDGE_SYSTEM,
            max_tokens=800,
        )
        label = str(payload.get("label") if isinstance(payload, dict) else "").upper()
        correct = label == "CORRECT"
        reason = str(payload.get("reason") if isinstance(payload, dict) else "")
        meta = {
            "mode": "llm",
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "elapsed_seconds": result.elapsed_seconds,
        }
    output = {**qa, "correct": bool(correct), "judge_reason": reason, "judge_meta": meta}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(output_path, output)
    return {"status": "ok", "path": str(output_path), "correct": bool(correct)}


def run_qa_judge_run(
    run_root: str,
    *,
    retrieval_name: str,
    qa_client: Optional[OpenAICompatibleClient],
    judge_client: Optional[OpenAICompatibleClient],
    heuristic_qa: bool = False,
    exact_judge_mode: bool = False,
    force: bool = False,
    question_types: Optional[Iterable[str]] = None,
    workers: int = 0,
    fail_fast: bool = False,
) -> Dict[str, Any]:
    """Run QA for all cases in parallel, then Judge for all cases in parallel.

    Judge depends on the prediction for the same case, so the two phases remain
    ordered. Within each phase, every selected LongMemEval case is concurrent.
    """
    root = Path(run_root)
    manifest = read_json(root / "run_manifest.json")
    allowed = {normalize_question_type(value) for value in question_types or []}
    samples = [
        sample for sample in (manifest.get("samples") or [])
        if not allowed or sample.get("question_type") in allowed
    ]
    manifest_path = root / f"qa_judge_manifest_{retrieval_name}.json"

    def qa_worker(sample: Dict[str, Any]) -> Dict[str, Any]:
        return run_qa_sample(
            Path(sample["sample_dir"]),
            retrieval_name=retrieval_name,
            client=qa_client,
            heuristic=heuristic_qa,
            force=force,
        )

    qa_results, qa_stats = run_parallel(
        samples,
        qa_worker,
        workers=workers,
        stage=f"qa-{retrieval_name}",
        merge_input=True,
        fail_fast=fail_fast,
    )

    qa_by_id = {str(row.get("sample_id") or row.get("question_id")): row for row in qa_results}
    judge_inputs = [
        sample for sample in samples
        if qa_by_id.get(str(sample.get("sample_id") or sample.get("question_id")), {}).get("status") != "failed"
    ]

    def judge_worker(sample: Dict[str, Any]) -> Dict[str, Any]:
        return run_judge_sample(
            Path(sample["sample_dir"]),
            retrieval_name=retrieval_name,
            client=judge_client,
            exact=exact_judge_mode,
            force=force,
        )

    judge_results, judge_stats = run_parallel(
        judge_inputs,
        judge_worker,
        workers=workers,
        stage=f"judge-{retrieval_name}",
        merge_input=True,
        fail_fast=fail_fast,
    )
    judge_by_id = {str(row.get("sample_id") or row.get("question_id")): row for row in judge_results}

    merged: List[Dict[str, Any]] = []
    for sample in samples:
        key = str(sample.get("sample_id") or sample.get("question_id"))
        qa_result = qa_by_id.get(key, {"status": "failed", "error": "QA result missing"})
        judge_result = judge_by_id.get(key, {"status": "skipped", "error": "Judge skipped because QA failed"})
        merged.append({
            **{k: v for k, v in sample.items() if not str(k).startswith("_")},
            "status": "ok" if qa_result.get("status") != "failed" and judge_result.get("status") != "failed" else "failed",
            "qa": qa_result,
            "judge": judge_result,
        })

    output = {
        "qa_parallel": qa_stats.to_dict(),
        "judge_parallel": judge_stats.to_dict(),
        "samples": merged,
    }
    write_json(manifest_path, output)
    return output


def _gold_session_ids(item: Dict[str, Any]) -> List[str]:
    values = (
        item.get("answer_session_ids")
        or item.get("gold_session_ids")
        or item.get("answer_sessions")
        or []
    )
    if isinstance(values, str):
        values = [values]
    return [str(value) for value in values if str(value).strip()] if isinstance(values, list) else []


def build_report(
    run_root: str,
    retrieval_name: str,
    *,
    workers: int = 0,
    fail_fast: bool = False,
) -> Dict[str, Any]:
    root = Path(run_root)
    manifest = read_json(root / "run_manifest.json")
    samples = list(manifest.get("samples") or [])

    def worker(sample: Dict[str, Any]) -> Dict[str, Any]:
        sample_dir = Path(sample["sample_dir"])
        path = sample_dir / "judge_v2" / retrieval_name / "judge.json"
        if not path.exists():
            return {"status": "missing"}
        row = read_json(path)
        item = read_json(sample_dir / "input_item.json")
        gold_sessions = set(_gold_session_ids(item))
        session_recall = None
        retrieval_path = sample_dir / "retrieval_v2" / retrieval_name / "top_records.json"
        if gold_sessions and retrieval_path.exists():
            retrieved_sessions = {
                str(record.get("provenance", {}).get("session_id") or "")
                for record in read_json(retrieval_path)
            }
            session_recall = len(gold_sessions & retrieved_sessions) / len(gold_sessions)
        retrieval_seconds = None
        meta_path = sample_dir / "retrieval_v2" / retrieval_name / "retrieval_meta.json"
        if meta_path.exists():
            retrieval_seconds = float(read_json(meta_path).get("elapsed_seconds") or 0.0)
        rounds_count = None
        tool_count = None
        trace_path = sample_dir / "retrieval_v2" / retrieval_name / "retrieval_trace.json"
        if trace_path.exists():
            trace = read_json(trace_path)
            rounds = trace.get("rounds") or []
            rounds_count = len(rounds)
            tool_count = sum(len(r.get("tool_results") or []) for r in rounds if isinstance(r, dict))
        return {
            "status": "ok",
            "row": row,
            "session_recall": session_recall,
            "retrieval_seconds": retrieval_seconds,
            "active_rounds": rounds_count,
            "tool_calls": tool_count,
        }

    gathered, stats = run_parallel(
        samples,
        worker,
        workers=workers,
        stage=f"report-{retrieval_name}",
        merge_input=False,
        fail_fast=fail_fast,
    )
    gathered = [item for item in gathered if isinstance(item, dict) and item.get("status") == "ok"]
    rows = [item["row"] for item in gathered]
    by_type: DefaultDict[str, List[bool]] = defaultdict(list)
    session_recalls = [float(item["session_recall"]) for item in gathered if item.get("session_recall") is not None]
    retrieval_times = [float(item["retrieval_seconds"]) for item in gathered if item.get("retrieval_seconds") is not None]
    active_rounds = [int(item["active_rounds"]) for item in gathered if item.get("active_rounds") is not None]
    tool_calls = [int(item["tool_calls"]) for item in gathered if item.get("tool_calls") is not None]
    token_totals = {"qa_prompt": 0, "qa_completion": 0, "judge_prompt": 0, "judge_completion": 0}
    for row in rows:
        by_type[normalize_question_type(row.get("question_type"))].append(bool(row.get("correct")))
        qa_meta = row.get("meta") or {}
        judge_meta = row.get("judge_meta") or {}
        token_totals["qa_prompt"] += int(qa_meta.get("prompt_tokens") or 0)
        token_totals["qa_completion"] += int(qa_meta.get("completion_tokens") or 0)
        token_totals["judge_prompt"] += int(judge_meta.get("prompt_tokens") or 0)
        token_totals["judge_completion"] += int(judge_meta.get("completion_tokens") or 0)

    summary = {
        qtype: {
            "n": len(values),
            "correct": sum(values),
            "accuracy": sum(values) / len(values) if values else 0.0,
        }
        for qtype, values in sorted(by_type.items())
    }
    all_values = [bool(row.get("correct")) for row in rows]
    report = {
        "retrieval_name": retrieval_name,
        "n": len(all_values),
        "correct": sum(all_values),
        "accuracy": sum(all_values) / len(all_values) if all_values else 0.0,
        "parallel": stats.to_dict(),
        "by_question_type": summary,
        "retrieval_metrics": {
            "gold_session_recall_mean": (sum(session_recalls) / len(session_recalls)) if session_recalls else None,
            "gold_session_recall_n": len(session_recalls),
            "mean_retrieval_seconds": (sum(retrieval_times) / len(retrieval_times)) if retrieval_times else None,
            "mean_active_rounds": (sum(active_rounds) / len(active_rounds)) if active_rounds else None,
            "mean_tool_calls": (sum(tool_calls) / len(tool_calls)) if tool_calls else None,
        },
        "token_totals": token_totals,
        "rows": rows,
    }
    report_dir = root / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    write_json(report_dir / f"report_{retrieval_name}.json", report)
    markdown = [
        f"# Report: {retrieval_name}",
        "",
        f"- N: {report['n']}",
        f"- Correct: {report['correct']}",
        f"- Accuracy: {report['accuracy']:.4f}",
        f"- Case workers: {stats.workers}",
        f"- Gold-session recall: {report['retrieval_metrics']['gold_session_recall_mean']}",
        f"- Mean retrieval seconds: {report['retrieval_metrics']['mean_retrieval_seconds']}",
        "",
        "| Question type | N | Correct | Accuracy |",
        "|---|---:|---:|---:|",
    ]
    for qtype, values in report["by_question_type"].items():
        markdown.append(f"| {qtype} | {values['n']} | {values['correct']} | {values['accuracy']:.4f} |")
    (report_dir / f"report_{retrieval_name}.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    return report

def _paired_binomial_p_value(fixed_count: int, broken_count: int) -> float:
    n = fixed_count + broken_count
    if n == 0:
        return 1.0
    tail = min(fixed_count, broken_count)
    probability = sum(math.comb(n, index) for index in range(tail + 1)) / (2 ** n)
    return min(1.0, 2.0 * probability)


def _bootstrap_delta_ci(pairs: Sequence[Tuple[int, int]], samples: int = 5000) -> Tuple[float, float]:
    if not pairs:
        return (0.0, 0.0)
    rng = random.Random(42)
    deltas: List[float] = []
    n = len(pairs)
    for _ in range(samples):
        draw = [pairs[rng.randrange(n)] for _ in range(n)]
        deltas.append(sum(right - left for left, right in draw) / n)
    deltas.sort()
    return deltas[int(0.025 * (samples - 1))], deltas[int(0.975 * (samples - 1))]


def compare_reports(
    run_root: str,
    baseline: str,
    candidate: str,
    *,
    workers: int = 0,
    fail_fast: bool = False,
) -> Dict[str, Any]:
    root = Path(run_root)

    def report_worker(name: str) -> Dict[str, Any]:
        report_path = root / "reports" / f"report_{name}.json"
        if report_path.exists():
            return {"name": name, "report": read_json(report_path), "status": "existing"}
        return {
            "name": name,
            "report": build_report(run_root, name, workers=workers, fail_fast=fail_fast),
            "status": "ok",
        }

    report_rows, report_stats = run_parallel(
        [baseline, candidate],
        report_worker,
        workers=0,
        stage="compare-report-load",
        progress=False,
        fail_fast=True,
    )
    reports = {row["name"]: row["report"] for row in report_rows}
    base = reports[baseline]
    cand = reports[candidate]
    base_by_id = {str(row.get("question_id")): row for row in base["rows"]}
    cand_by_id = {str(row.get("question_id")): row for row in cand["rows"]}
    paired_ids = sorted(set(base_by_id) & set(cand_by_id))

    def pair_worker(qid: str) -> Dict[str, Any]:
        left = bool(base_by_id[qid].get("correct"))
        right = bool(cand_by_id[qid].get("correct"))
        if not left and right:
            state = "fixed"
        elif left and not right:
            state = "broken"
        elif left:
            state = "unchanged_correct"
        else:
            state = "unchanged_wrong"
        return {"question_id": qid, "left": int(left), "right": int(right), "state": state}

    pair_rows, pair_stats = run_parallel(
        paired_ids,
        pair_worker,
        workers=workers,
        stage=f"compare-{baseline}-vs-{candidate}",
        progress=False,
        fail_fast=fail_fast,
    )
    fixed = [row["question_id"] for row in pair_rows if row["state"] == "fixed"]
    broken = [row["question_id"] for row in pair_rows if row["state"] == "broken"]
    unchanged_correct = sum(row["state"] == "unchanged_correct" for row in pair_rows)
    unchanged_wrong = sum(row["state"] == "unchanged_wrong" for row in pair_rows)
    pairs = [(int(row["left"]), int(row["right"])) for row in pair_rows]
    baseline_paired_accuracy = sum(left for left, _ in pairs) / len(pairs) if pairs else 0.0
    candidate_paired_accuracy = sum(right for _, right in pairs) / len(pairs) if pairs else 0.0
    ci_low, ci_high = _bootstrap_delta_ci(pairs)
    result = {
        "baseline": baseline,
        "candidate": candidate,
        "paired_n": len(paired_ids),
        "baseline_accuracy": baseline_paired_accuracy,
        "candidate_accuracy": candidate_paired_accuracy,
        "delta": candidate_paired_accuracy - baseline_paired_accuracy,
        "delta_bootstrap_95ci": [ci_low, ci_high],
        "paired_binomial_p_value": _paired_binomial_p_value(len(fixed), len(broken)),
        "fixed_count": len(fixed),
        "broken_count": len(broken),
        "fixed_question_ids": fixed,
        "broken_question_ids": broken,
        "unchanged_correct": unchanged_correct,
        "unchanged_wrong": unchanged_wrong,
        "parallel": {
            "report_loading": report_stats.to_dict(),
            "paired_cases": pair_stats.to_dict(),
        },
    }
    report_dir = root / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    write_json(report_dir / f"compare_{baseline}_vs_{candidate}.json", result)
    return result
