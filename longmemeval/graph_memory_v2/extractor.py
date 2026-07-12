from __future__ import annotations

import json
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .io_utils import ensure_dir, normalize_text, read_json, stable_hash, write_json
from .llm_client import OpenAICompatibleClient
from .parallel import run_parallel, should_checkpoint
from .prompts import MEMORY_EXTRACTION_SYSTEM, MEMORY_EXTRACTION_TEMPLATE, render
from .schemas import EnhancedDimension, MemoryRecord, Provenance


def _window_files(sample_dir: Path) -> List[Path]:
    return sorted(
        path for path in (sample_dir / "windows").glob("window_*.json")
        if not path.name.endswith("_assistant_replies.json")
    )


def _record_key(record: MemoryRecord) -> str:
    dim = record.dimension
    parts = [
        normalize_text(record.content),
        dim.memory_type,
        normalize_text(dim.state_key),
        normalize_text(dim.state_value),
        normalize_text(dim.time.primary()),
        normalize_text(dim.relation.canonical_key()),
        normalize_text(dim.preference.target),
        dim.preference.polarity,
    ]
    return "|".join(parts)


def _merge_duplicate(old: MemoryRecord, new: MemoryRecord) -> MemoryRecord:
    old.provenance.source_ids = sorted(set(old.provenance.source_ids + new.provenance.source_ids))
    old.provenance.source_uids = list(dict.fromkeys(old.provenance.source_uids + new.provenance.source_uids))
    old.provenance.source_times = list(dict.fromkeys(old.provenance.source_times + new.provenance.source_times))
    old.assistant_replies = list(dict.fromkeys(old.assistant_replies + new.assistant_replies))
    old.confidence = max(old.confidence, new.confidence)
    if len(new.evidence_span) > len(old.evidence_span):
        old.evidence_span = new.evidence_span
    return old


def _normalise_extracted_memory(
    raw: Dict[str, Any],
    *,
    window: Dict[str, Any],
    sample_id: str,
) -> Optional[MemoryRecord]:
    content = str(raw.get("content") or "").strip()
    if not content:
        return None

    raw_source_ids = raw.get("source_ids") or []
    if isinstance(raw_source_ids, (int, str)):
        raw_source_ids = [raw_source_ids]
    source_ids: List[int] = []
    for value in raw_source_ids:
        try:
            source_ids.append(int(value))
        except (TypeError, ValueError):
            continue

    overlap_count = int(window.get("overlap_count") or 0)
    allowed_local_ids = {
        int(message.get("window_source_id"))
        for message in window.get("messages") or []
        if int(message.get("window_source_id") or 0) > overlap_count
    }
    source_ids = sorted(set(value for value in source_ids if value in allowed_local_ids))
    if not source_ids:
        return None

    by_local = {
        int(message.get("window_source_id")): message
        for message in window.get("messages") or []
    }
    source_rows = [by_local[value] for value in source_ids if value in by_local]
    if not source_rows:
        return None

    dimension = EnhancedDimension.from_dict(raw.get("dimension") or raw)
    if not dimension.memory_type:
        # Conservative fallback based on dimensions, not benchmark categories.
        if dimension.preference.target:
            dimension.memory_type = "profile"
        elif dimension.time.primary() or dimension.modality == "planned":
            dimension.memory_type = "episodic"
        else:
            dimension.memory_type = "fact"

    source_uids = [str(row.get("source_uid") or "") for row in source_rows if row.get("source_uid")]
    source_times = [str(row.get("timestamp") or "") for row in source_rows if row.get("timestamp")]
    global_source_ids = [int(row.get("source_id") or 0) for row in source_rows if int(row.get("source_id") or 0) > 0]
    session_ids = [str(row.get("session_id") or "") for row in source_rows]
    session_indexes = [int(row.get("session_index", -1)) for row in source_rows]
    assistant_replies = [
        str(row.get("assistant_reply") or "").strip()
        for row in source_rows
        if str(row.get("assistant_reply") or "").strip()
    ]

    fingerprint = json.dumps(
        {
            "sample": sample_id,
            "content": normalize_text(content),
            "state_key": normalize_text(dimension.state_key),
            "state_value": normalize_text(dimension.state_value),
            "time": dimension.time.primary(),
            "sources": source_uids,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return MemoryRecord(
        memory_id="m_" + stable_hash(fingerprint, 20),
        content=content,
        dimension=dimension,
        provenance=Provenance(
            session_id=session_ids[0] if len(set(session_ids)) == 1 else "",
            session_index=session_indexes[0] if len(set(session_indexes)) == 1 else -1,
            source_ids=global_source_ids,
            source_uids=source_uids,
            source_times=source_times,
            source_role="user",
            window_index=int(window.get("window_index", -1)),
        ),
        confidence=max(0.0, min(1.0, float(raw.get("confidence") or 0.0))),
        evidence_span=str(raw.get("evidence_span") or "").strip(),
        assistant_replies=list(dict.fromkeys(assistant_replies)),
    )


def heuristic_extract_window(window: Dict[str, Any], sample_id: str) -> List[MemoryRecord]:
    """Deterministic smoke-test extractor.

    It intentionally does not claim benchmark quality. It turns each eligible user
    turn into one grounded memory so the complete pipeline can be tested without an
    API. Full experiments should use the LLM extractor.
    """
    records: List[MemoryRecord] = []
    overlap_count = int(window.get("overlap_count") or 0)
    for message in window.get("messages") or []:
        local_id = int(message.get("window_source_id") or 0)
        if local_id <= overlap_count:
            continue
        content = str(message.get("content") or "").strip()
        if not content:
            continue
        lower = content.casefold()
        modality = "planned" if re.search(r"\b(plan|planning|will|going to|intend)\b", lower) else "asserted"
        memory_type = "profile" if re.search(r"\b(prefer|favorite|like|dislike|love|hate)\b", lower) else (
            "episodic" if re.search(r"\b(yesterday|today|tomorrow|last|next|went|visited|met|bought|started|finished)\b", lower) else "fact"
        )
        raw = {
            "source_ids": [local_id],
            "content": content,
            "dimension": {
                "memory_type": memory_type,
                "time": {"raw": str(message.get("timestamp") or ""), "precision": "datetime"},
                "entities": [],
                "locations": [],
                "topic": "",
                "relation": {},
                "state_key": "",
                "state_value": "",
                "state_status": "planned" if modality == "planned" else "active",
                "reason": "",
                "purpose": "",
                "preference": {},
                "modality": modality,
                "keywords": re.findall(r"[A-Za-z0-9][A-Za-z0-9_-]*", content)[:8],
            },
            "confidence": 0.25,
            "evidence_span": content[:200],
        }
        record = _normalise_extracted_memory(raw, window=window, sample_id=sample_id)
        if record:
            records.append(record)
    return records


def extract_sample(
    sample_dir: Path,
    *,
    client: Optional[OpenAICompatibleClient],
    force: bool = False,
    heuristic: bool = False,
    max_tokens: int = 5000,
    window_workers: int = 1,
) -> Dict[str, Any]:
    output_dir = ensure_dir(sample_dir / "memory_v2")
    final_path = output_dir / "all_memories.json"
    trace_path = output_dir / "extraction_trace.json"
    if final_path.exists() and not force:
        memories = [MemoryRecord.from_dict(row) for row in read_json(final_path)]
        return {"status": "existing", "memory_count": len(memories), "path": str(final_path)}

    sample_id = sample_dir.name
    window_tasks = [
        {"name": path.name, "_path": str(path)}
        for path in _window_files(sample_dir)
    ]

    def process_window(task: Dict[str, Any]) -> Dict[str, Any]:
        window_path = Path(task["_path"])
        window = read_json(window_path)
        window_output = (
            output_dir
            / "windows"
            / f"{window_path.stem}_memories.json"
        )
        if window_output.exists() and not force:
            try:
                cached_rows = read_json(window_output)
                records = [
                    MemoryRecord.from_dict(row)
                    for row in cached_rows
                    if isinstance(row, dict)
                ]
                return {
                    "status": "existing",
                    "records": records,
                    "trace": {
                        "window": window_path.name,
                        "record_count": len(records),
                        "mode": "checkpoint",
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "elapsed_seconds": 0.0,
                    },
                }
            except Exception:
                pass
        started = time.time()
        if heuristic:
            records = heuristic_extract_window(window, sample_id)
            llm_meta: Dict[str, Any] = {"mode": "heuristic", "prompt_tokens": 0, "completion_tokens": 0}
            raw_payload: Any = {"memories": [record.to_dict() for record in records]}
        else:
            if client is None:
                raise ValueError("LLM client is required unless heuristic=True")
            prompt = render(
                MEMORY_EXTRACTION_TEMPLATE,
                overlap_count=int(window.get("overlap_count") or 0),
                conversation=window.get("conversation") or "",
            )
            raw_payload, result = client.json(
                prompt,
                system=MEMORY_EXTRACTION_SYSTEM,
                max_tokens=max_tokens,
            )
            rows = raw_payload.get("memories", []) if isinstance(raw_payload, dict) else []
            records = []
            for row in rows if isinstance(rows, list) else []:
                if not isinstance(row, dict):
                    continue
                record = _normalise_extracted_memory(row, window=window, sample_id=sample_id)
                if record:
                    records.append(record)
            llm_meta = {
                "mode": "llm",
                "elapsed_seconds": result.elapsed_seconds,
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
            }

        window_output = output_dir / "windows" / (window_path.stem + "_memories.json")
        write_json(window_output, [record.to_dict() for record in records])
        return {
            "status": "ok",
            "records": records,
            "trace": {
                "window": window_path.name,
                "record_count": len(records),
                "elapsed_seconds": time.time() - started,
                **llm_meta,
                "raw_response": raw_payload if heuristic else None,
            },
        }

    window_results, window_stats = run_parallel(
        window_tasks,
        process_window,
        workers=window_workers,
        stage=f"extract-v2-windows-{sample_id}",
        merge_input=False,
        fail_fast=True,
        progress=window_workers != 1,
    )

    all_records: Dict[str, MemoryRecord] = {}
    trace: List[Dict[str, Any]] = []
    for result in window_results:
        for record in result.get("records") or []:
            key = _record_key(record)
            if key in all_records:
                all_records[key] = _merge_duplicate(all_records[key], record)
            else:
                all_records[key] = record
        if result.get("trace"):
            trace.append(result["trace"])

    memories = list(all_records.values())
    memories.sort(key=lambda item: (
        item.provenance.source_ids[0] if item.provenance.source_ids else 10**9,
        item.memory_id,
    ))
    write_json(final_path, [record.to_dict() for record in memories])
    write_json(output_dir / "memory_bank.json", [record.to_dict() for record in memories])
    write_json(trace_path, trace)
    write_json(output_dir / "extraction_parallel.json", window_stats.to_dict())
    return {
        "status": "ok",
        "memory_count": len(memories),
        "path": str(final_path),
        "window_parallel": window_stats.to_dict(),
    }

def extract_run(
    run_root: str,
    *,
    client: Optional[OpenAICompatibleClient],
    force: bool = False,
    heuristic: bool = False,
    max_tokens: int = 5000,
    question_types: Optional[Iterable[str]] = None,
    workers: int = 0,
    fail_fast: bool = False,
    window_workers: int = 1,
) -> Dict[str, Any]:
    root = Path(run_root)
    manifest = read_json(root / "run_manifest.json")
    allowed = {str(value) for value in question_types or []}
    samples = [
        sample for sample in (manifest.get("samples") or [])
        if not allowed or sample.get("question_type") in allowed
    ]
    manifest_path = root / "extraction_manifest_v2.json"
    checkpoint_rows: List[Optional[Dict[str, Any]]] = [None] * len(samples)

    def worker(sample: Dict[str, Any]) -> Dict[str, Any]:
        return extract_sample(
            Path(sample["sample_dir"]),
            client=client,
            force=force,
            heuristic=heuristic,
            max_tokens=max_tokens,
            window_workers=window_workers,
        )

    def on_result(index: int, result: Dict[str, Any], completed: int, total: int) -> None:
        checkpoint_rows[index] = result
        if not should_checkpoint(completed, total):
            return
        write_json(manifest_path, {
            "parallel": {"requested_workers": workers, "completed": completed, "total": total},
            "samples": [row for row in checkpoint_rows if row is not None],
        })

    output, stats = run_parallel(
        samples,
        worker,
        workers=workers,
        stage="extract-v2",
        merge_input=True,
        fail_fast=fail_fast,
        on_result=on_result,
    )
    write_json(manifest_path, {"parallel": stats.to_dict(), "samples": output})
    return {"samples": output, "parallel": stats.to_dict()}
