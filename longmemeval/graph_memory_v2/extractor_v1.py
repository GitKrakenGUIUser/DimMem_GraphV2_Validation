from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .extractor import _merge_duplicate, _record_key, _window_files
from .io_utils import ensure_dir, normalize_text, read_json, stable_hash, write_json
from .llm_client import OpenAICompatibleClient
from .parallel import run_parallel, should_checkpoint
from .prompts import DIMMEM_V1_EXTRACTION_SYSTEM, DIMMEM_V1_EXTRACTION_TEMPLATE, render
from .schemas import EnhancedDimension, MemoryRecord, Provenance


def _normalise_v1(raw: Dict[str, Any], window: Dict[str, Any], sample_id: str) -> Optional[MemoryRecord]:
    content = str(raw.get("content") or "").strip()
    if not content:
        return None
    source_values = raw.get("source_ids") or []
    if isinstance(source_values, (int, str)):
        source_values = [source_values]
    local_ids: List[int] = []
    for value in source_values:
        try:
            local_ids.append(int(value))
        except (TypeError, ValueError):
            pass
    overlap_count = int(window.get("overlap_count") or 0)
    by_local = {int(row.get("window_source_id") or 0): row for row in window.get("messages") or []}
    local_ids = sorted(set(value for value in local_ids if value > overlap_count and value in by_local))
    if not local_ids:
        return None
    rows = [by_local[value] for value in local_ids]
    memory_type = str(raw.get("memory_type") or "").lower()
    if memory_type not in {"fact", "episodic", "profile"}:
        memory_type = "fact"
    dimension = EnhancedDimension.from_dict({
        "memory_type": memory_type,
        "time": str(raw.get("time") or ""),
        "location": raw.get("location") or "",
        "reason": raw.get("reason") or "",
        "purpose": raw.get("purpose") or "",
        "keywords": raw.get("keywords") or [],
    })
    source_uids = [str(row.get("source_uid") or "") for row in rows if row.get("source_uid")]
    fingerprint = json.dumps({
        "sample": sample_id,
        "content": normalize_text(content),
        "sources": source_uids,
        "schema": "v1",
    }, sort_keys=True, ensure_ascii=False)
    replies = [str(row.get("assistant_reply") or "").strip() for row in rows if str(row.get("assistant_reply") or "").strip()]
    return MemoryRecord(
        memory_id="v1_" + stable_hash(fingerprint, 20),
        content=content,
        dimension=dimension,
        provenance=Provenance(
            session_id=str(rows[0].get("session_id") or "") if len({row.get("session_id") for row in rows}) == 1 else "",
            session_index=int(rows[0].get("session_index", -1)) if len({row.get("session_index") for row in rows}) == 1 else -1,
            source_ids=[int(row.get("source_id") or 0) for row in rows],
            source_uids=source_uids,
            source_times=[str(row.get("timestamp") or "") for row in rows],
            window_index=int(window.get("window_index", -1)),
        ),
        confidence=max(0.0, min(1.0, float(raw.get("confidence") or 0.0))),
        evidence_span=str(raw.get("evidence_span") or "").strip(),
        assistant_replies=list(dict.fromkeys(replies)),
    )


def heuristic_v1(window: Dict[str, Any], sample_id: str) -> List[MemoryRecord]:
    records: List[MemoryRecord] = []
    overlap_count = int(window.get("overlap_count") or 0)
    for row in window.get("messages") or []:
        local_id = int(row.get("window_source_id") or 0)
        if local_id <= overlap_count:
            continue
        content = str(row.get("content") or "").strip()
        if not content:
            continue
        raw = {
            "source_ids": [local_id],
            "content": content,
            "memory_type": "episodic" if row.get("timestamp") else "fact",
            "time": str(row.get("timestamp") or ""),
            "location": "",
            "reason": "",
            "purpose": "",
            "keywords": content.split()[:8],
            "confidence": 0.2,
            "evidence_span": content[:200],
        }
        memory = _normalise_v1(raw, window, sample_id)
        if memory:
            records.append(memory)
    return records


def extract_sample_v1(
    sample_dir: Path,
    *,
    client: Optional[OpenAICompatibleClient],
    force: bool = False,
    heuristic: bool = False,
    max_tokens: int = 4200,
    window_workers: int = 1,
) -> Dict[str, Any]:
    output_dir = ensure_dir(sample_dir / "memory_v1")
    final_path = output_dir / "all_memories.json"
    if final_path.exists() and not force:
        return {"status": "existing", "memory_count": len(read_json(final_path)), "path": str(final_path)}

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
        if heuristic:
            records = heuristic_v1(window, sample_dir.name)
            meta = {"mode": "heuristic", "prompt_tokens": 0, "completion_tokens": 0}
        else:
            if client is None:
                raise ValueError("LLM client required unless heuristic=True")
            payload, result = client.json(
                render(
                    DIMMEM_V1_EXTRACTION_TEMPLATE,
                    overlap_count=int(window.get("overlap_count") or 0),
                    conversation=window.get("conversation") or "",
                ),
                system=DIMMEM_V1_EXTRACTION_SYSTEM,
                max_tokens=max_tokens,
            )
            raw_rows = payload.get("memories") if isinstance(payload, dict) else []
            records = []
            for raw in raw_rows if isinstance(raw_rows, list) else []:
                if isinstance(raw, dict):
                    record = _normalise_v1(raw, window, sample_dir.name)
                    if record:
                        records.append(record)
            meta = {
                "mode": "llm",
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
                "elapsed_seconds": result.elapsed_seconds,
            }
        write_json(
            output_dir / "windows" / f"{window_path.stem}_memories.json",
            [record.to_dict() for record in records],
        )
        return {
            "status": "ok",
            "records": records,
            "trace": {"window": window_path.name, "record_count": len(records), **meta},
        }

    window_results, window_stats = run_parallel(
        window_tasks,
        process_window,
        workers=window_workers,
        stage=f"extract-v1-windows-{sample_dir.name}",
        merge_input=False,
        fail_fast=True,
        progress=window_workers != 1,
    )

    records_by_key: Dict[str, MemoryRecord] = {}
    trace: List[Dict[str, Any]] = []
    for result in window_results:
        for record in result.get("records") or []:
            key = _record_key(record)
            if key in records_by_key:
                records_by_key[key] = _merge_duplicate(records_by_key[key], record)
            else:
                records_by_key[key] = record
        if result.get("trace"):
            trace.append(result["trace"])

    records = list(records_by_key.values())
    records.sort(key=lambda item: (
        item.provenance.source_ids[0] if item.provenance.source_ids else 10**9,
        item.memory_id,
    ))
    write_json(final_path, [record.to_dict() for record in records])
    write_json(output_dir / "extraction_trace.json", trace)
    write_json(output_dir / "extraction_parallel.json", window_stats.to_dict())
    return {
        "status": "ok",
        "memory_count": len(records),
        "path": str(final_path),
        "window_parallel": window_stats.to_dict(),
    }

def extract_run_v1(
    run_root: str,
    *,
    client: Optional[OpenAICompatibleClient],
    force: bool = False,
    heuristic: bool = False,
    max_tokens: int = 4200,
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
    manifest_path = root / "extraction_manifest_v1.json"
    checkpoint_rows: List[Optional[Dict[str, Any]]] = [None] * len(samples)

    def worker(sample: Dict[str, Any]) -> Dict[str, Any]:
        return extract_sample_v1(
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

    results, stats = run_parallel(
        samples,
        worker,
        workers=workers,
        stage="extract-v1",
        merge_input=True,
        fail_fast=fail_fast,
        on_result=on_result,
    )
    write_json(manifest_path, {"parallel": stats.to_dict(), "samples": results})
    return {"samples": results, "parallel": stats.to_dict()}
