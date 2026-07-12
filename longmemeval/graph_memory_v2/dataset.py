from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .io_utils import (
    ensure_dir,
    iter_json_inputs,
    normalize_question_type,
    question_type_from_file,
    sample_id,
    write_json,
)


DATE_FORMATS = (
    "%Y/%m/%d (%a) %H:%M",
    "%Y/%m/%d %H:%M",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d",
)


def parse_datetime(value: Any) -> datetime:
    text = str(value or "").strip()
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).replace(tzinfo=None)
    except Exception:
        return datetime(1970, 1, 1)


@dataclass
class SourceTurn:
    source_id: int
    source_uid: str
    session_id: str
    session_index: int
    session_local_user_index: int
    global_user_index: int
    timestamp: str
    weekday: str
    content: str
    assistant_reply: str

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()


def _assistant_after(session: List[Dict[str, Any]], index: int) -> str:
    chunks: List[str] = []
    for cursor in range(index + 1, len(session)):
        role = str(session[cursor].get("role") or "").lower()
        if role == "user":
            break
        if role == "assistant":
            content = str(session[cursor].get("content") or "").strip()
            if content:
                chunks.append(content)
    return "\n".join(chunks)


def flatten_user_turns(item: Dict[str, Any]) -> List[SourceTurn]:
    sessions = item.get("haystack_sessions") or item.get("sessions") or []
    dates = item.get("haystack_dates") or item.get("session_dates") or []
    session_ids = item.get("haystack_session_ids") or item.get("session_ids") or []

    result: List[SourceTurn] = []
    global_index = 0
    for session_index, raw_session in enumerate(sessions):
        if not isinstance(raw_session, list):
            continue
        session_id = (
            str(session_ids[session_index])
            if session_index < len(session_ids)
            else f"session_{session_index:04d}"
        )
        base_time = parse_datetime(dates[session_index] if session_index < len(dates) else "")
        local_user_index = 0
        for message_index, message in enumerate(raw_session):
            if not isinstance(message, dict):
                continue
            if str(message.get("role") or "").lower() != "user":
                continue
            content = str(message.get("content") or "").strip()
            if not content:
                continue
            global_index += 1
            local_user_index += 1
            timestamp = base_time + timedelta(seconds=0.5 * local_user_index)
            source_uid = f"s{session_index:04d}u{local_user_index:04d}"
            result.append(
                SourceTurn(
                    source_id=global_index,
                    source_uid=source_uid,
                    session_id=session_id,
                    session_index=session_index,
                    session_local_user_index=local_user_index,
                    global_user_index=global_index,
                    timestamp=timestamp.isoformat(),
                    weekday=timestamp.strftime("%a"),
                    content=content,
                    assistant_reply=_assistant_after(raw_session, message_index),
                )
            )
    return result


def _truncate(text: str, threshold: int, head: int = 500, middle: int = 200, tail: int = 300) -> str:
    if threshold <= 0 or len(text) <= threshold:
        return text
    midpoint = len(text) // 2
    return (
        text[:head]
        + "\n...[TRUNCATED]...\n"
        + text[max(0, midpoint - middle // 2) : midpoint + middle // 2]
        + "\n...[TRUNCATED]...\n"
        + text[-tail:]
    )


def build_windows(
    turns: List[SourceTurn],
    *,
    window_size: int = 15,
    overlap: int = 3,
    truncate_threshold: int = 8000,
) -> List[Dict[str, Any]]:
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    if overlap < 0 or overlap >= window_size:
        raise ValueError("overlap must satisfy 0 <= overlap < window_size")
    step = window_size - overlap
    windows: List[Dict[str, Any]] = []
    for window_index, start in enumerate(range(0, len(turns), step)):
        chunk = turns[start : start + window_size]
        if not chunk:
            break
        messages = []
        lines = []
        replies: Dict[str, Any] = {}
        for local_source_id, turn in enumerate(chunk, start=1):
            row = turn.to_dict()
            row["window_source_id"] = local_source_id
            row["content"] = _truncate(row["content"], truncate_threshold)
            messages.append(row)
            lines.append(
                f"[{row['timestamp']}, {row['weekday']}] "
                f"{local_source_id}.User: {row['content']}"
            )
            replies[row["source_uid"]] = {
                "source_uid": row["source_uid"],
                "source_id": row["source_id"],
                "window_source_id": local_source_id,
                "session_id": row["session_id"],
                "session_index": row["session_index"],
                "session_local_user_index": row["session_local_user_index"],
                "timestamp": row["timestamp"],
                "assistant_reply": row["assistant_reply"],
            }
        windows.append(
            {
                "window_index": window_index,
                "global_start": start,
                "global_end": start + len(chunk) - 1,
                "overlap_count": overlap if window_index > 0 else 0,
                "messages": messages,
                "conversation": "\n".join(lines),
                "assistant_replies": replies,
            }
        )
        if start + window_size >= len(turns):
            break
    return windows


def prepare_dataset(
    *,
    input_path: str,
    output_root: str,
    run_name: str,
    window_size: int = 15,
    overlap: int = 3,
    truncate_threshold: int = 8000,
    max_items: int = 0,
    question_types: Optional[Iterable[str]] = None,
    force: bool = False,
) -> Dict[str, Any]:
    output_run = ensure_dir(Path(output_root) / run_name)
    allowed = {normalize_question_type(value) for value in question_types or []}
    manifest: List[Dict[str, Any]] = []
    global_index = 0

    for source_file, rows in iter_json_inputs(input_path):
        file_type = normalize_question_type(question_type_from_file(source_file))
        for row in rows:
            qid = str(row.get("question_id") or f"row_{global_index}")
            qid_prefix = qid.split("/", 1)[0] if "/" in qid else ""
            qtype = normalize_question_type(row.get("question_type") or file_type or qid_prefix)
            if qtype == "unknown" and qid_prefix:
                qtype = normalize_question_type(qid_prefix)
            if allowed and qtype not in allowed:
                continue
            if max_items and global_index >= max_items:
                break
            sid = sample_id(global_index, qid)
            sample_dir = output_run / qtype / sid
            if sample_dir.exists() and not force:
                manifest.append({
                    "sample_id": sid,
                    "question_id": qid,
                    "question_type": qtype,
                    "sample_dir": str(sample_dir),
                    "status": "existing",
                })
                global_index += 1
                continue

            ensure_dir(sample_dir / "windows")
            turns = flatten_user_turns(row)
            windows = build_windows(
                turns,
                window_size=window_size,
                overlap=overlap,
                truncate_threshold=truncate_threshold,
            )
            write_json(sample_dir / "input_item.json", row)
            write_json(sample_dir / "source_turns.json", [turn.to_dict() for turn in turns])
            for window in windows:
                window_index = window["window_index"]
                window_path = sample_dir / "windows" / f"window_{window_index:04d}.json"
                write_json(window_path, window)
                (sample_dir / "windows" / f"window_{window_index:04d}.txt").write_text(
                    window["conversation"], encoding="utf-8"
                )
                write_json(
                    sample_dir / "windows" / f"window_{window_index:04d}_assistant_replies.json",
                    window["assistant_replies"],
                )
            write_json(
                sample_dir / "summary.json",
                {
                    "sample_id": sid,
                    "question_id": qid,
                    "question_type": qtype,
                    "turn_count": len(turns),
                    "window_count": len(windows),
                    "window_size": window_size,
                    "overlap": overlap,
                },
            )
            manifest.append({
                "sample_id": sid,
                "question_id": qid,
                "question_type": qtype,
                "sample_dir": str(sample_dir),
                "turn_count": len(turns),
                "window_count": len(windows),
                "status": "prepared",
            })
            global_index += 1
        if max_items and global_index >= max_items:
            break

    write_json(
        output_run / "run_manifest.json",
        {
            "run_name": run_name,
            "input_path": str(input_path),
            "window_size": window_size,
            "overlap": overlap,
            "samples": manifest,
        },
    )
    return {"run_root": str(output_run), "samples": manifest}
