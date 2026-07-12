from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Sequence, Tuple


def ensure_dir(path: Path | str) -> Path:
    value = Path(path)
    value.mkdir(parents=True, exist_ok=True)
    return value


def read_json(path: Path | str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path | str, payload: Any, *, indent: int = 2) -> None:
    target = Path(path)
    ensure_dir(target.parent)
    text = json.dumps(payload, ensure_ascii=False, indent=indent)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", delete=False, dir=str(target.parent), prefix=target.name + ".tmp."
    ) as handle:
        handle.write(text)
        temp_name = handle.name
    os.replace(temp_name, target)


def append_jsonl(path: Path | str, payload: Dict[str, Any]) -> None:
    target = Path(path)
    ensure_dir(target.parent)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def read_jsonl(path: Path | str) -> List[Dict[str, Any]]:
    result: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            result.append(json.loads(line))
    return result


def iter_json_inputs(input_path: Path | str) -> Iterator[Tuple[Path, List[Dict[str, Any]]]]:
    path = Path(input_path)
    if path.is_file():
        payload = read_json(path)
        records = payload if isinstance(payload, list) else payload.get("data", [])
        if not isinstance(records, list):
            raise ValueError(f"unsupported JSON dataset structure: {path}")
        yield path, [row for row in records if isinstance(row, dict)]
        return

    if not path.is_dir():
        raise FileNotFoundError(path)

    candidates = sorted(path.glob("*.json"))
    if not candidates:
        raise FileNotFoundError(f"no JSON files under {path}")
    for candidate in candidates:
        payload = read_json(candidate)
        records = payload if isinstance(payload, list) else payload.get("data", [])
        if not isinstance(records, list):
            continue
        yield candidate, [row for row in records if isinstance(row, dict)]


def stable_hash(text: str, length: int = 16) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:length]


def slugify(text: str, max_len: int = 96) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", str(text or "").strip())
    value = re.sub(r"_+", "_", value).strip("_")
    return (value or "item")[:max_len]


def sample_id(index: int, question_id: str) -> str:
    return f"{index:04d}_{slugify(question_id)}"


def question_type_from_file(path: Path) -> str:
    match = re.search(r"longmemeval_s_cleaned__(.+?)\.json$", path.name)
    return match.group(1) if match else ""


def normalize_question_type(value: Any) -> str:
    text = str(value or "").strip().lower().replace("_", "-")
    aliases = {
        "single-session-user": "single-session-user",
        "single-session-assistant": "single-session-assistant",
        "single-session-preference": "single-session-preference",
        "multi-session": "multi-session",
        "temporal-reasoning": "temporal-reasoning",
        "knowledge-update": "knowledge-update",
        "temporal": "temporal-reasoning",
        "preference": "single-session-preference",
    }
    return aliases.get(text, text or "unknown")


def tokenize(text: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9][A-Za-z0-9_'-]*|[\u4e00-\u9fff]", str(text or "").casefold())


def normalize_text(text: str) -> str:
    return " ".join(tokenize(text))


def batched(values: Sequence[Any], size: int) -> Iterator[Sequence[Any]]:
    for start in range(0, len(values), max(1, size)):
        yield values[start : start + max(1, size)]


def dedupe_dicts(rows: Iterable[Dict[str, Any]], key_fn) -> List[Dict[str, Any]]:
    result: List[Dict[str, Any]] = []
    seen = set()
    for row in rows:
        key = key_fn(row)
        if key in seen:
            continue
        seen.add(key)
        result.append(row)
    return result
