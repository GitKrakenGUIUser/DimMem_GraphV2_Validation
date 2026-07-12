from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .io_utils import read_json, write_json
from .llm_client import OpenAICompatibleClient
from .parallel import run_parallel, should_checkpoint
from .prompts import QUERY_PARSE_SYSTEM, QUERY_PARSE_TEMPLATE, render
from .schemas import ParsedQueryV2, QueryHypothesis


def _question_fields(item: Dict[str, Any]) -> tuple[str, str]:
    question = str(item.get("question") or item.get("query") or "").strip()
    question_date = str(
        item.get("question_date")
        or item.get("query_date")
        or item.get("question_timestamp")
        or ""
    ).strip()
    return question, question_date


def heuristic_parse(question: str, question_date: str = "") -> ParsedQueryV2:
    lower = question.casefold()
    answer_dim = ""
    if lower.startswith("when") or " what date" in lower or "what time" in lower:
        answer_dim = "time"
    elif lower.startswith("why") or " reason" in lower:
        answer_dim = "reason"
    elif lower.startswith("where"):
        answer_dim = "location"
    need_assistant = any(phrase in lower for phrase in (
        "did you say", "did you recommend", "assistant", "your response", "you suggest",
    ))
    hypothesis = QueryHypothesis(
        query_anchor=question,
        keywords=[token.strip("?.,") for token in question.split() if len(token.strip("?.,")) > 2][:12],
        answer_dim="assistant_reply" if need_assistant else answer_dim,
        need_assistant_context=need_assistant,
        need_multi_hop=any(token in lower for token in ("when", "while", "after", "before", "second", "same time")),
        expected_evidence_count=2 if any(token in lower for token in ("when", "while", "after", "before")) else 1,
        confidence=0.2,
    )
    return ParsedQueryV2(question=question, question_date=question_date, hypotheses=[hypothesis])


def parse_sample(
    sample_dir: Path,
    *,
    client: Optional[OpenAICompatibleClient],
    heuristic: bool = False,
    force: bool = False,
) -> Dict[str, Any]:
    output_dir = sample_dir / "query_v2"
    output_path = output_dir / "parsed_query.json"
    if output_path.exists() and not force:
        return {"status": "existing", "path": str(output_path)}
    item = read_json(sample_dir / "input_item.json")
    question, question_date = _question_fields(item)
    if not question:
        raise ValueError(f"missing question: {sample_dir}")
    if heuristic:
        parsed = heuristic_parse(question, question_date)
        raw = parsed.to_dict()
        meta = {"mode": "heuristic", "prompt_tokens": 0, "completion_tokens": 0}
    else:
        if client is None:
            raise ValueError("LLM client required unless heuristic=True")
        payload, result = client.json(
            render(QUERY_PARSE_TEMPLATE, question=question, question_date=question_date),
            system=QUERY_PARSE_SYSTEM,
            max_tokens=2500,
        )
        parsed = ParsedQueryV2.from_dict(payload, question=question, question_date=question_date)
        raw = payload
        meta = {
            "mode": "llm",
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "elapsed_seconds": result.elapsed_seconds,
        }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_path, parsed.to_dict())
    write_json(output_dir / "raw_response.json", raw)
    write_json(output_dir / "parse_meta.json", meta)
    return {"status": "ok", "path": str(output_path), **meta}


def parse_run(
    run_root: str,
    *,
    client: Optional[OpenAICompatibleClient],
    heuristic: bool = False,
    force: bool = False,
    question_types: Optional[Iterable[str]] = None,
    workers: int = 0,
    fail_fast: bool = False,
) -> Dict[str, Any]:
    root = Path(run_root)
    manifest = read_json(root / "run_manifest.json")
    allowed = {str(value) for value in question_types or []}
    samples = [
        sample for sample in (manifest.get("samples") or [])
        if not allowed or sample.get("question_type") in allowed
    ]
    manifest_path = root / "query_manifest_v2.json"
    checkpoint_rows: List[Optional[Dict[str, Any]]] = [None] * len(samples)

    def worker(sample: Dict[str, Any]) -> Dict[str, Any]:
        return parse_sample(
            Path(sample["sample_dir"]),
            client=client,
            heuristic=heuristic,
            force=force,
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
        stage="parse-query",
        merge_input=True,
        fail_fast=fail_fast,
        on_result=on_result,
    )
    write_json(manifest_path, {"parallel": stats.to_dict(), "samples": results})
    return {"samples": results, "parallel": stats.to_dict()}
