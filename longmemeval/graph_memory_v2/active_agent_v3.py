from __future__ import annotations

import json
import math
import os
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .active_agent import (
    ActiveReconstructionAgent as LegacyActiveReconstructionAgent,
    GraphTools,
    _action_key,
    _memory_view,
)
from .llm_client import OpenAICompatibleClient
from .parallel import run_parallel
from .prompts import render
from .prompts_v3 import ACTIVE_ROUTER_SYSTEM, ACTIVE_ROUTER_TEMPLATE
from .retrieval import StaticRetriever
from .schemas import ParsedQueryV2, QueryHypothesis, RetrievalRecord


ACTION_FAMILIES: Dict[str, str] = {
    "search_text": "semantic",
    "expand_entity": "entity",
    "expand_topic": "topic",
    "expand_time": "time",
    "expand_session": "session",
    "follow_state_chain": "state",
    "expand_relations": "relation",
    "inspect_sources": "source",
}

BRIDGE_TOOLS = {
    "search_text",
    "expand_entity",
    "expand_topic",
    "expand_time",
    "expand_session",
    "expand_relations",
}


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().casefold() not in {"0", "false", "no", "off", ""}


def _env_int(name: str, default: int, *, minimum: int = 0) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError:
        return default
    return max(minimum, parsed)


def _norm(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().casefold())


def _tokens(value: Any) -> Set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", _norm(value))
        if len(token) > 1
    }


def _unique(values: Iterable[str]) -> List[str]:
    result: List[str] = []
    seen: Set[str] = set()
    for value in values:
        text = str(value or "").strip()
        marker = text.casefold()
        if not text or marker in seen:
            continue
        seen.add(marker)
        result.append(text)
    return result


def _hypotheses(parsed: ParsedQueryV2) -> List[QueryHypothesis]:
    return list(parsed.hypotheses) or [parsed.primary()]


def _all_state_keys(parsed: ParsedQueryV2) -> List[str]:
    return _unique(
        state_key
        for hypothesis in _hypotheses(parsed)
        for state_key in hypothesis.state_keys
    )


def _all_entities(parsed: ParsedQueryV2) -> List[str]:
    return _unique(
        entity
        for hypothesis in _hypotheses(parsed)
        for entity in hypothesis.entities
    )


def _all_missing_slots(parsed: ParsedQueryV2) -> List[str]:
    return _unique(
        slot
        for hypothesis in _hypotheses(parsed)
        for slot in hypothesis.missing_slots
    )


def _needs_assistant(parsed: ParsedQueryV2) -> bool:
    return any(hypothesis.need_assistant_context for hypothesis in _hypotheses(parsed))


def _needs_multi_hop(parsed: ParsedQueryV2) -> bool:
    return any(
        hypothesis.need_multi_hop
        or hypothesis.expected_evidence_count > 1
        or bool(hypothesis.missing_slots)
        for hypothesis in _hypotheses(parsed)
    )


def _expected_evidence_count(parsed: ParsedQueryV2) -> int:
    return max(
        [hypothesis.expected_evidence_count for hypothesis in _hypotheses(parsed)]
        or [1]
    )


def _answer_dims(parsed: ParsedQueryV2) -> Set[str]:
    return {
        _norm(hypothesis.answer_dim)
        for hypothesis in _hypotheses(parsed)
        if hypothesis.answer_dim
    }


def _visited_tools(visited: Set[str]) -> Set[str]:
    tools: Set[str] = set()
    for item in visited:
        try:
            payload = json.loads(item)
        except Exception:
            continue
        tool = str(payload.get("tool") or "")
        if tool:
            tools.add(tool)
    return tools


def _has_time(memory_record: RetrievalRecord) -> bool:
    time_dim = memory_record.memory.dimension.time
    return any(
        str(value or "").strip()
        for value in (
            time_dim.event_start,
            time_dim.event_end,
            time_dim.valid_from,
            time_dim.valid_to,
            time_dim.raw,
        )
    )


def _has_assistant_evidence(evidence: Sequence[RetrievalRecord]) -> bool:
    return any(record.memory.assistant_replies for record in evidence)


def _evidence_state_keys(evidence: Sequence[RetrievalRecord]) -> Set[str]:
    return {
        _norm(record.memory.dimension.state_key)
        for record in evidence
        if record.memory.dimension.state_key
    }


def coverage_snapshot(
    parsed: ParsedQueryV2,
    evidence: Sequence[RetrievalRecord],
    visited: Set[str],
    remaining_slots: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Compute generic evidence-completeness constraints.

    The gate intentionally uses query structure rather than benchmark category
    labels or case-specific keywords.
    """
    visited_tools = _visited_tools(visited)
    state_keys = _all_state_keys(parsed)
    state_keys_norm = {_norm(value) for value in state_keys}
    evidence_state_keys = _evidence_state_keys(evidence)
    expected_count = _expected_evidence_count(parsed)
    needs_assistant = _needs_assistant(parsed)
    needs_multi_hop = _needs_multi_hop(parsed)
    slots = _unique(remaining_slots or _all_missing_slots(parsed))
    answer_dims = _answer_dims(parsed)

    has_assistant = _has_assistant_evidence(evidence)
    state_chain_checked = (
        not state_keys
        or "follow_state_chain" in visited_tools
    )
    state_evidence_present = (
        not state_keys
        or bool(state_keys_norm.intersection(evidence_state_keys))
    )
    bridge_checked = (
        not needs_multi_hop
        or bool(visited_tools.intersection(BRIDGE_TOOLS))
    )
    time_evidence_present = (
        "time" not in answer_dims
        or any(_has_time(record) for record in evidence)
    )

    hard_gaps: List[str] = []

    # Assistant-dependent questions must inspect source replies at least once.
    if needs_assistant and "inspect_sources" not in visited_tools:
        hard_gaps.append("assistant_source_not_inspected")

    # Mutable-state questions must traverse the update chain at least once.
    if state_keys and not state_chain_checked:
        hard_gaps.append("state_chain_not_checked")

    # Multi-hop or multi-evidence queries must perform at least one bridge probe.
    if needs_multi_hop and not bridge_checked:
        hard_gaps.append("bridge_search_not_attempted")

    # A time answer cannot finish without any grounded time-bearing memory,
    # unless a time-oriented search has already been attempted.
    if (
        "time" in answer_dims
        and not time_evidence_present
        and "expand_time" not in visited_tools
        and "search_text" not in visited_tools
    ):
        hard_gaps.append("time_evidence_missing")

    # Missing slots are a hard gate only before a bridge probe. This avoids an
    # endless loop when the query parser emits an imperfect slot description.
    if slots and not bridge_checked:
        hard_gaps.append("unresolved_slots_without_probe")

    return {
        "must_continue": bool(hard_gaps),
        "hard_gaps": hard_gaps,
        "remaining_slots": slots,
        "expected_evidence_count": expected_count,
        "current_evidence_count": len(evidence),
        "needs_assistant_context": needs_assistant,
        "has_assistant_context": has_assistant,
        "state_keys": state_keys,
        "state_chain_checked": state_chain_checked,
        "state_evidence_present": state_evidence_present,
        "needs_multi_hop": needs_multi_hop,
        "bridge_checked": bridge_checked,
        "time_evidence_present": time_evidence_present,
        "visited_tools": sorted(visited_tools),
    }


def select_diverse_actions(
    actions: Sequence[Dict[str, Any]],
    visited: Set[str],
    *,
    limit: int = 3,
) -> List[Dict[str, Any]]:
    """Keep unvisited actions while preferring different route families."""
    valid: List[Dict[str, Any]] = []
    seen_keys: Set[str] = set()

    for action in actions:
        if not isinstance(action, dict):
            continue
        tool = str(action.get("tool") or "").strip()
        args = action.get("args") if isinstance(action.get("args"), dict) else {}
        if tool not in ACTION_FAMILIES:
            continue
        key = _action_key(tool, args)
        if key in visited or key in seen_keys:
            continue
        seen_keys.add(key)
        valid.append({"tool": tool, "args": dict(args)})

    result: List[Dict[str, Any]] = []
    used_families: Set[str] = set()

    for action in valid:
        family = ACTION_FAMILIES[action["tool"]]
        if family in used_families:
            continue
        result.append(action)
        used_families.add(family)
        if len(result) >= limit:
            return result

    for action in valid:
        if action in result:
            continue
        result.append(action)
        if len(result) >= limit:
            break

    return result


def _top_memory_ids(
    evidence: Sequence[RetrievalRecord],
    *,
    limit: int = 8,
) -> List[str]:
    return [
        record.memory.memory_id
        for record in evidence[:limit]
        if record.memory.memory_id
    ]


def plan_gap_actions(
    parsed: ParsedQueryV2,
    evidence: Sequence[RetrievalRecord],
    visited: Set[str],
    coverage: Dict[str, Any],
    *,
    limit: int = 3,
) -> List[Dict[str, Any]]:
    """Plan generic, gap-directed actions from query dimensions."""
    candidates: List[Dict[str, Any]] = []
    top_ids = _top_memory_ids(evidence, limit=8)

    if (
        coverage.get("needs_assistant_context")
        and not coverage.get("has_assistant_context")
        and top_ids
    ):
        candidates.append(
            {
                "tool": "inspect_sources",
                "args": {
                    "memory_ids": top_ids,
                    "include_assistant": True,
                },
            }
        )

    for state_key in coverage.get("state_keys") or []:
        candidates.append(
            {
                "tool": "follow_state_chain",
                "args": {
                    "state_key": state_key,
                    "limit": 12,
                },
            }
        )

    for hypothesis in _hypotheses(parsed):
        tc = hypothesis.time_constraint
        if tc.start or tc.end or tc.raw:
            candidates.append(
                {
                    "tool": "expand_time",
                    "args": {
                        "start": tc.start,
                        "end": tc.end,
                        "operator": tc.operator,
                        "limit": 10,
                    },
                }
            )

    # Each hypothesis gets an independent semantic reseed. This lets the agent
    # recover when the highest-confidence parse is wrong.
    for hypothesis in _hypotheses(parsed):
        query = hypothesis.query_anchor or parsed.question
        if query:
            candidates.append(
                {
                    "tool": "search_text",
                    "args": {
                        "query": query,
                        "entities": list(hypothesis.entities),
                        "keywords": list(hypothesis.keywords),
                        "memory_types": list(hypothesis.target_memory_types),
                        "limit": 10,
                    },
                }
            )

    for entity in _all_entities(parsed):
        candidates.append(
            {
                "tool": "expand_entity",
                "args": {
                    "entity": entity,
                    "limit": 10,
                },
            }
        )

    if top_ids:
        candidates.append(
            {
                "tool": "expand_relations",
                "args": {
                    "memory_ids": top_ids[:6],
                    "limit": 12,
                },
            }
        )

    # Session expansion is useful after a relevant seed reveals a session.
    session_ids = _unique(
        record.memory.provenance.session_id
        for record in evidence[:8]
        if record.memory.provenance.session_id
    )
    for session_id in session_ids[:2]:
        candidates.append(
            {
                "tool": "expand_session",
                "args": {
                    "session_id": session_id,
                    "limit": 10,
                },
            }
        )

    return select_diverse_actions(candidates, visited, limit=limit)


def _route_families(record: RetrievalRecord) -> Set[str]:
    families: Set[str] = set()
    for source in record.sources:
        route = str(source.get("route") or "")
        if "state" in route:
            families.add("state")
        elif "assistant" in route or "source" in route:
            families.add("source")
        elif "relation" in route or "graph" in route:
            families.add("relation")
        elif "time" in route:
            families.add("time")
        elif "entity" in route:
            families.add("entity")
        elif "topic" in route:
            families.add("topic")
        elif route:
            families.add("semantic")
    return families


def _query_match_score(
    record: RetrievalRecord,
    parsed: ParsedQueryV2,
) -> float:
    memory_text = _tokens(record.memory.searchable_text())
    if not memory_text:
        return 0.0

    best = 0.0
    for hypothesis in _hypotheses(parsed):
        query_tokens = _tokens(
            " ".join(
                [
                    hypothesis.query_anchor,
                    *hypothesis.entities,
                    *hypothesis.keywords,
                    *hypothesis.locations,
                    *hypothesis.state_keys,
                    *hypothesis.relation_hints,
                ]
            )
        )
        if not query_tokens:
            continue
        overlap = len(query_tokens.intersection(memory_text))
        best = max(best, overlap / max(1, len(query_tokens)))
    return min(1.0, best)


def _structural_score(
    record: RetrievalRecord,
    parsed: ParsedQueryV2,
) -> float:
    score = 0.0
    dim = record.memory.dimension

    score += 0.25 * _query_match_score(record, parsed)
    score += min(0.15, 0.05 * len(_route_families(record)))
    score += 0.10 if record.paths else 0.0
    score += 0.05 * max(0.0, min(1.0, record.memory.confidence))

    if _needs_assistant(parsed) and record.memory.assistant_replies:
        score += 0.25

    state_keys = {_norm(value) for value in _all_state_keys(parsed)}
    if state_keys and _norm(dim.state_key) in state_keys:
        score += 0.20

    if "time" in _answer_dims(parsed) and _has_time(record):
        score += 0.15

    if _needs_multi_hop(parsed) and (
        record.paths or len(_route_families(record)) >= 2
    ):
        score += 0.10

    return min(1.0, score)


def _record_similarity(
    left: RetrievalRecord,
    right: RetrievalRecord,
) -> float:
    left_tokens = _tokens(left.memory.content)
    right_tokens = _tokens(right.memory.content)
    union = left_tokens.union(right_tokens)
    token_similarity = (
        len(left_tokens.intersection(right_tokens)) / len(union)
        if union
        else 0.0
    )

    same_session = (
        bool(left.memory.provenance.session_id)
        and left.memory.provenance.session_id
        == right.memory.provenance.session_id
    )
    same_topic = (
        bool(left.memory.dimension.topic)
        and _norm(left.memory.dimension.topic)
        == _norm(right.memory.dimension.topic)
    )

    return min(
        1.0,
        token_similarity
        + (0.10 if same_session else 0.0)
        + (0.10 if same_topic else 0.0),
    )


def coverage_aware_rerank(
    records: Sequence[RetrievalRecord],
    parsed: ParsedQueryV2,
    *,
    final_k: int,
) -> List[RetrievalRecord]:
    """Blend retrieval score, structural usefulness and redundancy control."""
    candidates = list(records)
    if not candidates:
        return []

    max_base = max(max(0.0, record.score) for record in candidates) or 1.0
    combined: Dict[str, float] = {}

    for record in candidates:
        base_score = max(0.0, record.score) / max_base
        structure = _structural_score(record, parsed)
        combined[record.memory.memory_id] = (
            0.72 * base_score
            + 0.28 * structure
        )

    selected: List[RetrievalRecord] = []
    remaining = list(candidates)

    while remaining and len(selected) < final_k:
        best_record: Optional[RetrievalRecord] = None
        best_value = -math.inf

        for record in remaining:
            relevance = combined[record.memory.memory_id]
            redundancy = max(
                [_record_similarity(record, chosen) for chosen in selected]
                or [0.0]
            )
            value = relevance - 0.12 * redundancy

            if value > best_value:
                best_value = value
                best_record = record

        assert best_record is not None
        best_record.score = combined[best_record.memory.memory_id]
        selected.append(best_record)
        remaining.remove(best_record)

    for rank, record in enumerate(selected, start=1):
        record.rank = rank

    return selected


class ActiveReconstructionAgent:
    """Dimension-guided active reconstruction with generic completion gates.

    Feature flags are environment variables so the same code can run clean
    ablations without introducing benchmark-category branches:

    DIMMEM_V3_COVERAGE_GATE=0|1
    DIMMEM_V3_ALL_HYPOTHESES=0|1
    DIMMEM_V3_DIVERSE_ACTIONS=0|1
    DIMMEM_V3_COVERAGE_RERANK=0|1
    DIMMEM_V3_MAX_TOTAL_TOOL_CALLS=9
    DIMMEM_V3_ROUTER_EVIDENCE_K=18
    """

    def __init__(
        self,
        *,
        store: Any,
        retriever: StaticRetriever,
        client: Optional[OpenAICompatibleClient] = None,
        max_rounds: int = 4,
        final_k: int = 18,
        router_mode: str = "llm",
        tool_workers: int = 0,
    ) -> None:
        self.store = store
        self.retriever = retriever
        self.client = client
        self.legacy_mode = _env_bool(
            "DIMMEM_V3_LEGACY",
            False,
        )
        self._legacy_agent = None
        if self.legacy_mode:
            self._legacy_agent = LegacyActiveReconstructionAgent(
                store=store,
                retriever=retriever,
                client=client,
                max_rounds=max_rounds,
                final_k=final_k,
                router_mode=router_mode,
                tool_workers=tool_workers,
            )
        self.max_rounds = max(0, int(max_rounds))
        self.final_k = max(1, int(final_k))
        self.router_mode = router_mode
        self.tool_workers = int(tool_workers)
        self.tools = GraphTools(store, retriever)

        self.coverage_gate = _env_bool(
            "DIMMEM_V3_COVERAGE_GATE",
            True,
        )
        self.all_hypotheses = _env_bool(
            "DIMMEM_V3_ALL_HYPOTHESES",
            True,
        )
        self.diverse_actions = _env_bool(
            "DIMMEM_V3_DIVERSE_ACTIONS",
            True,
        )
        self.coverage_rerank = _env_bool(
            "DIMMEM_V3_COVERAGE_RERANK",
            True,
        )
        self.max_total_tool_calls = _env_int(
            "DIMMEM_V3_MAX_TOTAL_TOOL_CALLS",
            9,
            minimum=1,
        )
        self.router_evidence_k = _env_int(
            "DIMMEM_V3_ROUTER_EVIDENCE_K",
            18,
            minimum=4,
        )
        self.max_actions_per_round = _env_int(
            "DIMMEM_V3_MAX_ACTIONS_PER_ROUND",
            3,
            minimum=1,
        )

    def _query_payload(self, parsed: ParsedQueryV2) -> Dict[str, Any]:
        if self.all_hypotheses:
            return parsed.to_dict()
        return {
            "question": parsed.question,
            "question_date": parsed.question_date,
            "hypotheses": [parsed.primary().to_dict()],
        }

    def _heuristic_actions(
        self,
        parsed: ParsedQueryV2,
        evidence: Sequence[RetrievalRecord],
        visited: Set[str],
        coverage: Dict[str, Any],
    ) -> Dict[str, Any]:
        actions = plan_gap_actions(
            parsed,
            evidence,
            visited,
            coverage,
            limit=self.max_actions_per_round,
        )
        return {
            "active_hypothesis": 0,
            "mode": "navigate" if actions else "finish",
            "actions": actions,
            "updated_missing_slots": coverage.get("remaining_slots") or [],
            "revised_constraints": {},
            "reason": "generic dimension-guided fallback",
        }

    def _route(
        self,
        parsed: ParsedQueryV2,
        evidence: Sequence[RetrievalRecord],
        visited: Set[str],
        coverage: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        if self.router_mode == "heuristic" or self.client is None:
            return (
                self._heuristic_actions(
                    parsed,
                    evidence,
                    visited,
                    coverage,
                ),
                {"mode": "heuristic"},
            )

        payload, result = self.client.json(
            render(
                ACTIVE_ROUTER_TEMPLATE,
                question=parsed.question,
                question_date=parsed.question_date,
                query_json=self._query_payload(parsed),
                coverage_json=coverage,
                evidence_json=[
                    _memory_view(record, include_assistant=False)
                    for record in evidence[: self.router_evidence_k]
                ],
                visited_json=sorted(visited),
            ),
            system=ACTIVE_ROUTER_SYSTEM,
            max_tokens=3000,
        )
        if not isinstance(payload, dict):
            payload = {}

        return payload, {
            "mode": "llm",
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "elapsed_seconds": result.elapsed_seconds,
        }

    def retrieve(
        self,
        parsed: ParsedQueryV2,
        *,
        route_k: int = 30,
        initial_k: int = 15,
    ) -> Tuple[List[RetrievalRecord], Dict[str, Any]]:
        if self._legacy_agent is not None:
            records, trace = self._legacy_agent.retrieve(
                parsed,
                route_k=route_k,
                initial_k=initial_k,
            )
            trace["policy"] = {"name": "legacy"}
            return records, trace

        initial = self.retriever.retrieve(
            parsed,
            route_k=route_k,
            final_k=initial_k,
        )
        evidence: Dict[str, RetrievalRecord] = {
            record.memory.memory_id: record
            for record in initial
        }
        visited: Set[str] = set()
        remaining_slots = _all_missing_slots(parsed)
        total_tool_calls = 0

        trace: Dict[str, Any] = {
            "initial": [_memory_view(record) for record in initial],
            "rounds": [],
            "router_mode": self.router_mode,
            "policy": {
                "name": "active_v3",
                "coverage_gate": self.coverage_gate,
                "all_hypotheses": self.all_hypotheses,
                "diverse_actions": self.diverse_actions,
                "coverage_rerank": self.coverage_rerank,
                "max_rounds": self.max_rounds,
                "max_total_tool_calls": self.max_total_tool_calls,
                "router_evidence_k": self.router_evidence_k,
                "max_actions_per_round": self.max_actions_per_round,
            },
        }

        for round_index in range(self.max_rounds):
            ordered = sorted(
                evidence.values(),
                key=lambda record: (
                    -record.score,
                    record.memory.memory_id,
                ),
            )
            coverage = coverage_snapshot(
                parsed,
                ordered,
                visited,
                remaining_slots,
            )
            decision, meta = self._route(
                parsed,
                ordered,
                visited,
                coverage,
            )

            mode = str(decision.get("mode") or "navigate").lower()
            actions = (
                decision.get("actions")
                if isinstance(decision.get("actions"), list)
                else []
            )
            updated_slots = decision.get("updated_missing_slots")
            if isinstance(updated_slots, list):
                remaining_slots = _unique(
                    str(value)
                    for value in updated_slots
                )

            round_trace: Dict[str, Any] = {
                "round": round_index + 1,
                "coverage_before": coverage,
                "decision": decision,
                "meta": meta,
                "finish_overridden": False,
                "tool_results": [],
            }

            if (
                self.coverage_gate
                and mode == "finish"
                and coverage.get("must_continue")
            ):
                actions = plan_gap_actions(
                    parsed,
                    ordered,
                    visited,
                    coverage,
                    limit=self.max_actions_per_round,
                )
                if actions:
                    mode = "navigate"
                    round_trace["finish_overridden"] = True
                    round_trace["override_reason"] = (
                        coverage.get("hard_gaps") or []
                    )

            if (
                self.coverage_gate
                and (not actions)
                and coverage.get("must_continue")
            ):
                actions = plan_gap_actions(
                    parsed,
                    ordered,
                    visited,
                    coverage,
                    limit=self.max_actions_per_round,
                )
                if actions:
                    mode = "navigate"
                    round_trace["finish_overridden"] = True
                    round_trace["override_reason"] = (
                        coverage.get("hard_gaps") or []
                    )

            if mode == "finish" or not actions:
                trace["rounds"].append(round_trace)
                break

            if self.diverse_actions:
                runnable_actions = select_diverse_actions(
                    actions,
                    visited,
                    limit=self.max_actions_per_round,
                )
            else:
                runnable_actions = []
                for action in actions:
                    if not isinstance(action, dict):
                        continue
                    tool = str(action.get("tool") or "")
                    args = (
                        action.get("args")
                        if isinstance(action.get("args"), dict)
                        else {}
                    )
                    if tool not in ACTION_FAMILIES:
                        continue
                    if _action_key(tool, args) in visited:
                        continue
                    runnable_actions.append(
                        {"tool": tool, "args": dict(args)}
                    )
                    if (
                        len(runnable_actions)
                        >= self.max_actions_per_round
                    ):
                        break

            remaining_budget = (
                self.max_total_tool_calls
                - total_tool_calls
            )
            runnable_actions = runnable_actions[:remaining_budget]

            if not runnable_actions:
                round_trace["stopped_reason"] = (
                    "no_unvisited_action_or_tool_budget_exhausted"
                )
                trace["rounds"].append(round_trace)
                break

            for action in runnable_actions:
                visited.add(
                    _action_key(
                        str(action["tool"]),
                        dict(action["args"]),
                    )
                )

            def tool_worker(action: Dict[str, Any]) -> Dict[str, Any]:
                tool = str(action["tool"])
                args = dict(action["args"])
                try:
                    rows = self.tools.execute(tool, args)
                    return {
                        "status": "ok",
                        "tool": tool,
                        "args": args,
                        "rows": rows,
                    }
                except Exception as exc:
                    return {
                        "status": "failed",
                        "tool": tool,
                        "args": args,
                        "error": repr(exc),
                        "rows": [],
                    }

            tool_results, tool_stats = run_parallel(
                runnable_actions,
                tool_worker,
                workers=self.tool_workers,
                stage=f"active-v3-tools-r{round_index + 1}",
                progress=False,
                fail_fast=False,
            )
            total_tool_calls += len(runnable_actions)
            round_trace["tool_parallel"] = tool_stats.to_dict()

            new_memory_count = 0
            for result in tool_results:
                tool = str(result.get("tool") or "")
                args = (
                    result.get("args")
                    if isinstance(result.get("args"), dict)
                    else {}
                )
                rows = (
                    result.get("rows")
                    if isinstance(result.get("rows"), list)
                    else []
                )

                if result.get("status") == "failed":
                    round_trace["tool_results"].append(
                        {
                            "tool": tool,
                            "args": args,
                            "error": (
                                result.get("error")
                                or "tool execution failed"
                            ),
                        }
                    )
                    continue

                for record in rows:
                    memory_id = record.memory.memory_id
                    if memory_id in evidence:
                        existing = evidence[memory_id]
                        existing.score += record.score
                        existing.sources.extend(record.sources)
                        existing.paths.extend(record.paths)
                    else:
                        evidence[memory_id] = record
                        new_memory_count += 1

                round_trace["tool_results"].append(
                    {
                        "tool": tool,
                        "args": args,
                        "count": len(rows),
                        "memory_ids": [
                            record.memory.memory_id
                            for record in rows
                        ],
                    }
                )

            round_trace["new_memory_count"] = new_memory_count
            round_trace["total_tool_calls"] = total_tool_calls
            round_trace["coverage_after"] = coverage_snapshot(
                parsed,
                sorted(
                    evidence.values(),
                    key=lambda record: (
                        -record.score,
                        record.memory.memory_id,
                    ),
                ),
                visited,
                remaining_slots,
            )
            trace["rounds"].append(round_trace)

            if total_tool_calls >= self.max_total_tool_calls:
                trace["stopped_reason"] = "tool_budget_exhausted"
                break

        candidates = list(evidence.values())
        if self.coverage_rerank:
            final = coverage_aware_rerank(
                candidates,
                parsed,
                final_k=self.final_k,
            )
        else:
            final = sorted(
                candidates,
                key=lambda record: (
                    -record.score,
                    record.memory.memory_id,
                ),
            )[: self.final_k]
            for rank, record in enumerate(final, start=1):
                record.rank = rank

        trace["final"] = [
            _memory_view(
                record,
                include_assistant=_needs_assistant(parsed),
            )
            for record in final
        ]
        trace["visited_actions"] = sorted(visited)
        trace["total_tool_calls"] = total_tool_calls
        trace["final_coverage"] = coverage_snapshot(
            parsed,
            final,
            visited,
            remaining_slots,
        )

        return final, trace


__all__ = [
    "ActiveReconstructionAgent",
    "ACTION_FAMILIES",
    "coverage_snapshot",
    "select_diverse_actions",
    "plan_gap_actions",
    "coverage_aware_rerank",
]
