from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime
from typing import Any, DefaultDict, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .graph_store import GraphStore, canonical
from .io_utils import normalize_text
from .llm_client import OpenAICompatibleClient
from .prompts import ACTIVE_ROUTER_SYSTEM, ACTIVE_ROUTER_TEMPLATE, render
from .retrieval import StaticRetriever
from .schemas import MemoryRecord, ParsedQueryV2, RetrievalRecord


UPDATE_EDGES = {"SUPERSEDES", "SUPPORTS", "CONTRADICTS"}
RELATION_EDGES = {
    "SUPERSEDES", "SUPPORTS", "CONTRADICTS", "SAME_EVENT", "NEXT_IN_SESSION",
    "BEFORE", "ABOUT_ENTITY", "HAS_TOPIC", "AT_LOCATION", "ASSERTS_STATE",
    "IN_SESSION", "RELATION_SUBJECT", "RELATION_OBJECT", "RELATION",
}


def _action_key(tool: str, args: Dict[str, Any]) -> str:
    return json.dumps({"tool": tool, "args": args}, ensure_ascii=False, sort_keys=True)


def _memory_view(record: RetrievalRecord, *, include_assistant: bool = False) -> Dict[str, Any]:
    memory = record.memory
    payload = {
        "memory_id": memory.memory_id,
        "content": memory.content,
        "dimension": memory.dimension.to_dict(),
        "provenance": memory.provenance.to_dict(),
        "score": record.score,
        "retrieval_sources": record.sources,
        "graph_paths": record.paths,
    }
    if include_assistant and memory.assistant_replies:
        payload["assistant_replies"] = memory.assistant_replies
    return payload


class GraphTools:
    def __init__(self, store: GraphStore, retriever: StaticRetriever) -> None:
        self.store = store
        self.retriever = retriever

    def _from_ids(
        self,
        memory_ids: Sequence[str],
        *,
        source: Dict[str, Any],
        limit: int,
        path: Optional[List[str]] = None,
        score: float = 0.02,
    ) -> List[RetrievalRecord]:
        rows: List[RetrievalRecord] = []
        for rank, memory_id in enumerate(dict.fromkeys(memory_ids), start=1):
            if memory_id not in self.store.memories:
                continue
            rows.append(RetrievalRecord(
                memory=self.store.memories[memory_id],
                score=score / rank,
                rank=rank,
                sources=[dict(source)],
                paths=[list(path)] if path else [],
            ))
            if len(rows) >= limit:
                break
        return rows

    def search_text(self, **args: Any) -> List[RetrievalRecord]:
        return self.retriever.search_text(
            str(args.get("query") or ""),
            entities=args.get("entities") or [],
            keywords=args.get("keywords") or [],
            memory_types=args.get("memory_types") or [],
            limit=int(args.get("limit") or 8),
        )

    def expand_entity(self, **args: Any) -> List[RetrievalRecord]:
        entity = str(args.get("entity") or "")
        limit = int(args.get("limit") or 8)
        return self._from_ids(
            self.store.lookup("entity", entity, limit * 2),
            source={"route": "active_entity", "entity": entity},
            limit=limit,
            path=["entity", entity],
        )

    def expand_topic(self, **args: Any) -> List[RetrievalRecord]:
        topic = str(args.get("topic") or "")
        limit = int(args.get("limit") or 8)
        return self._from_ids(
            self.store.lookup("topic", topic, limit * 2),
            source={"route": "active_topic", "topic": topic},
            limit=limit,
            path=["topic", topic],
        )

    def expand_session(self, **args: Any) -> List[RetrievalRecord]:
        session_id = str(args.get("session_id") or "")
        limit = int(args.get("limit") or 8)
        ids = self.store.lookup("session", session_id, limit * 3)
        ids.sort(key=lambda memory_id: (
            self.store.memories[memory_id].provenance.source_ids[0]
            if self.store.memories[memory_id].provenance.source_ids else 10**9
        ))
        return self._from_ids(
            ids,
            source={"route": "active_session", "session_id": session_id},
            limit=limit,
            path=["session", session_id],
        )

    def expand_time(self, **args: Any) -> List[RetrievalRecord]:
        start = str(args.get("start") or "")
        end = str(args.get("end") or "")
        operator = str(args.get("operator") or "")
        limit = int(args.get("limit") or 8)
        ids: Set[str] = set()
        if start:
            ids.update(self.store.lookup("date", start[:10], limit * 4))
        if end:
            ids.update(self.store.lookup("date", end[:10], limit * 4))
        # Range fallback scans dates only, not all text.
        if (operator in {"before", "after", "between", "around"}) and start:
            from .dataset import parse_datetime
            left = parse_datetime(start)
            right = parse_datetime(end) if end else left
            for date_key, memory_ids in self.store.indexes["date"].items():
                current = parse_datetime(date_key)
                match = (
                    (operator == "before" and current < left)
                    or (operator == "after" and current > left)
                    or (operator == "between" and left <= current <= right)
                    or (operator == "around" and abs((current - left).days) <= 7)
                )
                if match:
                    ids.update(memory_ids)
                if len(ids) >= limit * 5:
                    break
        return self._from_ids(
            sorted(ids),
            source={"route": "active_time", "start": start, "end": end, "operator": operator},
            limit=limit,
            path=["time", operator, start, end],
        )

    def follow_state_chain(self, **args: Any) -> List[RetrievalRecord]:
        state_key = str(args.get("state_key") or "")
        memory_id = str(args.get("memory_id") or "")
        limit = int(args.get("limit") or 10)
        seeds: List[str] = []
        if state_key:
            seeds.extend(self.store.lookup("state_key", state_key, limit * 3))
        if memory_id and memory_id in self.store.memories:
            seeds.append(memory_id)
        related = self.store.related_memory_ids(seeds, edge_types=UPDATE_EDGES, limit=limit * 2)
        ids = list(dict.fromkeys(seeds + [row[0] for row in related]))
        path_map = {row[0]: row[2] for row in related}
        rows = self._from_ids(
            ids,
            source={"route": "active_state_chain", "state_key": state_key, "seed": memory_id},
            limit=limit,
            path=["state_chain", state_key or memory_id],
            score=0.035,
        )
        for row in rows:
            if row.memory.memory_id in path_map:
                row.paths = [path_map[row.memory.memory_id]]
        return rows

    def expand_relations(self, **args: Any) -> List[RetrievalRecord]:
        memory_ids = [str(value) for value in args.get("memory_ids") or []]
        edge_types = {str(value) for value in args.get("edge_types") or []} or RELATION_EDGES
        limit = int(args.get("limit") or 10)
        related = self.store.related_memory_ids(memory_ids, edge_types=edge_types, limit=limit)
        rows: List[RetrievalRecord] = []
        for rank, (memory_id, edge, path) in enumerate(related, start=1):
            rows.append(RetrievalRecord(
                memory=self.store.memories[memory_id],
                score=0.03 * max(0.3, edge.weight) / rank,
                rank=rank,
                sources=[{"route": "active_relation", "edge_type": edge.edge_type}],
                paths=[path],
            ))
        return rows

    def inspect_sources(self, **args: Any) -> List[RetrievalRecord]:
        memory_ids = [str(value) for value in args.get("memory_ids") or []]
        include_assistant = bool(args.get("include_assistant", False))
        rows = self._from_ids(
            memory_ids,
            source={"route": "source_inspection", "include_assistant": include_assistant},
            limit=max(1, len(memory_ids)),
            score=0.04,
        )
        if include_assistant:
            for row in rows:
                row.sources.append({"assistant_replies": row.memory.assistant_replies})
        return rows

    def execute(self, tool: str, args: Dict[str, Any]) -> List[RetrievalRecord]:
        handler = getattr(self, tool, None)
        if tool == "finish":
            return []
        if handler is None or tool.startswith("_"):
            raise ValueError(f"unsupported tool: {tool}")
        return handler(**args)


class ActiveReconstructionAgent:
    def __init__(
        self,
        *,
        store: GraphStore,
        retriever: StaticRetriever,
        client: Optional[OpenAICompatibleClient] = None,
        max_rounds: int = 3,
        final_k: int = 15,
        router_mode: str = "llm",
    ) -> None:
        self.store = store
        self.retriever = retriever
        self.client = client
        self.max_rounds = max(0, int(max_rounds))
        self.final_k = max(1, int(final_k))
        self.router_mode = router_mode
        self.tools = GraphTools(store, retriever)

    def _heuristic_actions(
        self,
        parsed: ParsedQueryV2,
        evidence: Sequence[RetrievalRecord],
        visited: Set[str],
        round_index: int,
    ) -> Dict[str, Any]:
        hypothesis = parsed.primary()
        candidates: List[Dict[str, Any]] = []
        for entity in hypothesis.entities:
            candidates.append({"tool": "expand_entity", "args": {"entity": entity, "limit": 8}})
        tc = hypothesis.time_constraint
        if tc.start or tc.end:
            candidates.append({"tool": "expand_time", "args": {"start": tc.start, "end": tc.end, "operator": tc.operator, "limit": 8}})
        for state_key in hypothesis.state_keys:
            candidates.append({"tool": "follow_state_chain", "args": {"state_key": state_key, "limit": 10}})
        if evidence:
            top_ids = [row.memory.memory_id for row in evidence[:4]]
            candidates.append({"tool": "expand_relations", "args": {"memory_ids": top_ids, "limit": 10}})
            if hypothesis.need_assistant_context:
                candidates.append({"tool": "inspect_sources", "args": {"memory_ids": top_ids, "include_assistant": True}})
        actions = [action for action in candidates if _action_key(action["tool"], action["args"]) not in visited][:3]
        return {
            "mode": "navigate" if actions and round_index < self.max_rounds else "finish",
            "actions": actions,
            "updated_missing_slots": hypothesis.missing_slots,
            "reason": "deterministic graph fallback",
        }

    def _route(
        self,
        parsed: ParsedQueryV2,
        evidence: Sequence[RetrievalRecord],
        visited: Set[str],
        round_index: int,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        if self.router_mode == "heuristic" or self.client is None:
            return self._heuristic_actions(parsed, evidence, visited, round_index), {"mode": "heuristic"}
        payload, result = self.client.json(
            render(
                ACTIVE_ROUTER_TEMPLATE,
                question=parsed.question,
                question_date=parsed.question_date,
                query_json=parsed.primary().to_dict(),
                evidence_json=[_memory_view(row, include_assistant=False) for row in evidence[:12]],
                visited_json=sorted(visited),
            ),
            system=ACTIVE_ROUTER_SYSTEM,
            max_tokens=2200,
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
        route_k: int = 20,
        initial_k: int = 12,
    ) -> Tuple[List[RetrievalRecord], Dict[str, Any]]:
        initial = self.retriever.retrieve(parsed, route_k=route_k, final_k=initial_k)
        evidence: Dict[str, RetrievalRecord] = {row.memory.memory_id: row for row in initial}
        visited: Set[str] = set()
        trace: Dict[str, Any] = {
            "initial": [_memory_view(row) for row in initial],
            "rounds": [],
            "router_mode": self.router_mode,
        }

        for round_index in range(self.max_rounds):
            ordered = sorted(evidence.values(), key=lambda row: (-row.score, row.memory.memory_id))
            decision, meta = self._route(parsed, ordered, visited, round_index)
            mode = str(decision.get("mode") or "navigate").lower()
            actions = decision.get("actions") or []
            round_trace: Dict[str, Any] = {
                "round": round_index + 1,
                "decision": decision,
                "meta": meta,
                "tool_results": [],
            }
            if mode == "finish" or not isinstance(actions, list) or not actions:
                trace["rounds"].append(round_trace)
                break
            for action in actions[:3]:
                if not isinstance(action, dict):
                    continue
                tool = str(action.get("tool") or "")
                args = action.get("args") if isinstance(action.get("args"), dict) else {}
                key = _action_key(tool, args)
                if key in visited or tool == "finish":
                    continue
                visited.add(key)
                try:
                    rows = self.tools.execute(tool, args)
                    for row in rows:
                        memory_id = row.memory.memory_id
                        if memory_id in evidence:
                            existing = evidence[memory_id]
                            existing.score += row.score
                            existing.sources.extend(row.sources)
                            existing.paths.extend(row.paths)
                        else:
                            evidence[memory_id] = row
                    round_trace["tool_results"].append({
                        "tool": tool,
                        "args": args,
                        "count": len(rows),
                        "memory_ids": [row.memory.memory_id for row in rows],
                    })
                except Exception as exc:
                    round_trace["tool_results"].append({"tool": tool, "args": args, "error": repr(exc)})
            trace["rounds"].append(round_trace)

        final = sorted(evidence.values(), key=lambda row: (-row.score, row.memory.memory_id))[: self.final_k]
        for rank, row in enumerate(final, start=1):
            row.rank = rank
        trace["final"] = [_memory_view(row, include_assistant=parsed.primary().need_assistant_context) for row in final]
        trace["visited_actions"] = sorted(visited)
        return final, trace
