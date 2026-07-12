from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, DefaultDict, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .dataset import parse_datetime
from .io_utils import ensure_dir, normalize_text, read_json, stable_hash, write_json
from .schemas import GraphEdge, GraphNode, MemoryRecord, memories_from_payload


def canonical(value: Any) -> str:
    return normalize_text(str(value or ""))


def _node_id(node_type: str, label: str) -> str:
    return f"{node_type}_{stable_hash(node_type + '|' + canonical(label), 20)}"


def _memory_time(memory: MemoryRecord) -> datetime:
    dim = memory.dimension.time
    candidates = [dim.event_start, dim.valid_from, *(memory.provenance.source_times or [])]
    for value in candidates:
        if value:
            parsed = parse_datetime(value)
            if parsed.year > 1970:
                return parsed
    return datetime(1970, 1, 1)


class GraphStore:
    """Compact typed graph plus inverted indexes.

    The graph is stored as JSON so it can be inspected without Neo4j. Indexes make
    graph expansion cheap and deterministic while exposing graph-like tools to the
    active LLM controller.
    """

    def __init__(self, memories: Sequence[MemoryRecord]) -> None:
        self.memories: Dict[str, MemoryRecord] = {item.memory_id: item for item in memories}
        self.nodes: Dict[str, GraphNode] = {}
        self.edges: List[GraphEdge] = []
        self.adjacency: DefaultDict[str, List[GraphEdge]] = defaultdict(list)
        self.reverse_adjacency: DefaultDict[str, List[GraphEdge]] = defaultdict(list)
        self.indexes: Dict[str, DefaultDict[str, Set[str]]] = {
            name: defaultdict(set)
            for name in (
                "entity", "topic", "location", "session", "state_key", "keyword",
                "memory_type", "date", "relation_subject", "relation_object",
                "preference_target",
            )
        }

    def add_node(self, node: GraphNode) -> None:
        self.nodes.setdefault(node.node_id, node)

    def add_edge(self, edge: GraphEdge, *, dedupe: bool = True) -> None:
        if dedupe:
            marker = (edge.source, edge.target, edge.edge_type)
            for existing in self.adjacency.get(edge.source, []):
                if (existing.source, existing.target, existing.edge_type) == marker:
                    return
        self.edges.append(edge)
        self.adjacency[edge.source].append(edge)
        self.reverse_adjacency[edge.target].append(edge)

    def related_memory_ids(
        self,
        memory_ids: Sequence[str],
        *,
        edge_types: Optional[Set[str]] = None,
        limit: int = 20,
    ) -> List[Tuple[str, GraphEdge, List[str]]]:
        output: List[Tuple[str, GraphEdge, List[str]]] = []
        seen: Set[str] = set(memory_ids)
        queue: List[Tuple[str, List[str]]] = [(memory_id, [memory_id]) for memory_id in memory_ids]
        while queue and len(output) < limit:
            current, path = queue.pop(0)
            incident = list(self.adjacency.get(current, [])) + list(self.reverse_adjacency.get(current, []))
            for edge in incident:
                if edge_types and edge.edge_type not in edge_types:
                    continue
                other = edge.target if edge.source == current else edge.source
                if other in seen:
                    continue
                seen.add(other)
                next_path = path + [edge.edge_type, other]
                if other in self.memories:
                    output.append((other, edge, next_path))
                else:
                    queue.append((other, next_path))
                if len(output) >= limit:
                    break
        return output

    def lookup(self, index_name: str, value: str, limit: int = 20) -> List[str]:
        index = self.indexes.get(index_name)
        if not index:
            return []
        key = canonical(value)
        exact = list(index.get(key, set()))
        if len(exact) >= limit:
            return sorted(exact)[:limit]
        # Conservative substring fallback for names and topics.
        candidates = set(exact)
        if key:
            for indexed_key, memory_ids in index.items():
                if key in indexed_key or indexed_key in key:
                    candidates.update(memory_ids)
                    if len(candidates) >= limit * 3:
                        break
        return sorted(candidates)[:limit]

    def save(self, output_dir: Path | str) -> None:
        root = ensure_dir(output_dir)
        write_json(root / "memory_bank.json", [memory.to_dict() for memory in self.memories.values()])
        write_json(root / "nodes.json", [node.to_dict() for node in self.nodes.values()])
        write_json(root / "edges.json", [edge.to_dict() for edge in self.edges])
        write_json(
            root / "indexes.json",
            {
                name: {key: sorted(values) for key, values in index.items()}
                for name, index in self.indexes.items()
            },
        )
        edge_counts: DefaultDict[str, int] = defaultdict(int)
        node_counts: DefaultDict[str, int] = defaultdict(int)
        for edge in self.edges:
            edge_counts[edge.edge_type] += 1
        for node in self.nodes.values():
            node_counts[node.node_type] += 1
        write_json(
            root / "graph_stats.json",
            {
                "memory_count": len(self.memories),
                "node_count": len(self.nodes),
                "edge_count": len(self.edges),
                "node_types": dict(sorted(node_counts.items())),
                "edge_types": dict(sorted(edge_counts.items())),
                "index_cardinality": {name: len(index) for name, index in self.indexes.items()},
            },
        )

    @classmethod
    def load(cls, graph_dir: Path | str) -> "GraphStore":
        root = Path(graph_dir)
        store = cls(memories_from_payload(read_json(root / "memory_bank.json")))
        for row in read_json(root / "nodes.json"):
            store.add_node(GraphNode(**row))
        for row in read_json(root / "edges.json"):
            store.add_edge(GraphEdge(**row), dedupe=False)
        raw_indexes = read_json(root / "indexes.json")
        for name, index in raw_indexes.items():
            if name not in store.indexes:
                store.indexes[name] = defaultdict(set)
            for key, values in index.items():
                store.indexes[name][key].update(values)
        return store


class GraphBuilder:
    def __init__(self, *, add_weak_same_event_edges: bool = True) -> None:
        self.add_weak_same_event_edges = add_weak_same_event_edges

    def _dimension_node(
        self,
        store: GraphStore,
        memory_id: str,
        node_type: str,
        label: str,
        edge_type: str,
        *,
        attrs: Optional[Dict[str, Any]] = None,
        index_name: Optional[str] = None,
        weight: float = 1.0,
    ) -> None:
        if not str(label or "").strip():
            return
        node_id = _node_id(node_type, label)
        store.add_node(GraphNode(node_id=node_id, node_type=node_type, label=str(label), attrs=attrs or {}))
        store.add_edge(GraphEdge(source=memory_id, target=node_id, edge_type=edge_type, weight=weight))
        if index_name:
            store.indexes[index_name][canonical(label)].add(memory_id)

    def build(self, memories: Sequence[MemoryRecord]) -> GraphStore:
        store = GraphStore(memories)
        by_session: DefaultDict[str, List[MemoryRecord]] = defaultdict(list)
        by_state: DefaultDict[str, List[MemoryRecord]] = defaultdict(list)
        same_event_groups: DefaultDict[str, List[MemoryRecord]] = defaultdict(list)

        for memory in memories:
            dim = memory.dimension
            store.add_node(GraphNode(memory.memory_id, "memory", memory.content, {
                "memory_type": dim.memory_type,
                "confidence": memory.confidence,
            }))
            if dim.memory_type:
                store.indexes["memory_type"][canonical(dim.memory_type)].add(memory.memory_id)
            for keyword in dim.keywords:
                store.indexes["keyword"][canonical(keyword)].add(memory.memory_id)
            for entity in dim.entities:
                self._dimension_node(
                    store, memory.memory_id, "entity", entity.name, "ABOUT_ENTITY",
                    attrs={"entity_type": entity.entity_type, "role": entity.role},
                    index_name="entity",
                )
            for location in dim.locations:
                self._dimension_node(
                    store, memory.memory_id, "location", location, "AT_LOCATION",
                    index_name="location",
                )
            if dim.topic:
                self._dimension_node(store, memory.memory_id, "topic", dim.topic, "HAS_TOPIC", index_name="topic")
            if memory.provenance.session_id:
                self._dimension_node(
                    store, memory.memory_id, "session", memory.provenance.session_id,
                    "IN_SESSION", attrs={"session_index": memory.provenance.session_index},
                    index_name="session",
                )
                by_session[memory.provenance.session_id].append(memory)
            for field_name, value, edge_type in (
                ("event_start", dim.time.event_start, "EVENT_START"),
                ("event_end", dim.time.event_end, "EVENT_END"),
                ("valid_from", dim.time.valid_from, "VALID_FROM"),
                ("valid_to", dim.time.valid_to, "VALID_TO"),
            ):
                if value:
                    date_key = value[:10]
                    self._dimension_node(
                        store, memory.memory_id, "time", value, edge_type,
                        attrs={"field": field_name, "precision": dim.time.precision},
                        index_name="date",
                    )
                    store.indexes["date"][canonical(date_key)].add(memory.memory_id)
            if dim.state_key:
                self._dimension_node(
                    store, memory.memory_id, "state", dim.state_key, "ASSERTS_STATE",
                    attrs={"value": dim.state_value, "status": dim.state_status},
                    index_name="state_key",
                )
                by_state[canonical(dim.state_key)].append(memory)
            if dim.preference.target:
                self._dimension_node(
                    store, memory.memory_id, "preference", dim.preference.target,
                    "PREFERENCE_ABOUT", attrs=dim.preference.to_dict(),
                    index_name="preference_target",
                )
            relation = dim.relation
            if relation.subject:
                store.indexes["relation_subject"][canonical(relation.subject)].add(memory.memory_id)
            if relation.object:
                store.indexes["relation_object"][canonical(relation.object)].add(memory.memory_id)
            if relation.subject and relation.predicate and relation.object:
                subject_id = _node_id("relation_entity", relation.subject)
                object_id = _node_id("relation_entity", relation.object)
                store.add_node(GraphNode(subject_id, "relation_entity", relation.subject))
                store.add_node(GraphNode(object_id, "relation_entity", relation.object))
                store.add_edge(GraphEdge(memory.memory_id, subject_id, "RELATION_SUBJECT"))
                store.add_edge(GraphEdge(memory.memory_id, object_id, "RELATION_OBJECT"))
                store.add_edge(GraphEdge(
                    subject_id, object_id, "RELATION",
                    attrs={"predicate": relation.predicate, "memory_id": memory.memory_id},
                ))

            entity_key = "+".join(sorted(canonical(entity.name) for entity in dim.entities)[:3])
            topic_key = canonical(dim.topic)
            date_key = (dim.time.event_start or dim.time.raw or "")[:10]
            if entity_key and (topic_key or date_key):
                same_event_groups[f"{entity_key}|{topic_key}|{date_key}"].append(memory)

        # Session chronology is explicit and deterministic.
        for session_memories in by_session.values():
            ordered = sorted(session_memories, key=lambda item: (_memory_time(item), item.memory_id))
            for previous, current in zip(ordered, ordered[1:]):
                store.add_edge(GraphEdge(previous.memory_id, current.memory_id, "NEXT_IN_SESSION", 0.8))
                store.add_edge(GraphEdge(previous.memory_id, current.memory_id, "BEFORE", 0.8))

        # Mutable-state chains make knowledge updates directly traversable.
        for state_key, state_memories in by_state.items():
            ordered = sorted(state_memories, key=lambda item: (_memory_time(item), item.memory_id))
            for earlier, later in zip(ordered, ordered[1:]):
                old_value = canonical(earlier.dimension.state_value)
                new_value = canonical(later.dimension.state_value)
                edge_type = "SUPPORTS" if old_value and old_value == new_value else "SUPERSEDES"
                store.add_edge(GraphEdge(
                    later.memory_id,
                    earlier.memory_id,
                    edge_type,
                    attrs={"state_key": state_key, "old_value": earlier.dimension.state_value, "new_value": later.dimension.state_value},
                ))

        if self.add_weak_same_event_edges:
            for group in same_event_groups.values():
                if 1 < len(group) <= 12:
                    ordered = sorted(group, key=lambda item: item.memory_id)
                    for left, right in zip(ordered, ordered[1:]):
                        store.add_edge(GraphEdge(left.memory_id, right.memory_id, "SAME_EVENT", 0.55))
                        store.add_edge(GraphEdge(right.memory_id, left.memory_id, "SAME_EVENT", 0.55))

        return store


def build_sample_graph(sample_dir: Path, *, force: bool = False) -> Dict[str, Any]:
    memory_path = sample_dir / "memory_v2" / "all_memories.json"
    if not memory_path.exists():
        raise FileNotFoundError(f"run extraction first: {memory_path}")
    output_dir = sample_dir / "graph_v2"
    stats_path = output_dir / "graph_stats.json"
    if stats_path.exists() and not force:
        return {"status": "existing", **read_json(stats_path)}
    memories = memories_from_payload(read_json(memory_path))
    store = GraphBuilder().build(memories)
    store.save(output_dir)
    return {"status": "ok", **read_json(stats_path)}


def build_run_graphs(run_root: str, *, force: bool = False, question_types: Optional[Iterable[str]] = None) -> Dict[str, Any]:
    root = Path(run_root)
    manifest = read_json(root / "run_manifest.json")
    allowed = {str(value) for value in question_types or []}
    results: List[Dict[str, Any]] = []
    for sample in manifest.get("samples") or []:
        if allowed and sample.get("question_type") not in allowed:
            continue
        try:
            stats = build_sample_graph(Path(sample["sample_dir"]), force=force)
            results.append({**sample, **stats})
        except Exception as exc:
            results.append({**sample, "status": "failed", "error": repr(exc)})
    write_json(root / "graph_manifest_v2.json", {"samples": results})
    return {"samples": results}
