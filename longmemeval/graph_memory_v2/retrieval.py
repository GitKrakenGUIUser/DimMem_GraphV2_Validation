from __future__ import annotations

import math
from collections import Counter, defaultdict
from datetime import datetime
from typing import Any, DefaultDict, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .dataset import parse_datetime
from .embedding import Embedder, HashEmbedder, cosine
from .graph_store import GraphStore, canonical
from .io_utils import normalize_text, tokenize
from .schemas import MemoryRecord, ParsedQueryV2, QueryHypothesis, RetrievalRecord


class BM25Index:
    def __init__(self, documents: Dict[str, str], *, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.tokens = {doc_id: tokenize(text) for doc_id, text in documents.items()}
        self.lengths = {doc_id: len(tokens) for doc_id, tokens in self.tokens.items()}
        self.avgdl = sum(self.lengths.values()) / max(1, len(self.lengths))
        self.tf = {doc_id: Counter(tokens) for doc_id, tokens in self.tokens.items()}
        df: Counter[str] = Counter()
        for tokens in self.tokens.values():
            df.update(set(tokens))
        self.idf = {
            token: math.log(1.0 + (len(self.tokens) - count + 0.5) / (count + 0.5))
            for token, count in df.items()
        }

    def search(self, query: str, limit: int = 20) -> List[Tuple[str, float]]:
        query_tokens = tokenize(query)
        scores: List[Tuple[str, float]] = []
        for doc_id, frequencies in self.tf.items():
            score = 0.0
            dl = self.lengths[doc_id]
            for token in query_tokens:
                if token not in frequencies:
                    continue
                tf = frequencies[token]
                numerator = tf * (self.k1 + 1.0)
                denominator = tf + self.k1 * (1.0 - self.b + self.b * dl / max(1.0, self.avgdl))
                score += self.idf.get(token, 0.0) * numerator / denominator
            if score > 0:
                scores.append((doc_id, score))
        scores.sort(key=lambda pair: (-pair[1], pair[0]))
        return scores[:limit]


def _date_value(value: str) -> Optional[datetime]:
    if not value:
        return None
    parsed = parse_datetime(value)
    return parsed if parsed.year > 1970 else None


def _time_match(memory: MemoryRecord, hypothesis: QueryHypothesis) -> float:
    constraint = hypothesis.time_constraint
    if not (constraint.start or constraint.end or constraint.raw):
        return 0.0
    memory_times = [
        memory.dimension.time.event_start,
        memory.dimension.time.event_end,
        memory.dimension.time.valid_from,
        memory.dimension.time.valid_to,
        *(memory.provenance.source_times or []),
    ]
    parsed_times = [value for value in (_date_value(item) for item in memory_times) if value]
    if not parsed_times:
        raw_query = normalize_text(constraint.raw)
        raw_memory = normalize_text(memory.dimension.time.raw)
        return 1.0 if raw_query and raw_query in raw_memory else 0.0
    start = _date_value(constraint.start)
    end = _date_value(constraint.end)
    operator = constraint.operator
    candidate = parsed_times[0]
    if operator == "before" and start:
        return 1.0 if candidate < start else 0.0
    if operator == "after" and start:
        return 1.0 if candidate > start else 0.0
    if operator == "between" and start and end:
        return 1.0 if start <= candidate <= end else 0.0
    if operator in {"on", "around"} and start:
        delta = abs((candidate - start).total_seconds())
        if candidate.date() == start.date():
            return 1.0
        return max(0.0, 1.0 - delta / (86400.0 * (7 if operator == "around" else 1)))
    if start:
        return 1.0 if candidate.date() == start.date() else 0.0
    return 0.0


def _contains_any(values: Iterable[str], targets: Iterable[str]) -> float:
    values_norm = [canonical(value) for value in values if canonical(value)]
    targets_norm = [canonical(value) for value in targets if canonical(value)]
    if not targets_norm:
        return 0.0
    hits = 0
    for target in targets_norm:
        if any(target in value or value in target for value in values_norm):
            hits += 1
    return hits / len(targets_norm)


def dimension_score(memory: MemoryRecord, hypothesis: QueryHypothesis, *, enhanced: bool = True) -> float:
    dim = memory.dimension
    active: List[float] = []
    if hypothesis.target_memory_types:
        active.append(1.0 if dim.memory_type in hypothesis.target_memory_types else 0.0)
    if hypothesis.entities:
        active.append(_contains_any((entity.name for entity in dim.entities), hypothesis.entities))
    if hypothesis.keywords:
        fields = dim.keywords + [memory.content, dim.topic, dim.reason, dim.purpose]
        active.append(_contains_any(fields, hypothesis.keywords))
    if hypothesis.locations:
        active.append(_contains_any(dim.locations, hypothesis.locations))
    if hypothesis.time_constraint.start or hypothesis.time_constraint.end or hypothesis.time_constraint.raw:
        active.append(_time_match(memory, hypothesis))
    if enhanced and hypothesis.state_keys:
        active.append(_contains_any([dim.state_key], hypothesis.state_keys))
    if enhanced and hypothesis.relation_hints:
        relation_fields = [dim.relation.predicate, dim.state_status, dim.modality]
        active.append(_contains_any(relation_fields, hypothesis.relation_hints))
    if not active:
        return 0.0
    return sum(active) / len(active)


def reciprocal_rank_fusion(
    routes: Dict[str, List[Tuple[str, float]]],
    *,
    k: int = 60,
    route_weights: Optional[Dict[str, float]] = None,
) -> List[Tuple[str, float, List[Dict[str, Any]]]]:
    route_weights = route_weights or {}
    totals: DefaultDict[str, float] = defaultdict(float)
    sources: DefaultDict[str, List[Dict[str, Any]]] = defaultdict(list)
    for route_name, rows in routes.items():
        weight = float(route_weights.get(route_name, 1.0))
        for rank, (memory_id, raw_score) in enumerate(rows, start=1):
            totals[memory_id] += weight / (k + rank)
            sources[memory_id].append({
                "route": route_name,
                "rank": rank,
                "raw_score": float(raw_score),
                "rrf_contribution": weight / (k + rank),
            })
    ordered = sorted(totals.items(), key=lambda pair: (-pair[1], pair[0]))
    return [(memory_id, score, sources[memory_id]) for memory_id, score in ordered]


class StaticRetriever:
    def __init__(
        self,
        store: GraphStore,
        *,
        embedder: Optional[Embedder] = None,
        original_projection: bool = False,
    ) -> None:
        self.store = store
        self.original_projection = original_projection
        self.embedder = embedder or HashEmbedder()
        self.documents = {
            memory_id: self._document_text(memory)
            for memory_id, memory in store.memories.items()
        }
        self.bm25 = BM25Index(self.documents)
        memory_ids = list(self.documents)
        vectors = self.embedder.encode([self.documents[memory_id] for memory_id in memory_ids])
        self.memory_ids = memory_ids
        self.vectors = dict(zip(memory_ids, vectors))

    def _document_text(self, memory: MemoryRecord) -> str:
        if not self.original_projection:
            return memory.searchable_text()
        projection = memory.dimension.original_dimmem_projection()
        fields = [memory.content, projection["memory_type"], projection["time"], projection["location"], projection["reason"], projection["purpose"], " ".join(projection["keywords"])]
        return " | ".join(str(field) for field in fields if field)

    def _query_text(self, parsed: ParsedQueryV2) -> str:
        parts = [parsed.question]
        for hypothesis in parsed.hypotheses:
            parts.extend([
                hypothesis.query_anchor,
                " ".join(hypothesis.entities),
                " ".join(hypothesis.keywords),
                " ".join(hypothesis.locations),
                " ".join(hypothesis.state_keys if not self.original_projection else []),
            ])
        return " | ".join(part for part in parts if part)

    def retrieve(
        self,
        parsed: ParsedQueryV2,
        *,
        route_k: int = 20,
        final_k: int = 15,
        graph_expand: bool = False,
        graph_expand_k: int = 20,
    ) -> List[RetrievalRecord]:
        query_text = self._query_text(parsed)
        bm25 = self.bm25.search(query_text, route_k)
        query_vector = self.embedder.encode([query_text])[0]
        dense = sorted(
            ((memory_id, cosine(query_vector, vector)) for memory_id, vector in self.vectors.items()),
            key=lambda pair: (-pair[1], pair[0]),
        )[:route_k]
        dimension_rows: List[Tuple[str, float]] = []
        for memory_id, memory in self.store.memories.items():
            score = max(
                dimension_score(memory, hypothesis, enhanced=not self.original_projection)
                for hypothesis in parsed.hypotheses
            )
            if score > 0:
                dimension_rows.append((memory_id, score))
        dimension_rows.sort(key=lambda pair: (-pair[1], pair[0]))
        dimension_rows = dimension_rows[:route_k]

        routes = {"bm25": bm25, "dense": dense, "dimension": dimension_rows}
        fused = reciprocal_rank_fusion(
            routes,
            route_weights={"bm25": 1.0, "dense": 1.0, "dimension": 1.15 if not self.original_projection else 1.0},
        )
        source_map = {memory_id: sources for memory_id, _, sources in fused}
        score_map = {memory_id: score for memory_id, score, _ in fused}
        path_map: DefaultDict[str, List[List[str]]] = defaultdict(list)

        if graph_expand:
            seeds = [memory_id for memory_id, _, _ in fused[: min(8, len(fused))]]
            edge_types = {
                "SUPERSEDES", "SUPPORTS", "SAME_EVENT", "NEXT_IN_SESSION", "BEFORE",
                "ABOUT_ENTITY", "HAS_TOPIC", "AT_LOCATION", "ASSERTS_STATE", "IN_SESSION",
            }
            for memory_id, edge, path in self.store.related_memory_ids(
                seeds, edge_types=edge_types, limit=graph_expand_k
            ):
                graph_score = 0.006 * max(0.25, float(edge.weight))
                score_map[memory_id] = score_map.get(memory_id, 0.0) + graph_score
                source_map.setdefault(memory_id, []).append({
                    "route": "graph_static",
                    "edge_type": edge.edge_type,
                    "rrf_contribution": graph_score,
                })
                path_map[memory_id].append(path)

        ordered = sorted(score_map.items(), key=lambda pair: (-pair[1], pair[0]))[:final_k]
        output: List[RetrievalRecord] = []
        for rank, (memory_id, score) in enumerate(ordered, start=1):
            output.append(RetrievalRecord(
                memory=self.store.memories[memory_id],
                score=float(score),
                rank=rank,
                sources=source_map.get(memory_id, []),
                paths=path_map.get(memory_id, []),
            ))
        return output

    def search_text(
        self,
        query: str,
        *,
        entities: Sequence[str] = (),
        keywords: Sequence[str] = (),
        memory_types: Sequence[str] = (),
        limit: int = 8,
    ) -> List[RetrievalRecord]:
        hypothesis = QueryHypothesis(
            query_anchor=query,
            entities=list(entities),
            keywords=list(keywords),
            target_memory_types=list(memory_types),
        )
        parsed = ParsedQueryV2(question=query, hypotheses=[hypothesis])
        return self.retrieve(parsed, route_k=max(limit * 3, 12), final_k=limit)
