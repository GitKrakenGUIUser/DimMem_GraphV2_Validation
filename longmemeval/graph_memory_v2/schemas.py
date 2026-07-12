from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence


VALID_MEMORY_TYPES = {"fact", "episodic", "profile"}
VALID_MODALITIES = {"asserted", "planned", "hypothetical", "negated", "uncertain"}
VALID_STATE_STATUS = {"active", "historical", "planned", "unknown"}
VALID_ENTITY_TYPES = {
    "person", "organization", "place", "object", "product", "project",
    "tool", "dataset", "event", "topic", "other",
}
VALID_TIME_PRECISIONS = {"year", "month", "day", "datetime", "range", "unknown"}
VALID_PREFERENCE_POLARITIES = {"positive", "negative", "neutral", "mixed", "unknown"}
VALID_PREFERENCE_SCOPES = {"specific", "contextual", "general", "unknown"}


def clean(value: Any) -> str:
    return str(value or "").strip()


def unique_strings(values: Any, *, lower_dedupe: bool = False) -> List[str]:
    if values is None:
        return []
    if isinstance(values, str):
        values = [values]
    if not isinstance(values, (list, tuple, set)):
        return []
    result: List[str] = []
    seen = set()
    for value in values:
        text = clean(value)
        if not text:
            continue
        marker = text.casefold() if lower_dedupe else text
        if marker in seen:
            continue
        seen.add(marker)
        result.append(text)
    return result


def clamp_float(value: Any, default: float = 0.0, low: float = 0.0, high: float = 1.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return max(low, min(high, number))


@dataclass
class TimeDimension:
    """Event time and fact-validity time are deliberately separated.

    event_start/event_end: when an event occurred.
    valid_from/valid_to: when a state/fact was true.
    raw: original temporal expression for auditability.
    """

    event_start: str = ""
    event_end: str = ""
    valid_from: str = ""
    valid_to: str = ""
    precision: str = "unknown"
    raw: str = ""

    @classmethod
    def from_dict(cls, payload: Any) -> "TimeDimension":
        data = payload if isinstance(payload, dict) else {}
        precision = clean(data.get("precision")).lower()
        if precision not in VALID_TIME_PRECISIONS:
            precision = "unknown"
        return cls(
            event_start=clean(data.get("event_start") or data.get("start")),
            event_end=clean(data.get("event_end") or data.get("end")),
            valid_from=clean(data.get("valid_from")),
            valid_to=clean(data.get("valid_to")),
            precision=precision,
            raw=clean(data.get("raw") or data.get("text")),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def primary(self) -> str:
        return self.event_start or self.valid_from or self.raw


@dataclass
class EntityRef:
    name: str = ""
    entity_type: str = "other"
    role: str = ""

    @classmethod
    def from_dict(cls, payload: Any) -> "EntityRef":
        if isinstance(payload, str):
            return cls(name=clean(payload))
        data = payload if isinstance(payload, dict) else {}
        entity_type = clean(data.get("type") or data.get("entity_type")).lower()
        if entity_type not in VALID_ENTITY_TYPES:
            entity_type = "other"
        return cls(
            name=clean(data.get("name") or data.get("text")),
            entity_type=entity_type,
            role=clean(data.get("role")),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RelationDimension:
    subject: str = ""
    predicate: str = ""
    object: str = ""

    @classmethod
    def from_dict(cls, payload: Any) -> "RelationDimension":
        data = payload if isinstance(payload, dict) else {}
        return cls(
            subject=clean(data.get("subject")),
            predicate=clean(data.get("predicate") or data.get("relation")),
            object=clean(data.get("object")),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def canonical_key(self) -> str:
        return "|".join(part.casefold() for part in (self.subject, self.predicate) if part)


@dataclass
class PreferenceDimension:
    target: str = ""
    polarity: str = "unknown"
    strength: float = 0.0
    scope: str = "unknown"

    @classmethod
    def from_dict(cls, payload: Any) -> "PreferenceDimension":
        data = payload if isinstance(payload, dict) else {}
        polarity = clean(data.get("polarity")).lower()
        if polarity not in VALID_PREFERENCE_POLARITIES:
            polarity = "unknown"
        scope = clean(data.get("scope")).lower()
        if scope not in VALID_PREFERENCE_SCOPES:
            scope = "unknown"
        return cls(
            target=clean(data.get("target")),
            polarity=polarity,
            strength=clamp_float(data.get("strength"), default=0.0),
            scope=scope,
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EnhancedDimension:
    """Operational memory dimensions.

    The first six fields preserve the original DimMem interface. The remaining
    fields expose structures needed by temporal, multi-session, update and
    preference reasoning without relying on benchmark-specific rules.
    """

    memory_type: str = ""
    time: TimeDimension = field(default_factory=TimeDimension)
    locations: List[str] = field(default_factory=list)
    reason: str = ""
    purpose: str = ""
    keywords: List[str] = field(default_factory=list)

    entities: List[EntityRef] = field(default_factory=list)
    topic: str = ""
    relation: RelationDimension = field(default_factory=RelationDimension)
    state_key: str = ""
    state_value: str = ""
    state_status: str = "unknown"
    preference: PreferenceDimension = field(default_factory=PreferenceDimension)
    modality: str = "asserted"

    @classmethod
    def from_dict(cls, payload: Any) -> "EnhancedDimension":
        data = payload if isinstance(payload, dict) else {}
        memory_type = clean(data.get("memory_type")).lower()
        if memory_type not in VALID_MEMORY_TYPES:
            memory_type = ""

        modality = clean(data.get("modality")).lower()
        if modality not in VALID_MODALITIES:
            modality = "asserted"

        state_status = clean(data.get("state_status")).lower()
        if state_status not in VALID_STATE_STATUS:
            state_status = "unknown"

        raw_entities = data.get("entities") or []
        if isinstance(raw_entities, (str, dict)):
            raw_entities = [raw_entities]
        entities: List[EntityRef] = []
        seen = set()
        for item in raw_entities if isinstance(raw_entities, list) else []:
            entity = EntityRef.from_dict(item)
            if not entity.name:
                continue
            marker = entity.name.casefold()
            if marker in seen:
                continue
            seen.add(marker)
            entities.append(entity)

        locations = data.get("locations")
        if locations is None:
            locations = data.get("location")

        time_payload = data.get("time")
        if isinstance(time_payload, str):
            time_payload = {"event_start": time_payload, "raw": time_payload}

        return cls(
            memory_type=memory_type,
            time=TimeDimension.from_dict(time_payload),
            locations=unique_strings(locations, lower_dedupe=True),
            reason=clean(data.get("reason")),
            purpose=clean(data.get("purpose")),
            keywords=unique_strings(data.get("keywords"), lower_dedupe=True),
            entities=entities,
            topic=clean(data.get("topic")),
            relation=RelationDimension.from_dict(data.get("relation")),
            state_key=clean(data.get("state_key")),
            state_value=clean(data.get("state_value")),
            state_status=state_status,
            preference=PreferenceDimension.from_dict(data.get("preference")),
            modality=modality,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "memory_type": self.memory_type,
            "time": self.time.to_dict(),
            "locations": list(self.locations),
            "reason": self.reason,
            "purpose": self.purpose,
            "keywords": list(self.keywords),
            "entities": [entity.to_dict() for entity in self.entities],
            "topic": self.topic,
            "relation": self.relation.to_dict(),
            "state_key": self.state_key,
            "state_value": self.state_value,
            "state_status": self.state_status,
            "preference": self.preference.to_dict(),
            "modality": self.modality,
        }

    def original_dimmem_projection(self) -> Dict[str, Any]:
        """Project to the original DimMem schema for controlled ablations."""
        return {
            "memory_type": self.memory_type,
            "time": self.time.primary(),
            "location": self.locations[0] if self.locations else "",
            "reason": self.reason,
            "purpose": self.purpose,
            "keywords": list(self.keywords),
        }

    def searchable_text(self) -> str:
        entity_text = " ".join(entity.name for entity in self.entities)
        relation_text = " ".join(
            part for part in (
                self.relation.subject,
                self.relation.predicate,
                self.relation.object,
            ) if part
        )
        preference_text = " ".join(
            part for part in (
                self.preference.target,
                self.preference.polarity,
                self.preference.scope,
            ) if part and part != "unknown"
        )
        time_text = " ".join(
            part for part in (
                self.time.event_start,
                self.time.event_end,
                self.time.valid_from,
                self.time.valid_to,
                self.time.raw,
            ) if part
        )
        return " | ".join(
            part
            for part in (
                self.memory_type,
                entity_text,
                self.topic,
                relation_text,
                self.state_key,
                self.state_value,
                time_text,
                " ".join(self.locations),
                self.reason,
                self.purpose,
                preference_text,
                self.modality,
                " ".join(self.keywords),
            )
            if clean(part)
        )


@dataclass
class Provenance:
    session_id: str = ""
    session_index: int = -1
    source_ids: List[int] = field(default_factory=list)
    source_uids: List[str] = field(default_factory=list)
    source_times: List[str] = field(default_factory=list)
    source_role: str = "user"
    window_index: int = -1

    @classmethod
    def from_dict(cls, payload: Any) -> "Provenance":
        data = payload if isinstance(payload, dict) else {}
        source_ids: List[int] = []
        for value in data.get("source_ids") or []:
            try:
                source_ids.append(int(value))
            except (TypeError, ValueError):
                continue
        try:
            session_index = int(data.get("session_index", -1))
        except (TypeError, ValueError):
            session_index = -1
        try:
            window_index = int(data.get("window_index", -1))
        except (TypeError, ValueError):
            window_index = -1
        return cls(
            session_id=clean(data.get("session_id")),
            session_index=session_index,
            source_ids=source_ids,
            source_uids=unique_strings(data.get("source_uids")),
            source_times=unique_strings(data.get("source_times")),
            source_role=clean(data.get("source_role") or "user"),
            window_index=window_index,
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MemoryRecord:
    memory_id: str = ""
    content: str = ""
    dimension: EnhancedDimension = field(default_factory=EnhancedDimension)
    provenance: Provenance = field(default_factory=Provenance)
    confidence: float = 0.0
    evidence_span: str = ""
    assistant_replies: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, payload: Any) -> "MemoryRecord":
        data = payload if isinstance(payload, dict) else {}
        return cls(
            memory_id=clean(data.get("memory_id") or data.get("id")),
            content=clean(data.get("content")),
            dimension=EnhancedDimension.from_dict(data.get("dimension")),
            provenance=Provenance.from_dict(data.get("provenance")),
            confidence=clamp_float(data.get("confidence"), default=0.0),
            evidence_span=clean(data.get("evidence_span")),
            assistant_replies=unique_strings(data.get("assistant_replies")),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "content": self.content,
            "dimension": self.dimension.to_dict(),
            "provenance": self.provenance.to_dict(),
            "confidence": self.confidence,
            "evidence_span": self.evidence_span,
            "assistant_replies": list(self.assistant_replies),
        }

    def searchable_text(self) -> str:
        return " | ".join(part for part in (self.content, self.dimension.searchable_text()) if part)


@dataclass
class TimeConstraint:
    operator: str = ""
    start: str = ""
    end: str = ""
    raw: str = ""

    @classmethod
    def from_dict(cls, payload: Any) -> "TimeConstraint":
        if isinstance(payload, str):
            return cls(raw=clean(payload))
        data = payload if isinstance(payload, dict) else {}
        return cls(
            operator=clean(data.get("operator")).lower(),
            start=clean(data.get("start")),
            end=clean(data.get("end")),
            raw=clean(data.get("raw")),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class QueryHypothesis:
    query_anchor: str = ""
    target_memory_types: List[str] = field(default_factory=list)
    entities: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    time_constraint: TimeConstraint = field(default_factory=TimeConstraint)
    locations: List[str] = field(default_factory=list)
    state_keys: List[str] = field(default_factory=list)
    relation_hints: List[str] = field(default_factory=list)
    answer_dim: str = ""
    need_assistant_context: bool = False
    need_multi_hop: bool = False
    expected_evidence_count: int = 1
    missing_slots: List[str] = field(default_factory=list)
    confidence: float = 1.0

    @classmethod
    def from_dict(cls, payload: Any) -> "QueryHypothesis":
        data = payload if isinstance(payload, dict) else {}
        target_types = [
            value for value in unique_strings(
                data.get("target_memory_types") or data.get("target_memory_type"),
                lower_dedupe=True,
            )
            if value in VALID_MEMORY_TYPES
        ]
        try:
            expected = int(data.get("expected_evidence_count", 1))
        except (TypeError, ValueError):
            expected = 1
        return cls(
            query_anchor=clean(data.get("query_anchor")),
            target_memory_types=target_types,
            entities=unique_strings(data.get("entities"), lower_dedupe=True),
            keywords=unique_strings(data.get("keywords"), lower_dedupe=True),
            time_constraint=TimeConstraint.from_dict(data.get("time_constraint") or data.get("time")),
            locations=unique_strings(data.get("locations") or data.get("location"), lower_dedupe=True),
            state_keys=unique_strings(data.get("state_keys") or data.get("state_key"), lower_dedupe=True),
            relation_hints=unique_strings(data.get("relation_hints"), lower_dedupe=True),
            answer_dim=clean(data.get("answer_dim")).lower(),
            need_assistant_context=bool(data.get("need_assistant_context", False)),
            need_multi_hop=bool(data.get("need_multi_hop", False)),
            expected_evidence_count=max(1, min(5, expected)),
            missing_slots=unique_strings(data.get("missing_slots")),
            confidence=clamp_float(data.get("confidence"), default=1.0),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query_anchor": self.query_anchor,
            "target_memory_types": list(self.target_memory_types),
            "entities": list(self.entities),
            "keywords": list(self.keywords),
            "time_constraint": self.time_constraint.to_dict(),
            "locations": list(self.locations),
            "state_keys": list(self.state_keys),
            "relation_hints": list(self.relation_hints),
            "answer_dim": self.answer_dim,
            "need_assistant_context": self.need_assistant_context,
            "need_multi_hop": self.need_multi_hop,
            "expected_evidence_count": self.expected_evidence_count,
            "missing_slots": list(self.missing_slots),
            "confidence": self.confidence,
        }


@dataclass
class ParsedQueryV2:
    question: str = ""
    question_date: str = ""
    hypotheses: List[QueryHypothesis] = field(default_factory=list)
    raw: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: Any, *, question: str = "", question_date: str = "") -> "ParsedQueryV2":
        data = payload if isinstance(payload, dict) else {}
        raw_hypotheses = data.get("hypotheses")
        if not isinstance(raw_hypotheses, list):
            raw_hypotheses = [data]
        hypotheses = [QueryHypothesis.from_dict(item) for item in raw_hypotheses]
        hypotheses = [item for item in hypotheses if item.query_anchor or item.keywords or item.entities]
        if not hypotheses:
            hypotheses = [QueryHypothesis(query_anchor=question)]
        hypotheses.sort(key=lambda item: item.confidence, reverse=True)
        return cls(
            question=clean(data.get("question") or question),
            question_date=clean(data.get("question_date") or question_date),
            hypotheses=hypotheses[:3],
            raw=dict(data),
        )

    def primary(self) -> QueryHypothesis:
        return self.hypotheses[0] if self.hypotheses else QueryHypothesis(query_anchor=self.question)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "question_date": self.question_date,
            "hypotheses": [item.to_dict() for item in self.hypotheses],
        }


@dataclass
class GraphNode:
    node_id: str
    node_type: str
    label: str
    attrs: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GraphEdge:
    source: str
    target: str
    edge_type: str
    weight: float = 1.0
    attrs: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RetrievalRecord:
    memory: MemoryRecord
    score: float
    rank: int = 0
    sources: List[Dict[str, Any]] = field(default_factory=list)
    paths: List[List[str]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        data = self.memory.to_dict()
        data.update({
            "score": self.score,
            "rank": self.rank,
            "retrieval_sources": self.sources,
            "graph_paths": self.paths,
        })
        return data


def memories_from_payload(payload: Any) -> List[MemoryRecord]:
    if isinstance(payload, dict):
        payload = payload.get("memories") or payload.get("records") or payload.get("all_memories") or []
    if not isinstance(payload, list):
        return []
    result = []
    for item in payload:
        memory = MemoryRecord.from_dict(item)
        if memory.content:
            result.append(memory)
    return result


__all__ = [
    "EnhancedDimension",
    "EntityRef",
    "GraphEdge",
    "GraphNode",
    "MemoryRecord",
    "ParsedQueryV2",
    "PreferenceDimension",
    "Provenance",
    "QueryHypothesis",
    "RelationDimension",
    "RetrievalRecord",
    "TimeConstraint",
    "TimeDimension",
    "VALID_MEMORY_TYPES",
    "clean",
    "memories_from_payload",
    "unique_strings",
]
