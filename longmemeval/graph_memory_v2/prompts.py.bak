from __future__ import annotations

import json
from typing import Any, Dict, List


MEMORY_EXTRACTION_SYSTEM = """You are a precise long-term memory constructor.
Extract only information grounded in the supplied user turns. Resolve pronouns,
normalize relative dates from the timestamps, and return valid JSON only.
Never invent a dimension. Leave unsupported strings empty and lists empty.
The first overlap_count turns are context only and cannot be new memory sources."""


MEMORY_EXTRACTION_TEMPLATE = r"""
Build atomic, self-contained memory records from this conversation window.

The schema is designed for LongMemEval and extends DimMem while preserving its
original dimensions. Each record must use one or more window_source_ids from the
non-overlap suffix.

Output:
{
  "memories": [
    {
      "source_ids": [1],
      "content": "A self-contained statement.",
      "dimension": {
        "memory_type": "fact|episodic|profile",
        "time": {
          "event_start": "ISO date/datetime or empty",
          "event_end": "ISO date/datetime or empty",
          "valid_from": "ISO date/datetime or empty",
          "valid_to": "ISO date/datetime or empty",
          "precision": "year|month|day|datetime|range|unknown",
          "raw": "original temporal expression or empty"
        },
        "entities": [
          {"name": "canonical name", "type": "person|organization|place|object|product|project|tool|dataset|event|topic|other", "role": "optional"}
        ],
        "locations": ["physical, digital, platform or contextual location"],
        "topic": "short stable topic",
        "relation": {
          "subject": "canonical subject",
          "predicate": "stable relation/aspect",
          "object": "canonical object/value"
        },
        "state_key": "canonical mutable slot, e.g. user.residence or project.status",
        "state_value": "value asserted for the slot",
        "state_status": "active|historical|planned|unknown",
        "reason": "causal background or empty",
        "purpose": "goal/intent or empty",
        "preference": {
          "target": "preference target or empty",
          "polarity": "positive|negative|neutral|mixed|unknown",
          "strength": 0.0,
          "scope": "specific|contextual|general|unknown"
        },
        "modality": "asserted|planned|hypothetical|negated|uncertain",
        "keywords": ["3-8 precise retrieval anchors"]
      },
      "confidence": 0.0,
      "evidence_span": "short exact grounding phrase"
    }
  ]
}

Rules:
1. Keep memory content atomic: one independently retrievable fact/event/profile per record.
2. fact = stable or mutable state; episodic = dated event/action/plan; profile = preference, habit, capability or long-term goal.
3. Separate event time from validity time. A move happened on event_start; the new residence becomes valid_from that date.
4. state_key must be canonical and reusable across updates. Use it only for facts/preferences that may change.
5. For profile memories, populate preference only when the text really expresses a preference. Do not generalize one isolated event into a broad preference.
6. modality must preserve plans, negations, uncertainty and hypotheticals.
7. source_ids must point only to non-overlap turns: source_id > overlap_count.
8. Do not extract greetings, confirmations, formatting requests or one-off instructions without future value.
9. Use empty fields rather than guesses.

overlap_count: {overlap_count}

Conversation:
{conversation}
"""


QUERY_PARSE_SYSTEM = """You parse memory questions into reusable retrieval constraints.
Return valid JSON only. Do not use benchmark category labels as rules. Treat the
parse as hypotheses that can later be revised by active retrieval."""


QUERY_PARSE_TEMPLATE = r"""
Question date: {question_date}
Question: {question}

Return up to three hypotheses:
{
  "question": "...",
  "question_date": "...",
  "hypotheses": [
    {
      "query_anchor": "retrieval-friendly rewrite using 'the user'",
      "target_memory_types": ["fact","episodic","profile"],
      "entities": ["canonical entity anchors"],
      "keywords": ["short exact anchors"],
      "time_constraint": {
        "operator": "on|before|after|between|around|",
        "start": "ISO date or empty",
        "end": "ISO date or empty",
        "raw": "original expression or empty"
      },
      "locations": [],
      "state_keys": ["likely mutable slots if applicable"],
      "relation_hints": ["relations needed, e.g. caused_by, supersedes, same_time"],
      "answer_dim": "content|time|location|reason|purpose|state_value|preference|assistant_reply|",
      "need_assistant_context": false,
      "need_multi_hop": false,
      "expected_evidence_count": 1,
      "missing_slots": ["intermediate facts that may need retrieval"],
      "confidence": 0.0
    }
  ]
}

Important:
- Asking "when" means answer_dim=time; do not turn the unknown answer into a filter.
- Asking "why" means answer_dim=reason.
- Questions about what the assistant recommended/said require assistant context.
- Relative times must be normalized with question_date when possible.
- Multi-hop is true only when one piece of evidence is likely needed to locate another.
"""


ACTIVE_ROUTER_SYSTEM = """You control a graph memory search. Select tools only when
current evidence is insufficient. You may revise the initial query hypothesis from
new evidence. Return JSON only. Do not answer from unstated knowledge."""


ACTIVE_ROUTER_TEMPLATE = r"""
Question: {question}
Question date: {question_date}

Current query hypothesis:
{query_json}

Current evidence:
{evidence_json}

Visited actions:
{visited_json}

Available tools:
- search_text(query, entities=[], keywords=[], memory_types=[], limit=8)
- expand_entity(entity, limit=8)
- expand_topic(topic, limit=8)
- expand_time(start="", end="", operator="", limit=8)
- expand_session(session_id, limit=8)
- follow_state_chain(state_key="", memory_id="", limit=10)
- expand_relations(memory_ids=[], edge_types=[], limit=10)
- inspect_sources(memory_ids=[], include_assistant=false)
- finish(memory_ids=[], missing_slots=[], answer_plan="")

Return:
{
  "mode": "navigate|finish",
  "actions": [
    {"tool": "expand_entity", "args": {"entity": "...", "limit": 8}}
  ],
  "updated_missing_slots": [],
  "reason": "brief evidence-grounded reason"
}

Constraints:
- At most 3 actions.
- Never repeat an identical visited action.
- Prefer dimensions newly revealed by evidence, especially time anchors, entities,
  state keys and update chains.
- Use finish once enough evidence exists; do not browse for its own sake.
"""


QA_SYSTEM = """Answer only from the supplied memories and source replies.
Give the minimal answer requested. If evidence is insufficient, say
"Cannot be determined from the conversation." Return valid JSON only."""


QA_TEMPLATE = r"""
Question date: {question_date}
Question: {question}

Evidence:
{evidence_json}

Return:
{
  "answer": "minimal answer",
  "support_memory_ids": ["m..."],
  "confidence": 0.0,
  "reasoning_summary": "one short, evidence-grounded summary"
}
"""


JUDGE_SYSTEM = """Judge whether the prediction is semantically correct relative to
the reference answer. Be lenient about wording, date formatting and equivalent
units, but not about unsupported extra claims. Return JSON only."""


JUDGE_TEMPLATE = r"""
Question type: {question_type}
Question: {question}
Reference answer: {gold_answer}
Prediction: {prediction}

Return:
{
  "label": "CORRECT|WRONG",
  "reason": "brief reason"
}
"""


def render(template: str, **kwargs: Any) -> str:
    values = {
        key: json.dumps(value, ensure_ascii=False, indent=2)
        if isinstance(value, (dict, list))
        else str(value or "")
        for key, value in kwargs.items()
    }
    return template.format(**values)

DIMMEM_V1_EXTRACTION_SYSTEM = """You are a precise long-term memory constructor.
Extract only grounded, self-contained memories. Return valid JSON only. The first
overlap_count turns are context only and cannot be new memory sources."""

DIMMEM_V1_EXTRACTION_TEMPLATE = r"""
Extract atomic memories using the original six operational DimMem dimensions.

Output:
{
  "memories": [
    {
      "source_ids": [1],
      "content": "self-contained memory",
      "memory_type": "fact|episodic|profile",
      "time": "normalized time or empty",
      "location": "location or empty",
      "reason": "reason or empty",
      "purpose": "purpose or empty",
      "keywords": ["3-8 anchors"],
      "confidence": 0.0,
      "evidence_span": "short grounding phrase"
    }
  ]
}

Rules:
1. Use only source_id > overlap_count as new memory sources.
2. Resolve pronouns and normalize relative time using the timestamps.
3. Keep unsupported fields empty; do not invent metadata.
4. fact is factual state, episodic is a concrete event, profile is a stable preference/habit/capability.
5. Do not extract greetings or low-value transient wording.

overlap_count: {overlap_count}
Conversation:
{conversation}
"""
