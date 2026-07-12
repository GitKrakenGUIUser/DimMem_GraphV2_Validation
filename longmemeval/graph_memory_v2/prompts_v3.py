from __future__ import annotations


ACTIVE_ROUTER_SYSTEM = """You control a dimension-grounded graph memory search.

Your job is evidence collection, not answering the question. Use the structured
coverage state as a constraint. You may revise the initial parse when newly
retrieved evidence reveals a better entity, time anchor, state key, relation or
session.

Return one valid JSON object only. Never use benchmark category names or
case-specific rules."""


ACTIVE_ROUTER_TEMPLATE = r"""
Question: {question}
Question date: {question_date}

All query hypotheses:
{query_json}

Evidence coverage state:
{coverage_json}

Current evidence:
{evidence_json}

Visited actions:
{visited_json}

Available tools:
- search_text(query, entities=[], keywords=[], memory_types=[], limit=10)
- expand_entity(entity, limit=10)
- expand_topic(topic, limit=10)
- expand_time(start="", end="", operator="", limit=10)
- expand_session(session_id, limit=10)
- follow_state_chain(state_key="", memory_id="", limit=12)
- expand_relations(memory_ids=[], edge_types=[], limit=12)
- inspect_sources(memory_ids=[], include_assistant=false)

Return:
{
  "active_hypothesis": 0,
  "mode": "navigate|finish",
  "actions": [
    {
      "tool": "search_text",
      "args": {
        "query": "...",
        "entities": [],
        "keywords": [],
        "memory_types": [],
        "limit": 10
      }
    }
  ],
  "updated_missing_slots": [],
  "revised_constraints": {
    "entities": [],
    "time_anchor": "",
    "state_keys": [],
    "relation_hints": []
  },
  "reason": "brief evidence-grounded reason"
}

Rules:
1. At most three actions.
2. Finish only when coverage.must_continue is false and the evidence supports
   every part of the requested answer.
3. If expected_evidence_count is greater than one, seek distinct supporting
   facts rather than near-duplicate memories.
4. Prefer complementary action families in one round: semantic, time, state,
   relation, source, entity, session or topic.
5. When a mutable state key is present, inspect its update chain before
   finishing.
6. When assistant context is required, inspect source replies before finishing.
7. Use newly discovered cues from evidence to redirect later retrieval.
8. Never repeat an identical visited action.
9. Do not browse merely to use the full round budget.
"""


QA_SYSTEM = """Answer only from the supplied memories and source replies.

First construct an evidence table internally: identify the required answer
slots, bind each slot to supporting memory IDs, resolve temporal perspective
(current versus historical), then perform any grounded comparison, counting,
date difference or addition. Do not let unrelated ambiguity invalidate a
directly supported calculation.

If conflicting memories form an update chain, choose the value appropriate to
the question time. If a required slot is genuinely unsupported, answer
"Cannot be determined from the conversation."

Return one valid JSON object only."""


QA_TEMPLATE = r"""
Question date: {question_date}
Question: {question}

Parsed query hypotheses:
{query_json}

Evidence:
{evidence_json}

Return:
{
  "answer": "minimal final answer",
  "support_memory_ids": ["m..."],
  "confidence": 0.0,
  "required_slots": [
    {
      "name": "generic evidence slot",
      "value": "grounded value or empty",
      "support_memory_ids": ["m..."]
    }
  ],
  "unfilled_slots": [],
  "conflicts": [
    {
      "slot": "",
      "values": [],
      "resolution": ""
    }
  ],
  "operation": "none|compare|count|sum|date_difference|timeline|state_resolution|preference_inference|other",
  "answerable": true,
  "reasoning_summary": "short evidence-grounded summary"
}

Rules:
1. Answer every requested component; do not return a partial multi-part answer.
2. For current-state questions, prefer the latest valid non-superseded value.
3. For historical questions, choose the value valid at the referenced event or
   date, not the latest value.
4. A later unrelated event does not create a conflict with a clearly identified
   target event.
5. Generalize a preference only when the evidence supports the requested scope.
6. Use only listed evidence and source replies.
"""


JUDGE_SYSTEM = """Judge semantic correctness against the reference answer.

Be lenient about wording, date formatting and equivalent units, but do not mark
an abstention as correct when the reference contains a concrete answer. A
partial answer to a multi-part question is wrong. Unsupported extra claims that
change the answer are wrong.

Return one valid JSON object only."""


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
