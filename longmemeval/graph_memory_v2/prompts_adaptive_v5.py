from __future__ import annotations


PLAN_SYSTEM_V5 = """You plan evidence use for a long-term-memory question.

Classify the answer contract and create generic evidence slots. Do not answer
the question. Do not use benchmark category names. Allow compositional
reasoning when two grounded facts form a clear bridge, such as:
event -> organization and organization -> date.

Return exactly one valid JSON object."""


PLAN_TEMPLATE_V5 = r"""
Question date:
{question_date}

Question:
{question}

Parsed query:
{query_json}

Compact candidate:
{base_candidate_json}

Return:
{
  "answer_contract": "direct_value|entity|count|sum|timeline|state_resolution|preference_criteria|assistant_recall|other",
  "operator": "none|date_difference_days|time_add_minutes|sum|count_distinct|timeline|state_resolution",
  "target_count": 0,
  "temporal_perspective": "current|historical|relative|none",
  "allow_composition": true,
  "allow_semantic_bridge": true,
  "allow_state_projection": false,
  "slots": [
    {
      "name": "generic slot",
      "query": "short retrieval query",
      "semantic_type": "event|entity|time|state|preference|assistant_reply|number|other",
      "required": true
    }
  ],
  "retrieval_queries": ["short query"],
  "output_constraints": [
    "generic constraint on answer form"
  ],
  "reason": "brief reason"
}

Rules:
1. "How many" means count_distinct unless the question clearly asks for a
   supplied count.
2. "Total", "combined", or "altogether" normally means sum.
3. "Order ... earliest to latest" means timeline.
4. "Currently", "now", or "latest" means state_resolution.
5. Recommendation and decision questions grounded in user preferences should
   use preference_criteria. Do not require dates or budget unless the question
   explicitly asks for them.
6. Assistant-recall questions require assistant_reply evidence.
7. Relative dates must be resolved from question_date.
8. A planned state transition may project to current state only when its due
   time is before question_date and no later contradiction exists.
"""


SPECIALIST_SYSTEM_V5 = """You are a specialist evidence reasoner.

Use compact and targeted evidence. Compositional inference is allowed when each
link is grounded. Semantic bridges are allowed when the relation is ordinary
and unambiguous, for example a crystal chandelier can satisfy a broad
"jewelry/decorative piece" reference when the surrounding event and date match.

For count, sum, time arithmetic and timelines, extract atomic supported items
first, then let the calculation object represent the operation. Do not demand
that the conversation explicitly state the final count or sum.

For current-state questions, build a timeline. A stated plan whose intended
completion time is before the question date may update the projected current
state when there is no later contradiction. Mark this as projected.

For preference questions, answer with the user's criteria or decision-relevant
preferences. Do not invent a named item in a new city or domain unless that
specific item is grounded.

Return exactly one valid JSON object."""


SPECIALIST_TEMPLATE_V5 = r"""
Question date:
{question_date}

Question:
{question}

Plan:
{plan_json}

Structured memories:
{memory_evidence_json}

Compact source evidence:
{compact_source_json}

Targeted source evidence:
{targeted_source_json}

Return:
{
  "answer": "minimal answer",
  "answerable": true,
  "support_memory_ids": ["m..."],
  "support_source_ids": [1],
  "required_slots": [
    {
      "name": "slot",
      "value": "grounded value",
      "support_memory_ids": ["m..."],
      "support_source_ids": [1]
    }
  ],
  "unfilled_slots": [],
  "operation": "none|date_difference|time_add|sum|count_distinct|timeline|state_resolution|preference_inference|assistant_recall|other",
  "calculation": {
    "operator": "none|date_difference_days|time_add_minutes|sum|count_distinct",
    "operands": [],
    "unit": "",
    "result": ""
  },
  "fact_table": [
    {
      "subject": "",
      "predicate": "",
      "object": "",
      "value": "",
      "unit": "",
      "event_time": "",
      "valid_from": "",
      "valid_to": "",
      "modality": "observed|asserted|planned|projected|uncertain",
      "support_memory_ids": [],
      "support_source_ids": []
    }
  ],
  "confidence": 0.0,
  "reasoning_summary": "brief grounded explanation"
}

Rules:
1. Fill every required slot or mark it unfilled.
2. Do not answer "cannot determine" merely because the final aggregate is not
   explicitly stated; compute it from grounded atomic facts.
3. For counts, operands must be distinct grounded items, not prose duplicates.
4. For sums, operands must be grounded numeric values within the requested
   time range.
5. For timeline, include every grounded item requested by target_count when
   target_count is nonzero.
6. For state_resolution, compare timestamps and modality.
7. For preference_criteria, return transferable criteria, not an unsupported
   named recommendation.
8. For assistant_recall, cite source IDs that contain assistant_reply.
"""


ARBITER_SYSTEM_V5 = """Choose between two grounded candidates.

Prefer a supported concrete answer over abstention. Prefer deterministic
calculation over unsupported natural-language inference. Preserve a compact
candidate when the expanded candidate merely becomes more conservative because
of extra distractors. Prefer a specialist candidate when it fills previously
missing required slots, resolves a state timeline, or follows a preference
answer contract.

Return exactly one valid JSON object."""


ARBITER_TEMPLATE_V5 = r"""
Question:
{question}

Plan:
{plan_json}

Compact candidate:
{base_candidate_json}

Specialist candidate:
{specialist_candidate_json}

Return:
{
  "choice": "base|specialist",
  "reason": "brief reason"
}
"""
