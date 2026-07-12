from __future__ import annotations


PLAN_SYSTEM_P8 = """You plan evidence collection for a long-term-memory question.

Do not answer the question. Produce a generic answer contract, required evidence
slots, and short retrieval queries. Do not use benchmark category names or
case-specific examples. Return exactly one valid JSON object."""


PLAN_TEMPLATE_P8 = r"""
Question date:
{question_date}

Question:
{question}

Parsed query:
{query_json}

Deterministic contract hint:
{contract_hint}

Existing candidates:
{candidate_json}

Return:
{
  "answer_contract": "direct_value|direct_numeric|enumerate_count|sum|timeline|duration|state_resolution|comparison|preference_criteria|assistant_recall|insufficient_information_sensitive|other",
  "operator": "none|stated_count|count_distinct|sum|date_difference_days|date_difference_weeks|date_difference_months|time_add_minutes|timeline|state_resolution|comparison",
  "temporal_perspective": "current|historical|relative|none",
  "slots": [
    {
      "name": "generic evidence slot",
      "query": "short retrieval query",
      "semantic_type": "event|entity|time|state|preference|assistant_reply|number|other",
      "required": true
    }
  ],
  "retrieval_queries": ["short query"],
  "reason": "brief reason"
}
"""


SPECIALIST_SYSTEM_P8 = """Answer from the supplied evidence.

Use an atomic fact table before counting, summing, ordering, comparing, or
resolving state. A final aggregate need not appear verbatim when every operand
is grounded. Distinguish a stated count from counting a list of distinct
events. For current state, use the latest valid non-superseded state. For a
preference request, return transferable criteria unless a specific requested
option is grounded.

Return exactly one valid JSON object. Do not use outside knowledge."""


SPECIALIST_TEMPLATE_P8 = r"""
Question date:
{question_date}

Question:
{question}

Reconciled plan:
{plan_json}

Structured memory evidence:
{memory_json}

Compact source evidence:
{compact_source_json}

Slot-targeted source evidence:
{targeted_source_json}

Relevant V2 state/preference timeline:
{v2_timeline_json}

Return:
{
  "answer": "minimal final answer",
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
  "operation": "none|stated_count|count_distinct|sum|date_difference|time_add|timeline|state_resolution|comparison|preference_inference|assistant_recall|other",
  "calculation": {
    "operator": "none|stated_count|count_distinct|sum|date_difference_days|date_difference_weeks|date_difference_months|time_add_minutes",
    "operands": [],
    "unit": "",
    "result": ""
  },
  "fact_table": [
    {
      "fact_id": "f1",
      "subject": "",
      "predicate": "",
      "object": "",
      "numeric_value": null,
      "unit": "",
      "event_time": "",
      "valid_from": "",
      "valid_to": "",
      "modality": "observed|asserted|planned|projected|uncertain|negated",
      "support_memory_ids": [],
      "support_source_ids": []
    }
  ],
  "confidence": 0.0,
  "reasoning_summary": "brief grounded explanation"
}

Rules:
1. count_distinct operands must be atomic distinct fact IDs or item names.
2. stated_count operands must contain the stated number, not a phrase to count.
3. sum operands must contain only grounded numeric values.
4. The requested output unit controls duration formatting.
5. A concrete answer must cite valid support IDs.
6. Do not infer that the count is zero merely because no event was found.
"""


VERIFY_SYSTEM_P8 = """Select the best answer candidate using the evidence.

Check entailment, required-slot coverage, contradictions, temporal validity,
and calculation consistency. A concrete answer is not automatically better
than abstention. A supported calculation is valid even when the final aggregate
was not stated verbatim. Return exactly one valid JSON object."""


VERIFY_TEMPLATE_P8 = r"""
Question:
{question}

Plan:
{plan_json}

Candidates:
{candidate_json}

Compact verification evidence:
{evidence_json}

Return:
{
  "choice": "p5|p7|specialist|abstain",
  "assessments": {
    "p5": {
      "supported": true,
      "slot_coverage": 1.0,
      "contradicted": false,
      "calculation_consistent": true,
      "reason": ""
    },
    "p7": {
      "supported": true,
      "slot_coverage": 1.0,
      "contradicted": false,
      "calculation_consistent": true,
      "reason": ""
    },
    "specialist": {
      "supported": true,
      "slot_coverage": 1.0,
      "contradicted": false,
      "calculation_consistent": true,
      "reason": ""
    }
  },
  "reason": "brief reason"
}
"""
