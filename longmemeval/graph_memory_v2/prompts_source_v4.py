from __future__ import annotations

QA_SYSTEM_V4 = """You are an evidence-grounded long-term memory answerer.

Use both structured memories and reconstructed source turns. First bind every
required answer component to evidence. Then perform any deterministic
calculation. The final answer must agree with the calculation object.

Never invent a prior assistant response. Never treat absence of a mentioned
event as proof that its count is zero. For recommendation questions, transfer
stable preferences to the requested new target; exact trip dates or budget are
not required unless the question asks for them.

Return exactly one valid JSON object and no markdown."""


QA_TEMPLATE_V4 = r"""
Question date:
{question_date}

Question:
{question}

Parsed query:
{query_json}

Structured memories:
{memory_evidence_json}

Reconstructed source turns:
{source_evidence_json}

Return:
{
  "answer": "minimal final answer",
  "answerable": true,
  "support_memory_ids": ["m..."],
  "support_source_ids": [1],
  "required_slots": [
    {
      "name": "generic slot",
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
  "confidence": 0.0,
  "reasoning_summary": "brief evidence-grounded explanation"
}

Rules:
1. Every concrete answer must cite at least one memory ID or source ID.
2. The answer field must equal the calculation result whenever operator is not
   "none".
3. For date difference, put two ISO dates in calculation.operands.
4. For arrival-time reasoning, put start time and travel minutes in operands
   and use time_add_minutes.
5. For sums, list only supported numeric operands.
6. For counts, count distinct supported entities/events. If there are no
   positive mentions, do not infer zero; mark the answer unanswerable.
7. For current-state questions, prefer the latest valid update. For historical
   questions, use the value valid at the referenced time.
8. For assistant-recall questions, answer only from reconstructed
   assistant_reply fields.
9. A recommendation may be answered from stable preference evidence even when
   target-specific dates or budget are unavailable.
10. If a required fact remains unsupported, answer:
    "Cannot be determined from the conversation."
"""


VERIFY_SYSTEM_V4 = """You verify a candidate answer against supplied evidence.

Correct only evidence-grounding, completeness, temporal perspective, arithmetic,
and answer/calculation consistency. Do not add outside knowledge. Return exactly
one JSON object."""


VERIFY_TEMPLATE_V4 = r"""
Question:
{question}

Candidate:
{candidate_json}

Structured memories:
{memory_evidence_json}

Reconstructed source turns:
{source_evidence_json}

Return:
{
  "accept": true,
  "corrected": {
    "answer": "",
    "answerable": true,
    "support_memory_ids": [],
    "support_source_ids": [],
    "required_slots": [],
    "unfilled_slots": [],
    "operation": "none",
    "calculation": {
      "operator": "none",
      "operands": [],
      "unit": "",
      "result": ""
    },
    "confidence": 0.0,
    "reasoning_summary": ""
  },
  "reason": "brief reason"
}
"""


JUDGE_SYSTEM_V4 = """Judge semantic correctness against the reference answer.

Equivalent wording, units and formatting are acceptable. A partial multi-part
answer is wrong. Return exactly one valid JSON object."""


JUDGE_TEMPLATE_V4 = r"""
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
