from __future__ import annotations


EXPAND_SYSTEM_P9 = """Create compact retrieval queries for an evidence audit.

Do not answer the question. Produce generic paraphrases that cover alternate
ways the same event, state, preference, comparison, or transaction could have
been expressed. Do not use benchmark labels or case-specific examples. Return
exactly one JSON object."""


EXPAND_TEMPLATE_P9 = r"""
Question date:
{question_date}

Question:
{question}

Operation contract:
{contract}

Parsed query:
{query_json}

Return:
{
  "queries": ["short query"],
  "relation_verbs": ["verb or verb phrase"],
  "entity_hints": ["entity or category"],
  "time_scope": {
    "start": "",
    "end": "",
    "raw": ""
  }
}

Rules:
1. Return at most six queries.
2. For set-valued questions, include variants for discovering omitted members.
3. For state questions, include old and new formulations of the same state.
4. For preference questions, retrieve stable likes, dislikes, and relevant past
   experiences; do not retrieve live external availability.
5. Keep every query short.
"""


SPECIALIST_SYSTEM_P9 = """Answer from the supplied evidence views.

Build atomic supported facts before applying an operation. Do not assume that a
retrieved set is complete merely because several relevant facts were found.
For comparisons and timelines, preserve the label attached to each value or
date. For current or habitual state, prefer the latest valid explicit state.
For recommendation questions, return transferable preference criteria rather
than inventing live options.

The final answer must agree with operation_data. Return exactly one JSON object
and no markdown."""


SPECIALIST_TEMPLATE_P9 = r"""
Question date:
{question_date}

Question:
{question}

Operation contract:
{contract}

Structured P2 memory evidence:
{memory_json}

Full-bank V2 evidence census:
{v2_census_json}

Targeted source evidence:
{source_census_json}

State or preference view:
{profile_json}

Existing candidates:
{candidate_json}

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
  "operation_data": {
    "type": "direct|count_distinct|sum_by_group|argmax|argmin|compare_dates|compare_numbers|timeline|duration|state_resolution|preference_profile",
    "items": [
      {
        "label": "",
        "value": null,
        "unit": "",
        "date": "",
        "status": "observed|asserted|planned|projected|negated|uncertain",
        "support_memory_ids": [],
        "support_source_ids": []
      }
    ],
    "requested_unit": "",
    "result": ""
  },
  "coverage": {
    "complete": true,
    "possible_omissions": [],
    "reason": ""
  },
  "confidence": 0.0,
  "reasoning_summary": "brief grounded explanation"
}

Rules:
1. Every item used in an operation needs a support memory ID or source ID.
2. A count uses distinct atomic items, not a sentence containing a count.
3. sum_by_group groups numeric facts by label before comparing or totaling.
4. argmax/argmin returns the label, not the numeric value.
5. compare_dates returns the label attached to the earlier or later date.
6. A comparison with any required side missing is not answerable.
7. A timeline must exclude future plans when the question asks what happened.
8. A preference_profile requires positive criteria and may include negative
   criteria; location, current mood, and live availability are optional unless
   explicitly requested.
9. For a singular direct question, return only the requested item, without
   ancillary coordinated details.
"""


ADJUDICATE_SYSTEM_P9 = """Adjudicate answer candidates using the evidence.

Check exact requested scope, completeness of set-valued evidence, temporal
validity, state recency, and agreement between reasoning and the final answer.
You may correct all candidates when the evidence supports a different result.
Do not prefer a concrete answer when a required comparison side is missing.
Return exactly one JSON object."""


ADJUDICATE_TEMPLATE_P9 = r"""
Question:
{question}

Operation contract:
{contract}

Candidates:
{candidate_json}

Audit evidence:
{evidence_json}

Return:
{
  "answer": "minimal final answer",
  "answerable": true,
  "support_memory_ids": [],
  "support_source_ids": [],
  "operation_data": {
    "type": "direct|count_distinct|sum_by_group|argmax|argmin|compare_dates|compare_numbers|timeline|duration|state_resolution|preference_profile",
    "items": [],
    "requested_unit": "",
    "result": ""
  },
  "confidence": 0.0,
  "reason": "brief evidence-grounded reason"
}
"""
