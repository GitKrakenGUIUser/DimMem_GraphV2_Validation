from longmemeval.graph_memory_v2.adaptive_qa_v5 import (
    choose_candidate_safely,
    infer_contract_hint,
    needs_specialist,
)


def test_more_context_cannot_replace_supported_base_with_abstention() -> None:
    base = {
        "answer": "70 pounds",
        "answerable": True,
        "support_source_ids": [1, 2],
        "calculation": {
            "operator": "sum",
            "operands": [50, 20],
            "result": "70 pounds",
        },
    }
    specialist = {
        "answer": "Cannot be determined from the conversation.",
        "answerable": False,
        "support_source_ids": [],
        "calculation": {"operator": "none", "operands": []},
    }
    choice, _ = choose_candidate_safely(
        base,
        specialist,
        {"answer_contract": "sum"},
    )
    assert choice == "base"


def test_specialist_can_replace_abstained_base() -> None:
    base = {
        "answer": "Cannot be determined from the conversation.",
        "answerable": False,
    }
    specialist = {
        "answer": "my aunt",
        "answerable": True,
        "support_source_ids": [174],
    }
    choice, _ = choose_candidate_safely(
        base,
        specialist,
        {"answer_contract": "entity"},
    )
    assert choice == "specialist"


def test_preference_contract_comes_from_query_schema() -> None:
    parsed = {
        "hypotheses": [
            {
                "target_memory_types": ["profile", "episodic"],
                "answer_dim": "preference",
            }
        ]
    }
    assert infer_contract_hint(
        "Can you suggest a hotel for my trip?",
        parsed,
    ) == "preference_criteria"


def test_supported_aggregate_is_routed_to_specialist_but_preserved() -> None:
    parsed = {"hypotheses": []}
    base = {
        "prediction": "2",
        "answerable": True,
        "operation": "count_distinct",
        "unfilled_slots": [],
    }
    needed, reasons = needs_specialist(
        base,
        "How many projects have I led?",
        parsed,
    )
    assert needed is True
    assert "specialist_contract:count" in reasons
