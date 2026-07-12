from longmemeval.graph_memory_v2.adaptive_qa_v6_lite import (
    infer_contract_v2,
    reconcile_plan,
    safe_calculation,
)


def test_total_precedes_how_many() -> None:
    parsed = {"hypotheses": []}
    assert infer_contract_v2(
        "How many total dollars did I spend altogether?",
        parsed,
    ) == "sum"


def test_forward_advice_is_not_assistant_recall() -> None:
    parsed = {
        "hypotheses": [
            {
                "need_assistant_context": True,
                "answer_dim": "preference",
                "target_memory_types": ["profile"],
            }
        ]
    }
    assert infer_contract_v2(
        "Do you have any tips based on my preferences?",
        parsed,
    ) == "preference_criteria"


def test_explicit_previous_answer_is_assistant_recall() -> None:
    parsed = {"hypotheses": [{"need_assistant_context": True}]}
    assert infer_contract_v2(
        "What did you recommend in your previous response?",
        parsed,
    ) == "assistant_recall"


def test_count_phrase_is_not_counted_as_one_item() -> None:
    assert safe_calculation(
        {
            "operator": "count_distinct",
            "operands": ["7 short stories"],
        },
        "enumerate_count",
    ) is None


def test_stated_count_extracts_number() -> None:
    assert safe_calculation(
        {
            "operator": "stated_count",
            "operands": ["7 short stories"],
        },
        "direct_numeric",
    ) == "7"


def test_plan_contract_is_reconciled() -> None:
    result = reconcile_plan(
        "How many plants did I acquire?",
        {"hypotheses": []},
        {
            "answer_contract": "direct_value",
            "operator": "none",
            "slots": [],
        },
    )
    assert result["answer_contract"] == "enumerate_count"
    assert result["operator"] == "count_distinct"
    assert result["slots"]
