from longmemeval.graph_memory_v2.adaptive_qa_v7_operation import (
    deterministic_operation_answer,
    infer_contract_v3,
    needs_p9_audit,
)


def test_duration_precedes_generic_how_many() -> None:
    assert infer_contract_v3(
        "How many weeks passed between the two events?",
        {"hypotheses": []},
    ) == "duration"


def test_argmax_contract() -> None:
    assert infer_contract_v3(
        "Which store did I spend the most money at?",
        {"hypotheses": []},
    ) == "argmax_or_argmin"


def test_habitual_state_contract() -> None:
    assert infer_contract_v3(
        "What time do I usually go to the gym?",
        {"hypotheses": []},
    ) == "habitual_state"


def test_singular_multi_answer_triggers_audit() -> None:
    p5 = {"prediction": "a yellow dress and earrings"}
    p7 = {"prediction": "a yellow dress and earrings"}
    p8 = {
        "prediction": "a yellow dress and earrings",
        "support_source_ids": [1],
    }
    audit, reasons = needs_p9_audit(
        p5,
        p7,
        p8,
        "What did I buy for the birthday gift?",
        {"hypotheses": []},
    )
    assert audit
    assert "singular_question_multi_answer" in reasons


def test_compare_dates_final_answer_uses_label() -> None:
    candidate = {
        "answer": "wrong label",
        "answerable": True,
        "operation_data": {
            "type": "compare_dates",
            "items": [
                {
                    "label": "Workshop",
                    "date": "2023-05-20",
                    "support_source_ids": [1],
                },
                {
                    "label": "Webinar",
                    "date": "2023-03-24",
                    "support_source_ids": [2],
                },
            ],
        },
    }
    result = deterministic_operation_answer(
        candidate,
        contract="comparison",
        question="Which event did I attend first?",
    )
    assert result["answer"] == "Webinar"


def test_missing_comparison_side_forces_abstention() -> None:
    candidate = {
        "answer": "Alex",
        "answerable": True,
        "operation_data": {
            "type": "compare_dates",
            "items": [
                {
                    "label": "Alex",
                    "date": "2023-01-10",
                    "support_source_ids": [1],
                }
            ],
        },
    }
    result = deterministic_operation_answer(
        candidate,
        contract="comparison",
        question="Who became a parent first?",
    )
    assert result["answerable"] is False
    assert "Cannot be determined" in result["answer"]


def test_argmax_sums_values_by_label() -> None:
    candidate = {
        "answer": "wrong",
        "answerable": True,
        "operation_data": {
            "type": "argmax",
            "items": [
                {
                    "label": "Store A",
                    "value": 40,
                    "support_source_ids": [1],
                },
                {
                    "label": "Store A",
                    "value": 30,
                    "support_source_ids": [2],
                },
                {
                    "label": "Store B",
                    "value": 60,
                    "support_source_ids": [3],
                },
            ],
        },
    }
    result = deterministic_operation_answer(
        candidate,
        contract="argmax_or_argmin",
        question="Which store did I spend the most money at?",
    )
    assert result["answer"] == "Store A"
