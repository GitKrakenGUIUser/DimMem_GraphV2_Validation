from longmemeval.graph_memory_v2.source_qa_v4 import (
    deterministic_calculation,
    is_abstention,
)


def test_abstention_variants() -> None:
    assert is_abstention(
        "The information provided is not enough. "
        "You did not mention the event."
    )
    assert is_abstention(
        "Cannot be determined from the conversation."
    )
    assert not is_abstention("February 1st")


def test_date_difference() -> None:
    result = deterministic_calculation(
        {
            "operator": "date_difference_days",
            "operands": ["2023-04-26", "2023-05-05"],
            "unit": "days",
        }
    )
    assert result == "9 days"


def test_time_add() -> None:
    result = deterministic_calculation(
        {
            "operator": "time_add_minutes",
            "operands": ["7:00 AM", 120],
            "unit": "",
        }
    )
    assert result == "9:00 AM"


def test_empty_count_is_not_zero() -> None:
    result = deterministic_calculation(
        {
            "operator": "count_distinct",
            "operands": [],
        }
    )
    assert result is None
