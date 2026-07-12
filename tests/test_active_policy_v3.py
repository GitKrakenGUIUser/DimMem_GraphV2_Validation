from longmemeval.graph_memory_v2.active_agent_v3 import (
    coverage_snapshot,
    select_diverse_actions,
)
from longmemeval.graph_memory_v2.schemas import (
    MemoryRecord,
    ParsedQueryV2,
    RetrievalRecord,
)


def _record(
    memory_id: str,
    *,
    content: str,
    state_key: str = "",
    assistant: bool = False,
    event_start: str = "",
) -> RetrievalRecord:
    memory = MemoryRecord.from_dict(
        {
            "memory_id": memory_id,
            "content": content,
            "dimension": {
                "memory_type": "fact",
                "state_key": state_key,
                "time": {
                    "event_start": event_start,
                },
            },
            "provenance": {
                "session_id": "s1",
                "source_ids": [1],
            },
            "assistant_replies": (
                ["Use the blue option."] if assistant else []
            ),
        }
    )
    return RetrievalRecord(
        memory=memory,
        score=0.1,
    )


def test_multihop_query_requires_a_bridge_probe() -> None:
    parsed = ParsedQueryV2.from_dict(
        {
            "question": "Combine two earlier events.",
            "hypotheses": [
                {
                    "query_anchor": "two earlier events",
                    "need_multi_hop": True,
                    "expected_evidence_count": 2,
                    "missing_slots": ["first event", "second event"],
                }
            ],
        }
    )
    coverage = coverage_snapshot(
        parsed,
        [_record("m1", content="One event happened.")],
        set(),
    )
    assert coverage["must_continue"] is True
    assert "bridge_search_not_attempted" in coverage["hard_gaps"]


def test_assistant_query_requires_source_inspection() -> None:
    parsed = ParsedQueryV2.from_dict(
        {
            "question": "What did the assistant recommend?",
            "hypotheses": [
                {
                    "query_anchor": "assistant recommendation",
                    "need_assistant_context": True,
                }
            ],
        }
    )
    coverage = coverage_snapshot(
        parsed,
        [_record("m1", content="The user asked for advice.")],
        set(),
    )
    assert coverage["must_continue"] is True
    assert (
        "assistant_source_not_inspected"
        in coverage["hard_gaps"]
    )


def test_state_query_requires_update_chain_probe() -> None:
    parsed = ParsedQueryV2.from_dict(
        {
            "question": "Where is the item now?",
            "hypotheses": [
                {
                    "query_anchor": "current item location",
                    "state_keys": ["item.location"],
                    "answer_dim": "state_value",
                }
            ],
        }
    )
    coverage = coverage_snapshot(
        parsed,
        [
            _record(
                "m1",
                content="The item was under the bed.",
                state_key="item.location",
            )
        ],
        set(),
    )
    assert coverage["must_continue"] is True
    assert "state_chain_not_checked" in coverage["hard_gaps"]


def test_action_selection_prefers_different_families() -> None:
    actions = [
        {
            "tool": "search_text",
            "args": {"query": "alpha"},
        },
        {
            "tool": "search_text",
            "args": {"query": "beta"},
        },
        {
            "tool": "expand_time",
            "args": {"start": "2025-01-01"},
        },
        {
            "tool": "expand_relations",
            "args": {"memory_ids": ["m1"]},
        },
    ]
    selected = select_diverse_actions(
        actions,
        set(),
        limit=3,
    )
    assert [action["tool"] for action in selected] == [
        "search_text",
        "expand_time",
        "expand_relations",
    ]
