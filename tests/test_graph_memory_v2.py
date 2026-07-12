from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from longmemeval.graph_memory_v2.active_agent import ActiveReconstructionAgent
from longmemeval.graph_memory_v2.dataset import build_windows, flatten_user_turns
from longmemeval.graph_memory_v2.graph_store import GraphBuilder
from longmemeval.graph_memory_v2.retrieval import StaticRetriever
from longmemeval.graph_memory_v2.schemas import MemoryRecord, ParsedQueryV2


class GraphMemoryV2Tests(unittest.TestCase):
    def make_memory(self, memory_id: str, content: str, state_value: str, time: str) -> MemoryRecord:
        return MemoryRecord.from_dict({
            "memory_id": memory_id,
            "content": content,
            "dimension": {
                "memory_type": "fact",
                "time": {"valid_from": time, "precision": "day"},
                "entities": [{"name": "the user", "type": "person"}],
                "state_key": "user.residence",
                "state_value": state_value,
                "state_status": "active",
                "keywords": ["residence", state_value],
            },
            "provenance": {
                "session_id": memory_id,
                "source_ids": [1 if memory_id == "m1" else 2],
                "source_times": [time],
            },
            "confidence": 0.9,
        })

    def test_update_chain_is_explicit(self):
        old = self.make_memory("m1", "The user lives in Boston.", "Boston", "2025-01-01")
        new = self.make_memory("m2", "The user lives in Seattle.", "Seattle", "2025-03-01")
        graph = GraphBuilder().build([old, new])
        edges = {(edge.source, edge.target, edge.edge_type) for edge in graph.edges}
        self.assertIn(("m2", "m1", "SUPERSEDES"), edges)
        self.assertEqual(set(graph.lookup("state_key", "user.residence")), {"m1", "m2"})

    def test_original_projection_hides_new_dimensions(self):
        memory = self.make_memory("m1", "The user lives in Boston.", "Boston", "2025-01-01")
        projection = memory.dimension.original_dimmem_projection()
        self.assertEqual(set(projection), {"memory_type", "time", "location", "reason", "purpose", "keywords"})
        self.assertNotIn("state_key", projection)

    def test_active_fallback_can_follow_state(self):
        old = self.make_memory("m1", "The user lives in Boston.", "Boston", "2025-01-01")
        new = self.make_memory("m2", "The user lives in Seattle.", "Seattle", "2025-03-01")
        graph = GraphBuilder().build([old, new])
        retriever = StaticRetriever(graph)
        parsed = ParsedQueryV2.from_dict({
            "question": "Where does the user live now?",
            "hypotheses": [{
                "query_anchor": "current user residence",
                "state_keys": ["user.residence"],
                "answer_dim": "state_value",
                "need_multi_hop": True,
            }],
        })
        agent = ActiveReconstructionAgent(
            store=graph,
            retriever=retriever,
            router_mode="heuristic",
            max_rounds=2,
            final_k=5,
        )
        records, trace = agent.retrieve(parsed, route_k=5, initial_k=1)
        self.assertIn("m2", {row.memory.memory_id for row in records})
        self.assertTrue(trace["visited_actions"])

    def test_window_overlap_sources(self):
        item = {
            "haystack_session_ids": ["s1"],
            "haystack_dates": ["2025/01/01 10:00"],
            "haystack_sessions": [[
                {"role": "user", "content": f"turn {index}"}
                for index in range(1, 7)
            ]],
        }
        turns = flatten_user_turns(item)
        windows = build_windows(turns, window_size=4, overlap=2)
        self.assertEqual(windows[0]["overlap_count"], 0)
        self.assertEqual(windows[1]["overlap_count"], 2)
        self.assertEqual(windows[1]["messages"][0]["content"], "turn 3")


if __name__ == "__main__":
    unittest.main()
