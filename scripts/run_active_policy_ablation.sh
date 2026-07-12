#!/usr/bin/env bash
set -euo pipefail

: "${RUN_ROOT:?Set RUN_ROOT to the prepared LongMemEval run root}"
: "${EMBEDDING_MODEL:?Set EMBEDDING_MODEL to the local sentence-transformers model path}"

WORKERS="${WORKERS:-0}"
ROUTE_WORKERS="${ROUTE_WORKERS:-0}"
TOOL_WORKERS="${TOOL_WORKERS:-0}"
ROUTE_K="${ROUTE_K:-30}"
INITIAL_K="${INITIAL_K:-15}"
FINAL_K="${FINAL_K:-18}"

run_retrieval() {
  local name="$1"
  local rounds="$2"
  shift 2

  echo "===== Retrieval: ${name} ====="
  env "$@" \
    python -m longmemeval.graph_memory_v2 retrieve \
      --run-root "$RUN_ROOT" \
      --mode graph_active \
      --output-name "$name" \
      --route-k "$ROUTE_K" \
      --initial-k "$INITIAL_K" \
      --final-k "$FINAL_K" \
      --max-rounds "$rounds" \
      --router-mode llm \
      --embedder sentence-transformers \
      --embedding-model "$EMBEDDING_MODEL" \
      --device cpu \
      --workers "$WORKERS" \
      --route-workers "$ROUTE_WORKERS" \
      --tool-workers "$TOOL_WORKERS" \
      --force \
      --fail-fast

  echo "===== QA/Judge: ${name} ====="
  python -m longmemeval.graph_memory_v2 qa-judge \
    --run-root "$RUN_ROOT" \
    --retrieval-name "$name" \
    --workers "$WORKERS" \
    --force \
    --fail-fast

  python -m longmemeval.graph_memory_v2 report \
    --run-root "$RUN_ROOT" \
    --retrieval-name "$name" \
    --workers "$WORKERS" \
    --fail-fast
}

# P0: exact legacy active agent, but with a fresh output name.
run_retrieval graph_active_p0_legacy 3 \
  DIMMEM_V3_LEGACY=1

# P1: only increase the maximum round count. No new gate/reranker.
run_retrieval graph_active_p1_round5 5 \
  DIMMEM_V3_LEGACY=0 \
  DIMMEM_V3_COVERAGE_GATE=0 \
  DIMMEM_V3_ALL_HYPOTHESES=0 \
  DIMMEM_V3_DIVERSE_ACTIONS=0 \
  DIMMEM_V3_COVERAGE_RERANK=0

# P2: add the generic evidence-completion gate.
run_retrieval graph_active_p2_gate 4 \
  DIMMEM_V3_LEGACY=0 \
  DIMMEM_V3_COVERAGE_GATE=1 \
  DIMMEM_V3_ALL_HYPOTHESES=0 \
  DIMMEM_V3_DIVERSE_ACTIONS=0 \
  DIMMEM_V3_COVERAGE_RERANK=0

# P3: gate + all hypotheses + complementary action directions.
run_retrieval graph_active_p3_planner 4 \
  DIMMEM_V3_LEGACY=0 \
  DIMMEM_V3_COVERAGE_GATE=1 \
  DIMMEM_V3_ALL_HYPOTHESES=1 \
  DIMMEM_V3_DIVERSE_ACTIONS=1 \
  DIMMEM_V3_COVERAGE_RERANK=0

# P4: full policy, including structural/MMR reranking.
run_retrieval graph_active_p4_full 4 \
  DIMMEM_V3_LEGACY=0 \
  DIMMEM_V3_COVERAGE_GATE=1 \
  DIMMEM_V3_ALL_HYPOTHESES=1 \
  DIMMEM_V3_DIVERSE_ACTIONS=1 \
  DIMMEM_V3_COVERAGE_RERANK=1 \
  DIMMEM_V3_MAX_TOTAL_TOOL_CALLS=9 \
  DIMMEM_V3_ROUTER_EVIDENCE_K=18

echo "===== Suggested paired comparisons ====="
for pair in \
  "graph_active_p0_legacy graph_active_p1_round5" \
  "graph_active_p1_round5 graph_active_p2_gate" \
  "graph_active_p2_gate graph_active_p3_planner" \
  "graph_active_p3_planner graph_active_p4_full" \
  "graph_active_p0_legacy graph_active_p4_full"
do
  set -- $pair
  python -m longmemeval.graph_memory_v2 compare \
    --run-root "$RUN_ROOT" \
    --baseline "$1" \
    --candidate "$2"
done
