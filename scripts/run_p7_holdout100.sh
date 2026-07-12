#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# 0. Paths
###############################################################################

cd ~/projects/DimMem_GraphV2_Validation

# 修改成你的100 Case子集实际路径
INPUT="${INPUT:-../DimMemTest-test/DimMem/data/longmemeval_s_cleaned_random100_non_overlap.json}"

OUTPUT_ROOT="${OUTPUT_ROOT:-./results/graph_memory_v2}"
RUN_NAME="${RUN_NAME:-holdout100_p7_frozen}"
RUN_ROOT="${OUTPUT_ROOT}/${RUN_NAME}"

EMBEDDING_MODEL="${EMBEDDING_MODEL:-/home/bowenzheng/huggingface/hub/all-MiniLM-L6-v2}"

# 首次从头运行时：
#   FRESH_START=1 bash scripts/run_p7_holdout100.sh
#
# 中断后断点续跑时：
#   bash scripts/run_p7_holdout100.sh
FRESH_START="${FRESH_START:-0}"

###############################################################################
# 1. Concurrency
###############################################################################

# API阶段同时处理的Case数量。
# 不建议对100个Case使用0，否则会将全部Case同时提交给API。
API_WORKERS="${API_WORKERS:-100}"

# CPU/磁盘阶段并行数。
CPU_WORKERS="${CPU_WORKERS:-100}"

# 每个Case内部：
WINDOW_WORKERS="${WINDOW_WORKERS:-1}"
ROUTE_WORKERS="${ROUTE_WORKERS:-3}"
TOOL_WORKERS="${TOOL_WORKERS:-3}"

###############################################################################
# 2. Basic validation
###############################################################################

if [[ ! -f "$INPUT" ]]; then
  echo "Input file not found: $INPUT" >&2
  exit 1
fi

if [[ -z "${OPENAI_BASE_URL:-}" ]]; then
  echo "OPENAI_BASE_URL is missing" >&2
  exit 1
fi

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "OPENAI_API_KEY is missing" >&2
  exit 1
fi

if [[ -z "${MODEL_NAME:-}" ]]; then
  echo "MODEL_NAME is missing" >&2
  exit 1
fi

if [[ ! -d "$EMBEDDING_MODEL" ]]; then
  echo "Embedding model not found: $EMBEDDING_MODEL" >&2
  exit 1
fi

echo "============================================================"
echo "INPUT=$INPUT"
echo "RUN_ROOT=$RUN_ROOT"
echo "MODEL_NAME=$MODEL_NAME"
echo "EMBEDDING_MODEL=$EMBEDDING_MODEL"
echo "API_WORKERS=$API_WORKERS"
echo "============================================================"

###############################################################################
# 3. Validate the new dataset
###############################################################################

CASE_COUNT="$(
  jq '
    if type == "array" then
      length
    elif type == "object" and (.data | type) == "array" then
      .data | length
    else
      0
    end
  ' "$INPUT"
)"

echo "Input case count: $CASE_COUNT"

if [[ "$CASE_COUNT" -ne 100 ]]; then
  echo "Expected 100 cases, found $CASE_COUNT" >&2
  exit 1
fi

UNIQUE_QIDS="$(
  jq -r '
    if type == "array" then
      .[]
    else
      .data[]
    end
    | .question_id
  ' "$INPUT" \
  | sort -u \
  | wc -l
)"

echo "Unique question_id count: $UNIQUE_QIDS"

if [[ "$UNIQUE_QIDS" -ne 100 ]]; then
  echo "question_id values are not unique" >&2
  exit 1
fi

###############################################################################
# 4. Fresh start or resume
###############################################################################

if [[ "$FRESH_START" == "1" ]]; then
  echo "Removing previous run: $RUN_ROOT"
  rm -rf "$RUN_ROOT"
else
  echo "Resume mode: existing completed files will be reused."
fi

mkdir -p "$OUTPUT_ROOT"

###############################################################################
# 5. Freeze experiment information
###############################################################################

mkdir -p "$RUN_ROOT"

{
  echo "date=$(date -Iseconds)"
  echo "input=$INPUT"
  echo "run_name=$RUN_NAME"
  echo "model_name=$MODEL_NAME"
  echo "openai_base_url=$OPENAI_BASE_URL"
  echo "embedding_model=$EMBEDDING_MODEL"
  echo "api_workers=$API_WORKERS"
  echo "window_workers=$WINDOW_WORKERS"
  echo "route_workers=$ROUTE_WORKERS"
  echo "tool_workers=$TOOL_WORKERS"
  echo "git_commit=$(git rev-parse HEAD)"
} > "$RUN_ROOT/p7_frozen_config.txt"

jq -r '
  if type == "array" then
    .[]
  else
    .data[]
  end
  | .question_id
' "$INPUT" \
| sort \
> "$RUN_ROOT/holdout100_question_ids.txt"

###############################################################################
# Stage 1: Prepare
###############################################################################

echo
echo "================ Stage 1: Prepare ================"

python -m longmemeval.graph_memory_v2 \
  prepare \
  --input "$INPUT" \
  --output-root "$OUTPUT_ROOT" \
  --run-name "$RUN_NAME" \
  --window-size 15 \
  --overlap 3 \
  --truncate-threshold 8000 \
  --workers "$CPU_WORKERS" \
  --fail-fast

PREPARED_COUNT="$(
  find "$RUN_ROOT" \
    -name input_item.json \
    -type f \
  | wc -l
)"

echo "Prepared cases: $PREPARED_COUNT"

if [[ "$PREPARED_COUNT" -ne 100 ]]; then
  echo "Prepare stage did not produce 100 cases" >&2
  exit 1
fi

###############################################################################
# Stage 2: V2 Memory Extraction
###############################################################################

echo
echo "================ Stage 2: V2 Extraction ================"

python -m longmemeval.graph_memory_v2 \
  --timeout 600 \
  --retries 5 \
  extract \
  --run-root "$RUN_ROOT" \
  --schema v2 \
  --max-tokens 8000 \
  --workers "$API_WORKERS" \
  --window-workers "$WINDOW_WORKERS" \
  --schema-workers 1 \
  --fail-fast

V2_COUNT="$(
  find "$RUN_ROOT" \
    -path '*/memory_v2/all_memories.json' \
    -type f \
  | wc -l
)"

echo "Completed V2 cases: $V2_COUNT"

if [[ "$V2_COUNT" -ne 100 ]]; then
  echo "V2 extraction did not complete all 100 cases" >&2
  exit 1
fi

echo "Checking empty V2 memory banks..."

EMPTY_V2="$(
  find "$RUN_ROOT" \
    -path '*/memory_v2/all_memories.json' \
    -type f \
    -exec sh -c '
      count=$(jq length "$1")
      if [ "$count" -eq 0 ]; then
        echo "$1"
      fi
    ' _ {} \;
)"

if [[ -n "$EMPTY_V2" ]]; then
  echo "The following V2 memory banks are empty:" >&2
  echo "$EMPTY_V2" >&2
  exit 1
fi

###############################################################################
# Stage 3: Build Graph
###############################################################################

echo
echo "================ Stage 3: Graph Build ================"

python -m longmemeval.graph_memory_v2 \
  build-graph \
  --run-root "$RUN_ROOT" \
  --workers "$CPU_WORKERS" \
  --fail-fast

GRAPH_COUNT="$(
  find "$RUN_ROOT" \
    -path '*/graph_v2/graph_stats.json' \
    -type f \
  | wc -l
)"

echo "Completed graphs: $GRAPH_COUNT"

if [[ "$GRAPH_COUNT" -ne 100 ]]; then
  echo "Graph build did not complete all 100 cases" >&2
  exit 1
fi

###############################################################################
# Stage 4: Query Parse
###############################################################################

echo
echo "================ Stage 4: Query Parse ================"

python -m longmemeval.graph_memory_v2 \
  --timeout 600 \
  --retries 5 \
  parse-query \
  --run-root "$RUN_ROOT" \
  --workers "$API_WORKERS" \
  --fail-fast

QUERY_COUNT="$(
  find "$RUN_ROOT" \
    -path '*/query_v2/parsed_query.json' \
    -type f \
  | wc -l
)"

echo "Completed query parses: $QUERY_COUNT"

if [[ "$QUERY_COUNT" -ne 100 ]]; then
  echo "Query parse did not complete all 100 cases" >&2
  exit 1
fi

###############################################################################
# Stage 5: P2 Gate Retrieval
###############################################################################

echo
echo "================ Stage 5: P2 Gate Retrieval ================"

# 精确复现P2：
# - 使用V3 Agent
# - 开启Evidence Completion Gate
# - 不开启all hypotheses
# - 不开启action diversity
# - 不开启coverage reranker
export DIMMEM_V3_LEGACY=0
export DIMMEM_V3_COVERAGE_GATE=1
export DIMMEM_V3_ALL_HYPOTHESES=0
export DIMMEM_V3_DIVERSE_ACTIONS=0
export DIMMEM_V3_COVERAGE_RERANK=0

# 固定内部预算，避免默认值将来变化。
export DIMMEM_V3_MAX_TOTAL_TOOL_CALLS=9
export DIMMEM_V3_ROUTER_EVIDENCE_K=18
export DIMMEM_V3_MAX_ACTIONS_PER_ROUND=3

python -m longmemeval.graph_memory_v2 \
  --timeout 600 \
  --retries 5 \
  retrieve \
  --run-root "$RUN_ROOT" \
  --mode graph_active \
  --output-name graph_active_p2_gate \
  --route-k 30 \
  --initial-k 15 \
  --final-k 18 \
  --max-rounds 4 \
  --router-mode llm \
  --embedder sentence-transformers \
  --embedding-model "$EMBEDDING_MODEL" \
  --device cpu \
  --workers "$API_WORKERS" \
  --route-workers "$ROUTE_WORKERS" \
  --tool-workers "$TOOL_WORKERS" \
  --fail-fast

P2_COUNT="$(
  find "$RUN_ROOT" \
    -path '*/retrieval_v2/graph_active_p2_gate/top_records.json' \
    -type f \
  | wc -l
)"

echo "Completed P2 retrieval cases: $P2_COUNT"

if [[ "$P2_COUNT" -ne 100 ]]; then
  echo "P2 retrieval did not complete all 100 cases" >&2
  exit 1
fi

###############################################################################
# Stage 6: P5 Compact Source QA
###############################################################################

echo
echo "================ Stage 6: P5 Source QA ================"

export QA_V4_TIMEOUT=600
export QA_V4_RETRIES=5

# 冻结开发集P5的默认参数
export QA_V4_MEMORY_K=18
export QA_V4_DIRECT_MEMORY_K=12
export QA_V4_ADJACENT_MEMORY_K=5
export QA_V4_ADJACENT_RADIUS=1
export QA_V4_SOURCE_LEXICAL_K=8
export QA_V4_MAX_SOURCE_ROWS=24
export QA_V4_MAX_TOKENS=3200
export QA_V4_VERIFY_TOKENS=2200

python -m longmemeval.graph_memory_v2.source_qa_v4 \
  --run-root "$RUN_ROOT" \
  --source-retrieval-name graph_active_p2_gate \
  --output-name graph_active_p5_source_qa \
  --qa-votes 2 \
  --workers "$API_WORKERS" \
  --fail-fast

P5_COUNT="$(
  find "$RUN_ROOT" \
    -path '*/judge_v2/graph_active_p5_source_qa/judge.json' \
    -type f \
  | wc -l
)"

echo "Completed P5 judged cases: $P5_COUNT"

if [[ "$P5_COUNT" -ne 100 ]]; then
  echo "P5 did not complete all 100 cases" >&2
  exit 1
fi

python -m longmemeval.graph_memory_v2 \
  report \
  --run-root "$RUN_ROOT" \
  --retrieval-name graph_active_p5_source_qa \
  --workers "$CPU_WORKERS" \
  --fail-fast

###############################################################################
# Stage 7: P7 Adaptive QA
###############################################################################

echo
echo "================ Stage 7: P7 Adaptive QA ================"

# 冻结26/30版本使用的P7参数
export QA_V5_EMBEDDING_MODEL="$EMBEDDING_MODEL"
export QA_V5_EMBEDDING_DEVICE=cpu

export QA_V5_MEMORY_K=18
export QA_V5_TARGET_SOURCE_K=12
export QA_V5_MAX_TARGET_ROWS=18
export QA_V5_TARGET_ADJACENT_TOP=4
export QA_V5_TARGET_ADJACENT_RADIUS=1
export QA_V5_DENSE_WEIGHT=18

export QA_V5_PLAN_TOKENS=1400
export QA_V5_SPECIALIST_TOKENS=3600
export QA_V5_ARBITER_TOKENS=900

python -m longmemeval.graph_memory_v2.adaptive_qa_v5 \
  --run-root "$RUN_ROOT" \
  --source-retrieval-name graph_active_p2_gate \
  --base-output-name graph_active_p5_source_qa \
  --output-name graph_active_p7_adaptive_qa \
  --workers "$API_WORKERS" \
  --fail-fast

P7_COUNT="$(
  find "$RUN_ROOT" \
    -path '*/judge_v2/graph_active_p7_adaptive_qa/judge.json' \
    -type f \
  | wc -l
)"

echo "Completed P7 judged cases: $P7_COUNT"

if [[ "$P7_COUNT" -ne 100 ]]; then
  echo "P7 did not complete all 100 cases" >&2
  exit 1
fi

###############################################################################
# Stage 8: Reports and paired comparison
###############################################################################

echo
echo "================ Stage 8: Reports ================"

python -m longmemeval.graph_memory_v2 \
  report \
  --run-root "$RUN_ROOT" \
  --retrieval-name graph_active_p7_adaptive_qa \
  --workers "$CPU_WORKERS" \
  --fail-fast

python -m longmemeval.graph_memory_v2 \
  compare \
  --run-root "$RUN_ROOT" \
  --baseline graph_active_p5_source_qa \
  --candidate graph_active_p7_adaptive_qa \
  --workers "$CPU_WORKERS" \
  --fail-fast

###############################################################################
# Final summary
###############################################################################

echo
echo "============================================================"
echo "P7 holdout100 experiment completed"
echo "RUN_ROOT=$RUN_ROOT"
echo
echo "P5 report:"
echo "$RUN_ROOT/reports/report_graph_active_p5_source_qa.json"
echo
echo "P7 report:"
echo "$RUN_ROOT/reports/report_graph_active_p7_adaptive_qa.json"
echo
echo "P5 vs P7 comparison:"
echo "$RUN_ROOT/reports/compare_graph_active_p5_source_qa_vs_graph_active_p7_adaptive_qa.json"
echo "============================================================"

jq '{
  retrieval_name,
  total_cases,
  correct,
  accuracy,
  by_question_type
}' \
  "$RUN_ROOT/reports/report_graph_active_p7_adaptive_qa.json"
