#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# 0. 仓库、数据和实验目录
###############################################################################

REPO_ROOT="${REPO_ROOT:-$HOME/projects/DimMem_GraphV2_Validation}"
cd "$REPO_ROOT"

# 必须在启动时传入，或在这里改成新100题JSON的实际路径。
INPUT="${INPUT:?请设置 INPUT 为新的100 Case JSON路径}"

OUTPUT_ROOT="${OUTPUT_ROOT:-./results/graph_memory_v2}"
RUN_NAME="${RUN_NAME:-holdout100_p8_frozen_02}"
RUN_ROOT="${OUTPUT_ROOT}/${RUN_NAME}"

EMBEDDING_MODEL="${EMBEDDING_MODEL:-/home/bowenzheng/huggingface/hub/all-MiniLM-L6-v2}"

# 首次运行：
#   FRESH_START=1 bash scripts/run_p8_holdout100_from_scratch.sh
#
# 中断后续跑：
#   bash scripts/run_p8_holdout100_from_scratch.sh
FRESH_START="${FRESH_START:-0}"

###############################################################################
# 1. 并发设置
###############################################################################

# 纯API或Case级任务，可以100 Case并发。
API_WORKERS="${API_WORKERS:-100}"

# 本地建图、报告等CPU/磁盘阶段。
CPU_WORKERS="${CPU_WORKERS:-32}"

# Extraction每个Case内部窗口顺序处理，避免100×窗口数瞬间展开。
WINDOW_WORKERS="${WINDOW_WORKERS:-1}"

# P2每个Case内部Route/Tool并发。
# 当前仓库冻结实验脚本使用3。
ROUTE_WORKERS="${ROUTE_WORKERS:-3}"
TOOL_WORKERS="${TOOL_WORKERS:-3}"

# P7/P8使用本地SentenceTransformer。
# 不建议100线程同时竞争同一个本地模型锁。
P7_WORKERS="${P7_WORKERS:-16}"
P8_WORKERS="${P8_WORKERS:-8}"

###############################################################################
# 2. 基础检查
###############################################################################

if [[ ! -f "$INPUT" ]]; then
  echo "找不到输入文件：$INPUT" >&2
  exit 1
fi

if [[ -z "${OPENAI_BASE_URL:-}" ]]; then
  echo "缺少 OPENAI_BASE_URL" >&2
  exit 1
fi

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "缺少 OPENAI_API_KEY" >&2
  exit 1
fi

if [[ -z "${MODEL_NAME:-}" ]]; then
  echo "缺少 MODEL_NAME" >&2
  exit 1
fi

if [[ ! -d "$EMBEDDING_MODEL" ]]; then
  echo "找不到本地Embedding模型：$EMBEDDING_MODEL" >&2
  exit 1
fi

# 确认P5/P7/P8模块已存在。
python - <<'PY'
import importlib

modules = [
    "longmemeval.graph_memory_v2.source_qa_v4",
    "longmemeval.graph_memory_v2.adaptive_qa_v5",
    "longmemeval.graph_memory_v2.adaptive_qa_v6_lite",
]

for name in modules:
    importlib.import_module(name)
    print(f"[module-ok] {name}")
PY

###############################################################################
# 3. 验证数据集
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

UNIQUE_QIDS="$(
  jq -r '
    if type == "array" then
      .[]
    elif type == "object" and (.data | type) == "array" then
      .data[]
    else
      empty
    end
    | .question_id
  ' "$INPUT" \
  | sort -u \
  | wc -l \
  | tr -d " "
)"

echo "Case数量：$CASE_COUNT"
echo "唯一question_id数量：$UNIQUE_QIDS"

if [[ "$CASE_COUNT" -ne 100 ]]; then
  echo "要求恰好100个Case，实际为：$CASE_COUNT" >&2
  exit 1
fi

if [[ "$UNIQUE_QIDS" -ne 100 ]]; then
  echo "question_id不唯一，唯一数量：$UNIQUE_QIDS" >&2
  exit 1
fi

###############################################################################
# 4. 全新运行或断点续跑
###############################################################################

if [[ "$FRESH_START" == "1" ]]; then
  echo "删除已有运行目录：$RUN_ROOT"
  rm -rf "$RUN_ROOT"
else
  echo "断点续跑模式：已有结果将被复用"
fi

mkdir -p "$RUN_ROOT"

###############################################################################
# 5. 冻结实验配置
###############################################################################

{
  echo "date=$(date -Iseconds)"
  echo "input=$(realpath "$INPUT")"
  echo "run_name=$RUN_NAME"
  echo "run_root=$RUN_ROOT"
  echo "git_commit=$(git rev-parse HEAD)"
  echo "model_name=$MODEL_NAME"
  echo "openai_base_url=$OPENAI_BASE_URL"
  echo "embedding_model=$EMBEDDING_MODEL"
  echo "api_workers=$API_WORKERS"
  echo "cpu_workers=$CPU_WORKERS"
  echo "window_workers=$WINDOW_WORKERS"
  echo "route_workers=$ROUTE_WORKERS"
  echo "tool_workers=$TOOL_WORKERS"
  echo "p7_workers=$P7_WORKERS"
  echo "p8_workers=$P8_WORKERS"
} > "$RUN_ROOT/p8_frozen_config.txt"

# P8的discover_samples会优先使用这个文件筛选本次100题。
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
# Stage 1：Prepare
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
  | wc -l \
  | tr -d " "
)"

echo "Prepare完成：$PREPARED_COUNT/100"

if [[ "$PREPARED_COUNT" -ne 100 ]]; then
  echo "Prepare阶段没有产生100个Case" >&2
  exit 1
fi

###############################################################################
# Stage 2：只提取V2 Memory
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
  | wc -l \
  | tr -d " "
)"

echo "V2 Extraction完成：$V2_COUNT/100"

if [[ "$V2_COUNT" -ne 100 ]]; then
  echo "V2 Extraction没有完成全部100题" >&2
  exit 1
fi

EMPTY_V2="$(
  find "$RUN_ROOT" \
    -path '*/memory_v2/all_memories.json' \
    -type f \
    -exec sh -c '
      jq -e "type == \"array\" and length > 0" "$1" >/dev/null ||
      echo "$1"
    ' _ {} \;
)"

if [[ -n "$EMPTY_V2" ]]; then
  echo "发现空的V2 Memory Bank：" >&2
  echo "$EMPTY_V2" >&2
  exit 1
fi

###############################################################################
# Stage 3：构建V2 Graph
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
  | wc -l \
  | tr -d " "
)"

echo "Graph Build完成：$GRAPH_COUNT/100"

if [[ "$GRAPH_COUNT" -ne 100 ]]; then
  echo "Graph Build没有完成全部100题" >&2
  exit 1
fi

###############################################################################
# Stage 4：Query Parse
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
  | wc -l \
  | tr -d " "
)"

echo "Query Parse完成：$QUERY_COUNT/100"

if [[ "$QUERY_COUNT" -ne 100 ]]; then
  echo "Query Parse没有完成全部100题" >&2
  exit 1
fi

###############################################################################
# Stage 5：P2 Gate Retrieval
###############################################################################

echo
echo "================ Stage 5: P2 Gate Retrieval ================"

# 冻结P2策略。
export DIMMEM_V3_LEGACY=0
export DIMMEM_V3_COVERAGE_GATE=1
export DIMMEM_V3_ALL_HYPOTHESES=0
export DIMMEM_V3_DIVERSE_ACTIONS=0
export DIMMEM_V3_COVERAGE_RERANK=0

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
  | wc -l \
  | tr -d " "
)"

echo "P2 Retrieval完成：$P2_COUNT/100"

if [[ "$P2_COUNT" -ne 100 ]]; then
  echo "P2 Retrieval没有完成全部100题" >&2
  exit 1
fi

###############################################################################
# Stage 6：P5 Compact Source QA
###############################################################################

echo
echo "================ Stage 6: P5 Source QA ================"

export QA_V4_TIMEOUT=600
export QA_V4_RETRIES=5

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
  | wc -l \
  | tr -d " "
)"

echo "P5完成：$P5_COUNT/100"

if [[ "$P5_COUNT" -ne 100 ]]; then
  echo "P5没有完成全部100题" >&2
  exit 1
fi

python -m longmemeval.graph_memory_v2 \
  report \
  --run-root "$RUN_ROOT" \
  --retrieval-name graph_active_p5_source_qa \
  --workers "$CPU_WORKERS" \
  --fail-fast

###############################################################################
# Stage 7：P7 Adaptive QA
###############################################################################

echo
echo "================ Stage 7: P7 Adaptive QA ================"

export QA_V5_EMBEDDING_MODEL="$EMBEDDING_MODEL"
export QA_V5_EMBEDDING_DEVICE=cpu
export QA_V5_EMBED_BATCH=64
export QA_V5_DENSE_CACHE_ENTRIES=256

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
  --workers "$P7_WORKERS" \
  --fail-fast

P7_COUNT="$(
  find "$RUN_ROOT" \
    -path '*/judge_v2/graph_active_p7_adaptive_qa/judge.json' \
    -type f \
  | wc -l \
  | tr -d " "
)"

echo "P7完成：$P7_COUNT/100"

if [[ "$P7_COUNT" -ne 100 ]]; then
  echo "P7没有完成全部100题" >&2
  exit 1
fi

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
# Stage 8：P8 Lite Generic
###############################################################################

echo
echo "================ Stage 8: P8 Lite Generic ================"

export QA_P8_MEMORY_K=12
export QA_P8_COMPACT_K=10

export QA_P8_PER_SLOT_K=3
export QA_P8_GLOBAL_K=6
export QA_P8_MAX_TARGET_ROWS=14
export QA_P8_SESSION_CAP=4

export QA_P8_ADJACENT_TOP=3
export QA_P8_ADJACENT_RADIUS=1

export QA_P8_SOURCE_CHARS=1100
export QA_P8_ASSISTANT_CHARS=1400
export QA_P8_V2_TIMELINE_K=8

export QA_P8_PLAN_TOKENS=900
export QA_P8_SPECIALIST_TOKENS=2200
export QA_P8_VERIFY_TOKENS=900

python -m longmemeval.graph_memory_v2.adaptive_qa_v6_lite \
  --run-root "$RUN_ROOT" \
  --retrieval-name graph_active_p2_gate \
  --p5-name graph_active_p5_source_qa \
  --p7-name graph_active_p7_adaptive_qa \
  --output-name graph_active_p8_lite_generic \
  --workers "$P8_WORKERS" \
  --fail-fast

P8_COUNT="$(
  find "$RUN_ROOT" \
    -path '*/judge_v2/graph_active_p8_lite_generic/judge.json' \
    -type f \
  | wc -l \
  | tr -d " "
)"

echo "P8完成：$P8_COUNT/100"

if [[ "$P8_COUNT" -ne 100 ]]; then
  echo "P8没有完成全部100题" >&2
  exit 1
fi

###############################################################################
# Stage 9：P8报告和配对比较
###############################################################################

echo
echo "================ Stage 9: Reports ================"

python -m longmemeval.graph_memory_v2 \
  report \
  --run-root "$RUN_ROOT" \
  --retrieval-name graph_active_p8_lite_generic \
  --workers "$CPU_WORKERS" \
  --fail-fast

python -m longmemeval.graph_memory_v2 \
  compare \
  --run-root "$RUN_ROOT" \
  --baseline graph_active_p7_adaptive_qa \
  --candidate graph_active_p8_lite_generic \
  --workers "$CPU_WORKERS" \
  --fail-fast

python -m longmemeval.graph_memory_v2 \
  compare \
  --run-root "$RUN_ROOT" \
  --baseline graph_active_p5_source_qa \
  --candidate graph_active_p8_lite_generic \
  --workers "$CPU_WORKERS" \
  --fail-fast

###############################################################################
# Final Summary
###############################################################################

echo
echo "============================================================"
echo "P8从零运行完成"
echo "RUN_ROOT=$RUN_ROOT"
echo
echo "P5 Report:"
echo "$RUN_ROOT/reports/report_graph_active_p5_source_qa.json"
echo
echo "P7 Report:"
echo "$RUN_ROOT/reports/report_graph_active_p7_adaptive_qa.json"
echo
echo "P8 Report:"
echo "$RUN_ROOT/reports/report_graph_active_p8_lite_generic.json"
echo
echo "P7 vs P8:"
echo "$RUN_ROOT/reports/compare_graph_active_p7_adaptive_qa_vs_graph_active_p8_lite_generic.json"
echo "============================================================"

jq '{
  retrieval_name,
  n: (.n // .total_cases),
  correct,
  accuracy,
  by_question_type,
  token_totals
}' \
  "$RUN_ROOT/reports/report_graph_active_p8_lite_generic.json"
