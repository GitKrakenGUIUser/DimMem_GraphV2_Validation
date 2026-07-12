#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: bash scripts/run_ablation.sh /path/to/output/RUN_NAME" >&2
  exit 2
fi

RUN_ROOT="$1"
: "${OPENAI_BASE_URL:?Set OPENAI_BASE_URL}"
: "${OPENAI_API_KEY:?Set OPENAI_API_KEY}"
: "${MODEL_NAME:?Set MODEL_NAME}"

CASE_WORKERS="${CASE_WORKERS:-0}"
# 0 runs dimmem_v1, dimension_v2, graph_static, and graph_active concurrently.
VARIANT_WORKERS="${VARIANT_WORKERS:-0}"
ROUTE_WORKERS="${ROUTE_WORKERS:-0}"
TOOL_WORKERS="${TOOL_WORKERS:-0}"

python -m longmemeval.graph_memory_v2 ablation \
  --run-root "$RUN_ROOT" \
  --route-k 20 \
  --initial-k 12 \
  --final-k 15 \
  --max-rounds 3 \
  --router-mode llm \
  --embedder sentence-transformers \
  --embedding-model sentence-transformers/all-MiniLM-L6-v2 \
  --workers "$CASE_WORKERS" \
  --variant-workers "$VARIANT_WORKERS" \
  --route-workers "$ROUTE_WORKERS" \
  --tool-workers "$TOOL_WORKERS"
