#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: bash scripts/run_from_scratch.sh DATA_JSON_OR_DIR OUTPUT_ROOT RUN_NAME [MAX_ITEMS]" >&2
  exit 2
fi

DATA="$1"
OUTPUT="$2"
RUN_NAME="$3"
MAX_ITEMS="${4:-0}"

: "${OPENAI_BASE_URL:?Set OPENAI_BASE_URL}"
: "${OPENAI_API_KEY:?Set OPENAI_API_KEY}"
: "${MODEL_NAME:?Set MODEL_NAME}"

python -m longmemeval.graph_memory_v2 all \
  --input "$DATA" \
  --output-root "$OUTPUT" \
  --run-name "$RUN_NAME" \
  --window-size 15 \
  --overlap 3 \
  --max-items "$MAX_ITEMS" \
  --mode graph_active \
  --output-name graph_active \
  --route-k 20 \
  --initial-k 12 \
  --final-k 15 \
  --max-rounds 3 \
  --router-mode llm \
  --embedder sentence-transformers \
  --embedding-model sentence-transformers/all-MiniLM-L6-v2
