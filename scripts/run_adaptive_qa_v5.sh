#!/usr/bin/env bash
set -euo pipefail

: "${RUN_ROOT:?Set RUN_ROOT}"
: "${QA_V5_EMBEDDING_MODEL:?Set QA_V5_EMBEDDING_MODEL}"

SOURCE_RETRIEVAL_NAME="${SOURCE_RETRIEVAL_NAME:-graph_active_p2_gate}"
BASE_OUTPUT_NAME="${BASE_OUTPUT_NAME:-graph_active_p5_source_qa}"
OUTPUT_NAME="${OUTPUT_NAME:-graph_active_p7_adaptive_qa}"
WORKERS="${WORKERS:-8}"

python -m longmemeval.graph_memory_v2.adaptive_qa_v5 \
  --run-root "$RUN_ROOT" \
  --source-retrieval-name "$SOURCE_RETRIEVAL_NAME" \
  --base-output-name "$BASE_OUTPUT_NAME" \
  --output-name "$OUTPUT_NAME" \
  --workers "$WORKERS" \
  --force \
  --fail-fast

python -m longmemeval.graph_memory_v2 report \
  --run-root "$RUN_ROOT" \
  --retrieval-name "$OUTPUT_NAME" \
  --workers "$WORKERS" \
  --fail-fast

python -m longmemeval.graph_memory_v2 compare \
  --run-root "$RUN_ROOT" \
  --baseline "$BASE_OUTPUT_NAME" \
  --candidate "$OUTPUT_NAME" \
  --workers "$WORKERS" \
  --fail-fast
