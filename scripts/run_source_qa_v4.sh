#!/usr/bin/env bash
set -euo pipefail

: "${RUN_ROOT:?Set RUN_ROOT}"
SOURCE_RETRIEVAL_NAME="${SOURCE_RETRIEVAL_NAME:-graph_active_p2_gate}"
OUTPUT_NAME="${OUTPUT_NAME:-graph_active_p5_source_qa}"
WORKERS="${WORKERS:-8}"
QA_VOTES="${QA_VOTES:-2}"

python -m longmemeval.graph_memory_v2.source_qa_v4 \
  --run-root "$RUN_ROOT" \
  --source-retrieval-name "$SOURCE_RETRIEVAL_NAME" \
  --output-name "$OUTPUT_NAME" \
  --qa-votes "$QA_VOTES" \
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
  --baseline "$SOURCE_RETRIEVAL_NAME" \
  --candidate "$OUTPUT_NAME" \
  --workers "$WORKERS" \
  --fail-fast
