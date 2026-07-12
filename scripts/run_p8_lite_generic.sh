#!/usr/bin/env bash
set -euo pipefail

: "${RUN_ROOT:?Set RUN_ROOT}"
: "${QA_V5_EMBEDDING_MODEL:?Set QA_V5_EMBEDDING_MODEL}"

RETRIEVAL_NAME="${RETRIEVAL_NAME:-graph_active_p2_gate}"
P5_NAME="${P5_NAME:-graph_active_p5_source_qa}"
P7_NAME="${P7_NAME:-graph_active_p7_adaptive_qa}"
OUTPUT_NAME="${OUTPUT_NAME:-graph_active_p8_lite_generic}"
WORKERS="${WORKERS:-100}"

python -m longmemeval.graph_memory_v2.adaptive_qa_v6_lite \
  --run-root "$RUN_ROOT" \
  --retrieval-name "$RETRIEVAL_NAME" \
  --p5-name "$P5_NAME" \
  --p7-name "$P7_NAME" \
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
  --baseline "$P7_NAME" \
  --candidate "$OUTPUT_NAME" \
  --workers "$WORKERS" \
  --fail-fast

python -m longmemeval.graph_memory_v2 compare \
  --run-root "$RUN_ROOT" \
  --baseline "$P5_NAME" \
  --candidate "$OUTPUT_NAME" \
  --workers "$WORKERS" \
  --fail-fast
