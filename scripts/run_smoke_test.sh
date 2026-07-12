#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
rm -rf "$ROOT/tests/output/smoke"
PYTHONPATH="$ROOT" python -m longmemeval.graph_memory_v2 all \
  --input "$ROOT/tests/fixtures/longmemeval_smoke.json" \
  --output-root "$ROOT/tests/output" \
  --run-name smoke \
  --smoke-test \
  --mode graph_active \
  --output-name graph_active \
  --final-k 5 \
  --max-rounds 2 \
  --workers 0 \
  --window-workers 0 \
  --force
PYTHONPATH="$ROOT" python -m unittest discover -s "$ROOT/tests" -p 'test_*.py' -v
