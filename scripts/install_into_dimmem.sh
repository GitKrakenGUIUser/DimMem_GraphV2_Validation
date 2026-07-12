#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: bash scripts/install_into_dimmem.sh /path/to/DimMem" >&2
  exit 2
fi

TARGET="$(realpath "$1")"
SOURCE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ ! -d "$TARGET/longmemeval" ]]; then
  echo "Not a DimMem checkout: missing $TARGET/longmemeval" >&2
  exit 1
fi

mkdir -p "$TARGET/longmemeval/graph_memory_v2"
cp -a "$SOURCE/longmemeval/graph_memory_v2/." "$TARGET/longmemeval/graph_memory_v2/"
printf 'Installed graph_memory_v2 into %s\n' "$TARGET/longmemeval/graph_memory_v2"
printf 'Run: cd %s && python -m longmemeval.graph_memory_v2 --help\n' "$TARGET"
