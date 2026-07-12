# P8 Lite Generic

P8 Lite directly reuses the existing 100-case artifacts:

```text
memory_v2/all_memories.json
graph_v2/
query_v2/
retrieval_v2/graph_active_p2_gate/
qa_v2/graph_active_p5_source_qa/
qa_v2/graph_active_p7_adaptive_qa/
source_turns.json
```

It does **not** re-run memory extraction, graph construction, query parsing or
P2 retrieval.

## Why this version

The current 100-case report has 98.6% mean gold-session recall, while P7 only
adds one net correct case over P5. The main targets are therefore:

1. answer-contract mistakes;
2. incomplete event-set retrieval for count/sum/timeline;
3. unsafe deterministic calculation;
4. current-state/preference reasoning;
5. P5/P7 candidate selection without evidence verification.

## Token controls

P8 escalates only risky cases. Simple P5/P7 agreements stay unchanged.

Default limits:

```bash
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
```

## Installation

```bash
unzip DimMem_P8_Lite_Generic.zip -d /tmp/DimMem_P8_Lite_Generic

python /tmp/DimMem_P8_Lite_Generic/install_p8_lite.py \
  --repo ~/projects/DimMem_GraphV2_Validation

cd ~/projects/DimMem_GraphV2_Validation
pytest -q tests/test_adaptive_qa_v6_lite.py
```

## Current 100-case run

```bash
cd ~/projects/DimMem_GraphV2_Validation

export RUN_ROOT=./results/graph_memory_v2/random30_parallel_01
export QA_V5_EMBEDDING_MODEL=/home/bowenzheng/huggingface/hub/all-MiniLM-L6-v2
export QA_V5_EMBEDDING_DEVICE=cpu
export WORKERS=100

bash scripts/run_p8_lite_generic.sh
```

The runner selects cases using `holdout100_question_ids.txt`, not the stale
30-case `run_manifest.json`.

## Outputs

```text
qa_v2/graph_active_p8_lite_generic/
judge_v2/graph_active_p8_lite_generic/
retrieval_v2/graph_active_p8_lite_generic/
reports/report_graph_active_p8_lite_generic.json
reports/compare_graph_active_p7_adaptive_qa_vs_graph_active_p8_lite_generic.json
```
