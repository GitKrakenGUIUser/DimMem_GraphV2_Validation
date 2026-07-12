# P9 Operation Census

P9 is a QA-only layer built on the existing P8 result. It reuses:

```text
memory_v2/all_memories.json
source_turns.json
retrieval_v2/graph_active_p2_gate/
qa_v2/graph_active_p5_source_qa/
qa_v2/graph_active_p7_adaptive_qa/
qa_v2/graph_active_p8_lite_generic/
```

It does not rerun V2 extraction, graph construction, query parsing, or P2
retrieval.

## Main changes

### 1. Operation contracts

P9 adds or strengthens:

```text
duration
argmax_or_argmin
habitual_state
direct_relation
enumerate_count
timeline
comparison
preference_criteria
```

Duration detection runs before generic `how many`, and superlative and habitual
questions are no longer treated as ordinary direct facts.

### 2. Full-bank V2 evidence census

For audited questions, P9 scans the complete existing V2 memory bank using
lexical+dense RRF. This is used especially for count, timeline, sum, argmax,
state, and preference questions.

### 3. Set completeness

The specialist receives both P2 top memories and a full-bank V2 census. It must
construct atomic facts and report possible omissions before counting, summing,
or ordering.

### 4. Preference profile contract

Recommendation questions require stable positive preferences. Negative
preferences and relevant past experiences are optional. Unknown location,
current mood, and live availability do not force abstention unless explicitly
requested.

### 5. Deterministic finalizer

P9 computes or validates:

```text
count_distinct
sum_by_group
argmax / argmin
compare_dates
compare_numbers
timeline
duration
```

The final answer is produced from labeled operands, preventing a reasoning
trace from saying one event is earlier while the answer names the other event.

### 6. Duration-equivalent judge guard

Equivalent durations such as `7 days` and `one week` are accepted
deterministically before the LLM judge.

## Install

```bash
unzip DimMem_P9_OperationCensus.zip -d /tmp/DimMem_P9_OperationCensus

python /tmp/DimMem_P9_OperationCensus/install_p9.py \
  --repo ~/projects/DimMem_GraphV2_Validation

cd ~/projects/DimMem_GraphV2_Validation
pytest -q tests/test_adaptive_qa_v7_operation.py
```

## Run on the existing 100 cases

```bash
cd ~/projects/DimMem_GraphV2_Validation

export RUN_ROOT=./results/graph_memory_v2/random30_parallel_01

export RETRIEVAL_NAME=graph_active_p2_gate
export P5_NAME=graph_active_p5_source_qa
export P7_NAME=graph_active_p7_adaptive_qa
export P8_NAME=graph_active_p8_lite_generic
export OUTPUT_NAME=graph_active_p9_operation_census

export QA_V5_EMBEDDING_MODEL=/home/bowenzheng/huggingface/hub/all-MiniLM-L6-v2
export QA_V5_EMBEDDING_DEVICE=cpu
export WORKERS=100

bash scripts/run_p9_operation_census.sh
```

## Token-controlled defaults

```bash
export QA_P9_MEMORY_K=12
export QA_P9_COMPACT_K=8

export QA_P9_MAX_QUERIES=6
export QA_P9_V2_PER_QUERY_K=5
export QA_P9_V2_GLOBAL_K=18
export QA_P9_V2_CENSUS_K=22
export QA_P9_V2_CONTENT_CHARS=700
export QA_P9_PROFILE_K=12

export QA_P9_EXPAND_TOKENS=650
export QA_P9_SPECIALIST_TOKENS=1900
export QA_P9_ADJUDICATE_TOKENS=1050
```

Only audited questions call the P9 LLM stages. Non-risk direct cases reuse P8
with zero new QA prompt tokens.

## Outputs

```text
qa_v2/graph_active_p9_operation_census/
judge_v2/graph_active_p9_operation_census/
retrieval_v2/graph_active_p9_operation_census/
reports/report_graph_active_p9_operation_census.json
reports/compare_graph_active_p8_lite_generic_vs_graph_active_p9_operation_census.json
```

Each audited case also saves:

```text
p9_trace.json
v2_evidence_census.json
source_evidence_census.json
state_preference_view.json
```
