# Adaptive QA V5

V5 以 P5 的 compact source QA 为默认结果，只对风险问题升级到 Specialist。
它不允许“更多上下文导致更保守”破坏一个已经由证据和确定性计算支持的答案。

## 核心设计

1. **Compact-first**：复用 `graph_active_p5_source_qa`。
2. **Risk gate**：Abstention、未填槽位、count/sum/timeline/current-state/
   preference/assistant-recall 才进入 Specialist。
3. **Answer contract planner**：输出通用操作类型和 evidence slots。
4. **Targeted source retrieval**：按 slot 同时执行 lexical 和本地 dense 检索。
5. **Specialist reasoning**：允许有证据的组合推理、语义桥接和状态投影。
6. **Monotonic candidate merge**：
   - Base 有支持而 Specialist 只变成 abstention：保留 Base；
   - Base abstain、Specialist 有支持：采用 Specialist；
   - 有确定性计算的一方优先；
   - Preference 和 Current State 使用专门 contract；
   - 其余冲突交给 Arbiter。
7. **不再重复相同 Prompt 做 QA votes**。候选差异来自 compact 与 targeted
   两种 evidence view。

## 安装

```bash
unzip DimMem_AdaptiveQA_V5.zip -d /tmp/DimMem_AdaptiveQA_V5

python /tmp/DimMem_AdaptiveQA_V5/install_adaptive_qa_v5.py \
  --repo ~/projects/DimMem_GraphV2_Validation

cd ~/projects/DimMem_GraphV2_Validation
pytest -q tests/test_adaptive_qa_v5.py
```

## 运行

```bash
cd ~/projects/DimMem_GraphV2_Validation

export RUN_ROOT=./results/graph_memory_v2/random30_parallel_01
export SOURCE_RETRIEVAL_NAME=graph_active_p2_gate
export BASE_OUTPUT_NAME=graph_active_p5_source_qa
export OUTPUT_NAME=graph_active_p7_adaptive_qa

export QA_V5_EMBEDDING_MODEL=/home/bowenzheng/huggingface/hub/all-MiniLM-L6-v2
export QA_V5_EMBEDDING_DEVICE=cpu
export WORKERS=8

bash scripts/run_adaptive_qa_v5.sh
```

## 推荐默认参数

```bash
export QA_V5_MEMORY_K=18
export QA_V5_TARGET_SOURCE_K=12
export QA_V5_MAX_TARGET_ROWS=18
export QA_V5_TARGET_ADJACENT_TOP=4
export QA_V5_TARGET_ADJACENT_RADIUS=1
export QA_V5_DENSE_WEIGHT=18
export QA_V5_PLAN_TOKENS=1400
export QA_V5_SPECIALIST_TOKENS=3600
export QA_V5_ARBITER_TOKENS=900
```

这组参数不会像 P6 那样把每个 Case 都扩展到 30 多条 source。只有风险 Case 才增加
targeted evidence。

## 预期重点观察

```text
adaptive_choice
risk_reasons
plan.answer_contract
targeted_source_ids
base_candidate
specialist_candidate
choice_reason
```

这些字段保存在：

```text
qa_v2/graph_active_p7_adaptive_qa/adaptive_trace.json
```
