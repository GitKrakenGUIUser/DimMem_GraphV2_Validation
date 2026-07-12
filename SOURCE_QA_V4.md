# Source-backed QA V4

该版本不重新提取 Memory、不重新建图，也不重新运行 Active Retrieval。它从现有
`graph_active_p2_gate` 结果出发，重点修复消融结果暴露出的三个问题：

1. **结构化 Memory 已命中相关 Session，但答案细节在抽取时丢失。**
   V4 根据 `provenance.source_ids` 恢复原始 user turn，并对高排名 Memory 加入同
   Session 相邻 turn。
2. **Assistant-dependent 问题缺少原始 Assistant Reply。**
   V4 对原始 `source_turns.json` 增加查询相关的 lexical source route，并在需要时
   带回 `assistant_reply`。
3. **QA 的 reasoning 与最终 answer 不一致。**
   V4 对日期差、时间加法、求和和 distinct count 使用确定性执行器；运行两次 QA
   生成并选择证据支持更强的结果，再执行一次 verifier。

Judge 同时修复了以下等价关系：

```text
Gold: The information provided is not enough.
Prediction: Cannot be determined from the conversation.
```

两者应判为正确，而不是把 Gold 当成“具体答案”。

## 安装

```bash
unzip DimMem_SourceQA_V4.zip -d /tmp/DimMem_SourceQA_V4

python /tmp/DimMem_SourceQA_V4/install_source_qa_v4.py \
  --repo ~/projects/DimMem_GraphV2_Validation

cd ~/projects/DimMem_GraphV2_Validation
pytest -q tests/test_source_qa_v4.py
```

## 推荐运行

```bash
cd ~/projects/DimMem_GraphV2_Validation

export RUN_ROOT=./results/graph_memory_v2/random30_parallel_01
export SOURCE_RETRIEVAL_NAME=graph_active_p2_gate
export OUTPUT_NAME=graph_active_p5_source_qa
export WORKERS=8
export QA_VOTES=2

bash scripts/run_source_qa_v4.sh
```

这会复用 P2 Retrieval，生成新的：

```text
retrieval_v2/graph_active_p5_source_qa/
qa_v2/graph_active_p5_source_qa/
judge_v2/graph_active_p5_source_qa/
reports/report_graph_active_p5_source_qa.json
```

## 建议参数

默认设置优先控制 Token：

```bash
export QA_V4_MEMORY_K=18
export QA_V4_DIRECT_MEMORY_K=12
export QA_V4_ADJACENT_MEMORY_K=5
export QA_V4_ADJACENT_RADIUS=1
export QA_V4_SOURCE_LEXICAL_K=8
export QA_V4_MAX_SOURCE_ROWS=24
export QA_V4_MAX_TOKENS=3200
export QA_V4_VERIFY_TOKENS=2200
```

更高 Token 版本：

```bash
export QA_V4_MEMORY_K=22
export QA_V4_DIRECT_MEMORY_K=16
export QA_V4_ADJACENT_MEMORY_K=8
export QA_V4_SOURCE_LEXICAL_K=12
export QA_V4_MAX_SOURCE_ROWS=32
export QA_VOTES=3
```

## 研究上如何解释

P5 不改变 Graph 或 Active Router，仅增加一个 source-backed answer layer，因此可回答：

> 当前上限受限于主动检索，还是受限于结构化 Memory 的信息损失与 QA 执行错误？

若 P5 显著提高，下一阶段应将 source route 正式前移到 Active Retrieval，而不是继续
盲目增加 graph rounds。
