# Active Policy V3

这版改动针对当前仓库中 Graph Active 实际工具调用偏少、容易第一轮提前
`finish` 的问题。它没有根据 LongMemEval 的 category 或当前30个 case
内容编写 `if/elif`，所有控制信号都来自 Query Parser 已经生成的通用字段：

- `need_assistant_context`
- `need_multi_hop`
- `expected_evidence_count`
- `missing_slots`
- `state_keys`
- `answer_dim`
- 最多三个 query hypotheses

## 代码改动

### 1. `active_agent_v3.py`

新增：

- Evidence completion gate；
- 所有 query hypotheses 对 Router 可见；
- 同一轮动作方向多样化；
- `state_key` 强制检查更新链；
- Assistant-dependent query 强制检查 source reply；
- 全局 tool-call budget；
- coverage-aware + MMR 最终排序；
- 详细的 `coverage_before/coverage_after/final_coverage` trace；
- 通过环境变量执行逐项消融。

原始 `active_agent.py` 保留不动。设置：

```bash
export DIMMEM_V3_LEGACY=1
```

即可复用原始 Agent，生成严格的旧策略对照。

### 2. `prompts_v3.py`

Router Prompt 不再只看到 primary hypothesis，而是可以看到全部 hypotheses 和
代码计算出的 coverage state。

QA Prompt 采用通用的 slot-binding 结构，显式处理：

- 多部分答案；
- 当前状态与历史状态；
- 更新链冲突；
- 日期差、求和、计数；
- Preference scope；
- 证据不足时再 abstain。

### 3. `evaluation.py`

修改为：

- 使用 Active Policy V3；
- Assistant context 判断覆盖所有 hypotheses；
- QA 输入包含 parsed query；
- QA token budget 从1600提高到2600；
- 保存 `required_slots/unfilled_slots/conflicts/operation/answerable`；
- Judge 增加确定性的 abstention guard。

### 4. 默认轮数

`retrieve/all/ablation` 的 `max_rounds` 默认值从3改为4。

## 应用

在仓库根目录：

```bash
python /path/to/apply_active_policy_v3.py --repo .
pytest -q tests/test_active_policy_v3.py
```

## 推荐实验

不要重新抽取 Memory，也不需要重新建图。复用已有：

```text
memory_v1/
memory_v2/
graph_v2/
query_v2/
```

设置：

```bash
export RUN_ROOT=./results/graph_memory_v2/random30_parallel_01
export EMBEDDING_MODEL=/home/bowenzheng/huggingface/hub/all-MiniLM-L6-v2
bash scripts/run_active_policy_ablation.sh
```

## 环境变量

| 变量 | 默认值 | 作用 |
|---|---:|---|
| `DIMMEM_V3_LEGACY` | `0` | 使用原始 Active Agent |
| `DIMMEM_V3_COVERAGE_GATE` | `1` | 阻止结构性证据未检查时提前结束 |
| `DIMMEM_V3_ALL_HYPOTHESES` | `1` | Router 看到全部 query hypotheses |
| `DIMMEM_V3_DIVERSE_ACTIONS` | `1` | 每轮优先不同检索方向 |
| `DIMMEM_V3_COVERAGE_RERANK` | `1` | 最终结构化/MMR rerank |
| `DIMMEM_V3_MAX_TOTAL_TOOL_CALLS` | `9` | 每个 case 的总工具预算 |
| `DIMMEM_V3_ROUTER_EVIDENCE_K` | `18` | Router 每轮可见证据条数 |
| `DIMMEM_V3_MAX_ACTIONS_PER_ROUND` | `3` | 每轮最大工具数 |

## 预期观察指标

除了 Accuracy，应检查报告和 trace 中：

```text
mean_active_rounds
mean_tool_calls
finish_overridden
new_memory_count
final_coverage.hard_gaps
fixed/broken question IDs
```

合理目标不是无约束增加工具调用，而是把平均 tool calls 从接近0提升到：
`2-6` 左右，并让 Multi-hop、Knowledge Update、Assistant-dependent Case 获得
更高 evidence completeness。
