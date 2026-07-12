# DimMem Graph Memory V2：LongMemEval 从零建图验证包

这是一套以 **DimMem 原始 LongMemEval 流程和数据接口**为起点的实验扩展。它不依赖之前生成的 DimMem retrieval 结果，也不读取 MRAgent 已建好的图，而是从 LongMemEval 原始 JSON 开始依次完成：

```text
LongMemEval 原始数据
  → overlap-aware 会话切窗
  → V1/V2 两套记忆抽取
  → V2 维度图从零构建
  → 查询维度解析
  → 静态/图扩展/主动图遍历
  → QA
  → Judge
  → 分类报告与成对显著性分析
```

代码以新增目录 `longmemeval/graph_memory_v2/` 的方式接入原始 DimMem 仓库，不覆盖其现有模块，便于保留原始实现并做公平对照。

---

## 1. 这版到底优化了什么

原始 DimMem 记忆记录的主要操作维度是：

```text
memory_type / time / location / reason / purpose / keywords
```

V2 保留这六类接口，同时补充 LongMemEval 中最容易丢失的结构：

| 新结构 | 解决的问题 |
|---|---|
| `event_start/event_end` | 事件什么时候发生 |
| `valid_from/valid_to` | 某个事实在哪段时间有效 |
| `entities(name/type/role)` | 跨会话实体归一和角色区分 |
| `topic` | 相同主题的跨会话定位 |
| `relation(subject/predicate/object)` | 关系型问答和事件组合 |
| `state_key/state_value/state_status` | Knowledge Update 的新旧事实版本链 |
| `preference(target/polarity/strength/scope)` | 防止把一次体验过度泛化为长期偏好 |
| `modality` | 区分事实、计划、否定、假设和不确定陈述 |
| `confidence/evidence_span/provenance` | 来源审计、错误定位与按需恢复 Assistant 回复 |

最关键的变化不是“增加 metadata”，而是让这些字段成为建图、检索、更新链和主动遍历共同使用的操作接口。

### 事件时间与事实有效时间分离

例如：

```text
The user moved to Seattle on March 10.
```

V2 会尽量表示为：

```json
{
  "time": {
    "event_start": "2025-03-10",
    "valid_from": "2025-03-10"
  },
  "state_key": "user.residence",
  "state_value": "Seattle"
}
```

这样既能回答“什么时候搬家”，也能回答“现在住在哪里”。

### Knowledge Update 不再简单覆盖旧记忆

若两条记录拥有相同 `state_key`，图构建器按时间建立：

```text
new memory --SUPERSEDES--> old memory
```

若值相同，则建立：

```text
new memory --SUPPORTS--> old memory
```

因此当前事实、历史事实和更新发生时间都能保留。

---

## 2. 从零建出的图包含什么

每个 LongMemEval case 单独建图，避免 case 间信息泄漏。

### 节点

```text
Memory
Entity
Time
Location
Topic
Session
State
Preference
RelationEntity
```

### 主要边

```text
ABOUT_ENTITY
EVENT_START / EVENT_END
VALID_FROM / VALID_TO
AT_LOCATION
HAS_TOPIC
IN_SESSION
ASSERTS_STATE
PREFERENCE_ABOUT
RELATION_SUBJECT / RELATION_OBJECT / RELATION
NEXT_IN_SESSION / BEFORE
SUPERSEDES / SUPPORTS
SAME_EVENT
```

底层使用 JSON 图和倒排索引，不要求 Neo4j。生成结果全部可读、可检查：

```text
graph_v2/
├── memory_bank.json
├── nodes.json
├── edges.json
├── indexes.json
└── graph_stats.json
```

---

## 3. 四个必须一起跑的实验变体

| 实验名 | 记忆抽取 | 检索方式 | 要验证的变量 |
|---|---|---|---|
| `dimmem_v1` | 独立 V1 六维抽取 | BM25 + Dense + 原维度 | 受控 DimMem 基线 |
| `dimension_v2` | 增强 V2 抽取 | BM25 + Dense + 增强维度 | 维度内容本身是否有效 |
| `graph_static` | 增强 V2 抽取 | V2 静态检索 + 固定图扩展 | 单纯建图是否有效 |
| `graph_active` | 增强 V2 抽取 | LLM 观察证据后自主选图工具 | 主动重构是否有效 |

四组使用相同的 Query Parser、QA 模型、Judge 模型、Top-K 和 embedding 配置，减少混杂变量。

> `dimmem_v1` 是本包对原始六维 Schema 的受控重实现，用于隔离“维度内容升级”的收益。它并不声称逐字复刻某个特定 commit 的完整原始运行结果。需要论文级严格复现时，应同时保留原始 DimMem pipeline 的独立结果，作为额外外部基线。

---

## 4. 安装到原始 DimMem 仓库

先准备原始仓库：

```bash
git clone https://github.com/ChowRunFa/DimMem.git
cd DimMem
```

解压本包后，在本包根目录执行：

```bash
bash scripts/install_into_dimmem.sh /path/to/DimMem
```

然后进入 DimMem 根目录：

```bash
cd /path/to/DimMem
pip install -r /path/to/DimMem_GraphV2_Validation/requirements.txt
```

推荐的 benchmark embedding 还需要：

```bash
pip install -r /path/to/DimMem_GraphV2_Validation/requirements-optional.txt
```

核心代码只依赖 `requests`；无 API 的 smoke test 使用内置 hash embedding 和 heuristic 模式。

---

## 5. API 配置

支持 OpenAI-compatible `/chat/completions` 接口：

```bash
export OPENAI_BASE_URL="https://your-endpoint/v1"
export OPENAI_API_KEY="YOUR_KEY"
export MODEL_NAME="YOUR_MODEL"
```

默认情况下，Extraction、Query Parse、Active Router、QA 和 Judge 共用以上模型。

也可以在 `all` 命令中分别传入：

```text
--extract-model
--query-model
--active-model
--qa-model
--judge-model
```

并为每个阶段分别传入对应的 `--*-base-url` 和 `--*-api-key`。

公平对照时建议：

1. V1 与 V2 抽取使用同一个模型；
2. 四个检索变体使用同一个 QA 与 Judge；
3. `temperature=0`；
4. 所有变体使用相同 `route-k/final-k`；
5. 不按 benchmark category 写 case-specific if/elif。

---

## 6. 一条命令从原始 LongMemEval 跑到报告

```bash
python -m longmemeval.graph_memory_v2 all \
  --input /path/to/longmemeval_s_cleaned.json \
  --output-root ./results/graph_memory_v2 \
  --run-name full_graph_active \
  --window-size 15 \
  --overlap 3 \
  --mode graph_active \
  --output-name graph_active \
  --route-k 20 \
  --initial-k 12 \
  --final-k 15 \
  --max-rounds 3 \
  --router-mode llm \
  --embedder sentence-transformers \
  --embedding-model sentence-transformers/all-MiniLM-L6-v2
```

`--input` 支持：

- 单个 JSON 文件；
- 一个包含各 category JSON 文件的目录；
- 顶层为 list；
- 或 `{"data": [...]}`。

关键输入字段兼容 LongMemEval 常见结构：

```text
question_id
question_type
question
answer
gold_answer
question_date
haystack_sessions
haystack_dates
haystack_session_ids
answer_session_ids（若存在，用于 evidence session recall）
```

---

## 7. 推荐按阶段运行，便于定位问题

### Stage 1：原始数据切窗

```bash
python -m longmemeval.graph_memory_v2 prepare \
  --input /path/to/longmemeval_s_cleaned.json \
  --output-root ./results/graph_memory_v2 \
  --run-name exp01 \
  --window-size 15 \
  --overlap 3
```

输出：

```text
results/graph_memory_v2/exp01/<question_type>/<sample_id>/windows/
```

重叠部分只用于消解指代和相对时间，不允许重复成为新 memory source。

### Stage 2：同时抽取 V1 与 V2 记忆

```bash
python -m longmemeval.graph_memory_v2 extract \
  --run-root ./results/graph_memory_v2/exp01 \
  --schema both
```

输出：

```text
memory_v1/all_memories.json
memory_v2/all_memories.json
```

### Stage 3：从 V2 记忆从零建图

```bash
python -m longmemeval.graph_memory_v2 build-graph \
  --run-root ./results/graph_memory_v2/exp01
```

### Stage 4：解析查询为可修正的多假设

```bash
python -m longmemeval.graph_memory_v2 parse-query \
  --run-root ./results/graph_memory_v2/exp01
```

Query Parse 不再输出一个不可变的硬判决，而是最多三个 hypothesis，包含：

```text
query_anchor
entities
keywords
time_constraint
state_keys
relation_hints
answer_dim
need_assistant_context
need_multi_hop
missing_slots
confidence
```

### Stage 5：分别跑四个检索版本

```bash
python -m longmemeval.graph_memory_v2 retrieve \
  --run-root ./results/graph_memory_v2/exp01 \
  --mode dimmem_v1 \
  --output-name dimmem_v1 \
  --route-k 20 --final-k 15 \
  --embedder sentence-transformers
```

```bash
python -m longmemeval.graph_memory_v2 retrieve \
  --run-root ./results/graph_memory_v2/exp01 \
  --mode dimension_v2 \
  --output-name dimension_v2 \
  --route-k 20 --final-k 15 \
  --embedder sentence-transformers
```

```bash
python -m longmemeval.graph_memory_v2 retrieve \
  --run-root ./results/graph_memory_v2/exp01 \
  --mode graph_static \
  --output-name graph_static \
  --route-k 20 --final-k 15 \
  --embedder sentence-transformers
```

```bash
python -m longmemeval.graph_memory_v2 retrieve \
  --run-root ./results/graph_memory_v2/exp01 \
  --mode graph_active \
  --output-name graph_active \
  --route-k 20 \
  --initial-k 12 \
  --final-k 15 \
  --max-rounds 3 \
  --router-mode llm \
  --embedder sentence-transformers
```

Active Router 可调用：

```text
search_text
expand_entity
expand_topic
expand_time
expand_session
follow_state_chain
expand_relations
inspect_sources
finish
```

每轮最多三次工具调用，记录 visited action，禁止重复相同调用，并保存完整 `retrieval_trace.json`。

### Stage 6：QA 与 Judge

对每个 retrieval name 执行：

```bash
python -m longmemeval.graph_memory_v2 qa-judge \
  --run-root ./results/graph_memory_v2/exp01 \
  --retrieval-name graph_active
```

### Stage 7：生成报告

```bash
python -m longmemeval.graph_memory_v2 report \
  --run-root ./results/graph_memory_v2/exp01 \
  --retrieval-name graph_active
```

输出：

```text
reports/report_graph_active.json
reports/report_graph_active.md
```

报告包含：

- Overall Accuracy；
- 各 LongMemEval category Accuracy；
- gold answer session recall（数据包含 `answer_session_ids` 时）；
- 平均 retrieval 时间；
- 平均 active rounds；
- 平均 tool calls；
- QA/Judge token 统计。

---

## 8. 一键跑完整消融

在 V1/V2 extraction、graph 和 query parse 完成后：

```bash
python -m longmemeval.graph_memory_v2 ablation \
  --run-root ./results/graph_memory_v2/exp01 \
  --route-k 20 \
  --initial-k 12 \
  --final-k 15 \
  --max-rounds 3 \
  --router-mode llm \
  --embedder sentence-transformers
```

自动生成四组结果和以下成对比较：

```text
dimmem_v1 vs dimension_v2
dimension_v2 vs graph_static
graph_static vs graph_active
dimmem_v1 vs graph_active
```

比较报告包括：

- 配对样本数；
- Accuracy delta；
- 修正 case 数；
- 破坏 case 数；
- paired exact binomial p-value；
- bootstrap 95% CI；
- fixed/broken question IDs。

---

## 9. 结果目录

```text
RUN_ROOT/
├── run_manifest.json
├── extraction_manifest_v1.json
├── extraction_manifest_v2.json
├── graph_manifest_v2.json
├── query_manifest_v2.json
├── <question_type>/
│   └── <sample_id>/
│       ├── input_item.json
│       ├── source_turns.json
│       ├── windows/
│       ├── memory_v1/
│       ├── memory_v2/
│       ├── graph_v2/
│       ├── query_v2/
│       ├── retrieval_v2/
│       │   ├── dimmem_v1/
│       │   ├── dimension_v2/
│       │   ├── graph_static/
│       │   └── graph_active/
│       ├── qa_v2/
│       └── judge_v2/
└── reports/
```

每个阶段均支持 `--force`，否则已存在的结果会跳过，方便断点续跑。

---

## 10. 快速 smoke test

无需 API：

```bash
bash scripts/run_smoke_test.sh
```

它会验证：

- 原始 LongMemEval 结构读取；
- overlap window；
- V1/V2 记忆生成；
- graph JSON 生成；
- state update chain；
- static/active retrieval；
- QA/Judge 输出；
- 报告输出；
- 单元测试。

`--smoke-test` 的 heuristic extraction/QA 只用于验证代码链路，不可用于报告 benchmark Accuracy。

---

## 11. 论文级实验建议

### 主表

至少报告：

| Method | Temporal | Multi-session | Knowledge-update | SS-user | SS-assistant | SS-preference | Overall | Tokens | Runtime |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|

### 关键消融

```text
V1 extraction + V1 static retrieval
a) V2 extraction + V2 static retrieval
b) a + static graph expansion
c) b + active graph traversal
c - state fields
c - event/validity split
c - preference structure
c - modality
c - active gate / max-round controls
```

### 重点错误分析

对 fixed 和 broken case 分别统计：

```text
Extraction error
Query parse error
Initial retrieval miss
Wrong graph edge
Wrong tool choice
Evidence found but QA failed
Over-generalized preference
Wrong current-vs-historical state
Assistant context omitted
```

### 防止过拟合

本代码没有根据 `question_type` 写检索分支，也没有针对 bike、project、times 等具体 benchmark 内容写 if/elif。Category 只用于筛选和结果统计。

---

## 12. 当前验证状态与限制

本包已完成：

- Python 全模块编译；
- 4 项单元测试；
- 2 个 synthetic LongMemEval case 的端到端 smoke run；
- V1/V2 独立抽取目录验证；
- graph nodes/edges/indexes/stats 输出验证；
- active trace、QA、Judge、report 输出验证。

尚未在本地执行完整 LongMemEval benchmark，因为交付环境没有用户的数据集、模型 API、原始 DimMem 环境和 Judge 配置。因此本包不声称已经取得高于 DimMem 的最终 Accuracy。

另有几个需要通过正式实验验证的研究风险：

1. V2 extraction 变复杂后，字段幻觉是否上升；
2. `state_key` 归一化是否稳定；
3. SAME_EVENT 弱边是否引入噪声；
4. Active Router 是否被额外推理预算而非图结构本身驱动；
5. V2 对 Preference 的收益是否来自更好 extraction，而不是 traversal；
6. 不同 LLM 对复杂 Schema 的遵循能力差异。

