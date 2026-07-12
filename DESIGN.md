# 设计说明：为什么不是“直接把 DimMem JSON 接到 MRAgent”

## 1. 分层而不是机械拼接

本实现把系统拆成四个可单独消融的层：

1. **Representation**：V1/V2 记忆内容；
2. **Indexing**：BM25、Dense、Dimension；
3. **Graph**：显式关系和更新链；
4. **Policy**：LLM 是否以及如何继续遍历。

若直接将 DimMem memory 变成 MRAgent Cue–Tag–Content 图，最终提升无法判断来自：

- 更好的 extraction；
- 更多 LLM token；
- 图扩展；
- rerank；
- Assistant source recovery。

四组主实验正是为拆开这些因素。

## 2. 图边的可信度层级

### 高可信边

由确定字段直接产生：

```text
IN_SESSION
ABOUT_ENTITY
AT_LOCATION
HAS_TOPIC
EVENT_START
VALID_FROM
ASSERTS_STATE
```

### 中可信边

由同一 `state_key` 和时间排序产生：

```text
SUPERSEDES
SUPPORTS
```

### 弱边

由实体、主题、时间联合分组产生：

```text
SAME_EVENT
```

弱边权重更低，并限制组大小，避免形成高噪声超级节点。

## 3. 为什么保留 JSON 图

- 无需额外数据库；
- 每个 case 可独立复现；
- 易于检查错误边；
- 易于写单元测试；
- 可直接迁移到 Neo4j、NetworkX 或向量数据库；
- 不把存储后端与研究假设绑定。

## 4. Active Router 的约束

Router 不允许生成任意 Cypher，而只能调用有限的类型化工具。这样可测量：

- tool selection；
- repeated action；
- active rounds；
- invalid tool calls；
- state-chain 使用频率；
- 时间锚点重新检索频率。

这比自由文本图导航更容易做严格实验。

## 5. 并行执行模型

当前实现采用“有向无环依赖 + case-level 并发”的执行模型：

```text
prepare
  ├─ extract_v1 ──────────────┐
  ├─ extract_v2 → build_graph ├→ retrieval → QA → Judge → report
  └─ parse_query ─────────────┘
```

- 同一阶段中的 LongMemEval case 彼此隔离，默认全部并行；
- V1 extraction、V2 extraction、query parse 在 prepare 后并行；
- 同一 case 的 extraction windows 可并行；
- BM25、Dense、Dimension 三条召回路由可并行；
- Active Router 同一轮产生的多个只读图工具可并行；
- QA 与 Judge 分成两个并行 phase，因为 Judge 依赖同 case 的 QA 输出；
- 四个消融变体可以同时运行；
- active reconstruction 的不同 reasoning rounds 必须顺序执行，因为后一轮依赖前一轮证据。

并行调度集中在 `parallel.py`，所有阶段保持确定性的 manifest 顺序，失败 case 会写入失败记录而不是无声丢失。`--fail-fast` 可改成首错终止。

SentenceTransformer 模型在 retrieval run 中只加载一次。各 case 共享模型，`encode()` 使用锁保护，以避免在单 GPU 上并发重入或为每个 case 重复加载模型导致显存溢出。外层 case、图扩展和 LLM Router 仍保持并发。
