# 本地交付验证报告

验证日期：2026-07-12

## 已执行

```text
python -m compileall longmemeval/graph_memory_v2
python -m unittest discover -s tests -p 'test_*.py' -v
bash scripts/run_smoke_test.sh
```

## 结果

- 全部 Python 模块编译通过；
- 6/6 单元测试通过；
- 两个 synthetic LongMemEval case 的端到端并行流程成功；
- smoke 日志确认以下 stage 使用 2 个 case workers：
  - prepare；
  - V1 extraction；
  - V2 extraction；
  - query parse；
  - graph build；
  - retrieval；
  - QA；
  - Judge；
  - report；
- prepare 后的 V1 extraction、V2 extraction、query parse 使用 3 个 branch workers 同时执行；
- 输出成功生成：
  - V1 memory；
  - V2 memory；
  - graph nodes/edges/indexes/stats；
  - parsed query；
  - active retrieval trace；
  - QA prediction；
  - Judge result；
  - JSON/Markdown report。

## 单元测试覆盖

1. 相同 `state_key` 的新事实建立 `SUPERSEDES`；
2. V1 projection 不泄漏 V2 state fields；
3. heuristic active router 可沿 state chain 找回更新记录；
4. overlap window 正确标记上下文 turn；
5. parallel runner 使用多个线程并保持 manifest 原始顺序；
6. `workers=0/-1/N` 的并发解析行为正确。

## 并行语义

```text
--workers 0          所有选中 case 同时提交
--workers -1         保守自动线程数
--workers N          最大同时执行 N 个 case
--pipeline-workers 0 V1/V2/query 三分支同时执行
--schema-workers 0   V1/V2 schema 同时抽取
--window-workers 0   同一 case 的全部窗口同时抽取
--route-workers 0    BM25/Dense/Dimension 同时召回
--tool-workers 0     同一主动推理轮的全部图工具同时执行
--variant-workers 0  四个消融版本同时执行
```

## 未执行

- 完整 LongMemEval-S；
- 外部 LLM API 的高并发压力测试；
- SentenceTransformer benchmark embedding 的真实 GPU 压力测试；
- 原始 DimMem 仓库的完整依赖环境；
- 与论文报告数值的复现比较。

因此 smoke Accuracy 不具有研究意义，不应写入论文或实验表。正式运行前需要根据 API 网关并发限制选择 `--workers`。默认 `0` 严格满足“全部 case 并行”，但可能触发 429、连接数上限或瞬时显存/内存压力。
