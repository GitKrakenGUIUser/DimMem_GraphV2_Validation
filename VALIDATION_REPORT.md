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
- 4/4 单元测试通过；
- synthetic LongMemEval 端到端流程成功生成：
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
4. overlap window 正确标记上下文 turn。

## 未执行

- 完整 LongMemEval-S；
- 外部 LLM API；
- SentenceTransformer benchmark embedding；
- 原始 DimMem 仓库的完整依赖环境；
- 与论文报告数值的复现比较。

因此 smoke Accuracy 不具有研究意义，不应写入论文或实验表。
