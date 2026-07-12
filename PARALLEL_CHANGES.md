# 全 Case 并行改造说明

## 原版本确认

原版本在以下 run-level 函数中逐 case 顺序执行：

```text
prepare_dataset
extract_run_v1
extract_run
build_run_graphs
parse_run
run_retrieval_run
run_qa_judge_run
build_report
```

因此原来的建图并不是并行的，后续 retrieval、QA/Judge 等也不是 case-level 并行。

## 当前版本

新增统一调度器：

```text
longmemeval/graph_memory_v2/parallel.py
```

并行范围：

1. 所有 case 的数据准备；
2. 所有 case 的 V1/V2 memory extraction；
3. 所有 case 的从零建图；
4. 所有 case 的 query parse；
5. 所有 case 的四类 retrieval；
6. 所有 case 的 QA；
7. 所有 case 的 Judge；
8. 所有 case 的 report 与 paired comparison；
9. V1 extraction、V2 extraction、query parse 三个 pipeline 分支；
10. 每个 case 内的 extraction windows；
11. 每个 case 内的 BM25/Dense/Dimension 三路召回；
12. Active Router 同一轮输出的多个图工具；
13. 四个 ablation variants。

## 参数

```text
--workers 0           所有 case 并行
--pipeline-workers 0  三个独立 pipeline 分支并行
--schema-workers 0    V1/V2 extraction 并行
--window-workers 0    每个 case 的全部 extraction windows 并行
--route-workers 0     BM25/Dense/Dimension 并行
--tool-workers 0      同一 active round 的图工具并行
--variant-workers 0   四个消融版本并行
```

`0` 表示全部展开，`-1` 表示保守自动值，正整数表示并发上限。

## 无法并行的依赖

- 建图必须等待 V2 memory extraction；
- retrieval 必须等待 query parse 和所需 memory/graph；
- Judge 必须等待同 case 的 QA；
- Active Reconstruction 的第 t+1 轮必须等待第 t 轮产生的新证据。

这些地方保留顺序执行，否则会改变算法语义或读取尚未产生的文件。
