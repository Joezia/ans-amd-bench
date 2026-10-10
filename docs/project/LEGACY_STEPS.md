# 原始 Step 0–4：产物索引与完成边界

这些编号来自最初任务描述，以前没有独立的 Step 索引，这是查找困难的原因。
2026-10-10 起它们作为历史记录；当前主任务是 [PLAN.md](PLAN.md) 的 S0–S3。

| 原始步骤 | 内容 | 可直接查看的产物 | 完成范围 |
|---|---|---|---|
| Step 0 | 盘点仓库、上游接口、旧结果 | [INVENTORY.md](../INVENTORY.md) | 完成 demo 工程盘点 |
| Step 1 | 标准 JSON CLI、正确性和 provenance | [CLI.md](../CLI.md)、[ans_bench.cpp](../../bench/ans_bench.cpp)、[standard_mi300.json](../../results/standard_mi300.json) | 完成全局输出 byte decoder；不是 LDS decoder |
| Step 2 | 指标口径与 BabelStream HBM 参考 | [METRICS.md](../METRICS.md)、[peak_bw.json](../../results/peak_bw.json)、[run_peak_bw.sh](../../scripts/run_peak_bw.sh) | 完成历史指标和 HBM 参考；新 LDS 指标需独立定义 |
| Step 3 | Profiling 与诊断 sweeps | [PROFILING_REPORT.md](../PROFILING_REPORT.md)、[profiling_summary.json](../../results/profiling_summary.json)、[profiling/](../../profiling/) | 完成 demo 路径采样与初步归因；未覆盖真实 LDS/GEMM 路径 |
| Step 4 | 论文模型的对照框架 | [reproduce_paper.py](../../scripts/reproduce_paper.py)、[paper_reproduction.json](../../results/paper_reproduction.json) | 只有脚本和缺失输入表；未完成模型测量，也未复现融合 microbenchmark |

P0–P5 是另一套旧战略编号，不是 Step 编号的同义词。完整历史定义与执行记录见
[LEGACY_PLAN.md](LEGACY_PLAN.md)、[LEGACY_STATUS.md](LEGACY_STATUS.md)。

637.57 GB/s 是旧 GPU 解码 API 的原始 byte 输出速率；15.54% 是它与 BabelStream
Triad 之比。它们不表示解码到 LDS 的速率，旧报告对性能限制的解释仅对其采样路径
成立。完整论文模型复现现为可选参考工作。
