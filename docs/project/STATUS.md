# 项目执行状态

最后更新：2026-10-10 CST

本文件与 [PLAN.md](PLAN.md) 的 S0–S3 一一对应。旧 P0–P5 与原始 Step 0–4
属于历史 demo 测量任务，入口见 [LEGACY_STEPS.md](LEGACY_STEPS.md)。

## 当前摘要

主任务已经调整为真实压缩权重推理的 AMD 性能优化。当前处于 S0 场景确认和
S1 实现准备阶段；解码到 LDS 的 kernel 尚未实现，ANS+GEMM 融合尚未实现。
此前 637.57 GB/s、15.54% 及相关 counter 只描述独立 byte decoder 向全局内存
输出的历史路径，不能作为新场景已完成或已经定位瓶颈的证据。

论文已经从用户指定的本地 PDF 复查：离线按 GEMM tile 压缩权重、每层共享
codebook、独立 tile-offset/state 元数据；运行时直接解码到 shared memory，
producer/consumer 与双缓冲重叠 GEMM。论文包含 dense 与 MoE，以及多种数值
格式。依据和限制见 [PAPER_SCENARIO.md](../PAPER_SCENARIO.md)。

## 任务状态

| 编号 | 状态 | 已完成的实际进展 | 下一验收点 |
|---|---|---|---|
| S0 | 进行中 | 论文场景、模型/格式范围已复查；旧任务完成边界已归档 | 选定有 provenance 的真实权重 tensor、精确 dtype、tile layout 和指标 |
| S1 | 待实现 | 第一阶段接口与验证要求已记录；现有 byte decoder 可作为参考实现 | 持久化预压缩权重和元数据；kernel 写入 LDS；验证实际 LDS store 与逐位正确性 |
| S2 | 未开始 | 已明确 LDS 数据必须在同一个 workgroup/kernel 中被 GEMM 消费 | 基于 S1 建立融合 kernel 和 producer/consumer 双缓冲 |
| S3 | 未开始 | 4-bit 权重与 dense/MoE 扩展列入目标 | 在同一数值格式和 workload 下衡量实际层级收益 |

## 可复用的历史资产

- `ans_bench`、HIP-event timing、roundtrip verification 与 out-of-tree 编译。
- rocprofv3 timeline、rocprof-compute counter 收集和不可用项记录。
- `results/peak_bw.json` 中 BabelStream Triad 为 4,102.71 GB/s。
- 历史 profiler 原始数据及 `results/profiling_summary.json`。
- `sepbase/` 的独立 GEMM 和串行 decode+GEMM，可作为后续比较参考。
- 保留 `ans_bench` 的现有 JSON API；新 LDS 路径使用独立接口以保留原字段含义。

## 已核实的缺口

- 当前 `ans_bench` 每次启动先 encode，再重复调用会重建 table 的全局输出 API；
  尚无可直接加载的推理期压缩权重 artifact 和每层共享 codebook 路径。
- 现有 hipANS 字母表是 256 个 byte symbols；BF16 拆成 byte 不等于 BF16 原生
  符号编码。4-bit packed bytes 也不等于独立 4-bit 符号。
- 当前 `inputs/` 实际只有生成的 normal/uniform/exponential 数据。README 曾提到
  的 Qwen BF16 文件当前不在该目录；需要定位或提取真实 tensor，而不必等待六个
  论文模型的全部权重。
- FP8 需要确定 E4M3/E5M2、FNUZ/OCP 编码及 scale。MI300X 原生 FP8 使用 FNUZ。
- 本地论文正文没有明确给出“30%”的带宽分子和分母；该数值暂不作为验收目标。
- 旧 counter 的 dependency wait 不能直接外推到未来 LDS kernel；barrier 与数据
  依赖的各自成本也尚未分离。

## 当前活动与后续节点

2026-10-10 启动 [真实 artifact + LDS baseline 独立交付](DELIVERY_LDS_BASELINE.md)。
已只读检查 entropyflow-lab 的 PLAN、registry、M2 结果、CPU rANS 与格式代码；
已核对 M2 parquet 中 `dsv4-flash-base / layers.0.ffn.experts.10.w1.weight`
为原生 FP8、8,388,608 元素。选择原生 FP8 避免模拟量化变量，本轮排除 4-bit。
checkpoint shard 将按 HTTP byte range 只提取目标 tensor/scale；上游 cache 不改写。
当前先实施离线包与逐位验证，之后记录 MI300X baseline 和 LDS store 证据。

当前没有本项目的活动 Slurm job。下一项可独立验收的实现是 S1 的首个 LDS
baseline：在计时前加载预压缩真实权重、预构建 table，按 tile 解码到 LDS 并
丢弃输出；验证模式从 LDS 导出结果逐位比对。详细要求见
[LDS_STAGE1.md](../LDS_STAGE1.md)。其后才推进 counter-guided 优化与 S2 融合。

此前 job `107394` 完整测量成功，job `107500` 已验证 ROCm provenance 为
`10.0.0`；两者均属于历史全局输出路径。

## 远端状态

2026-10-10 已将既有工程状态 `dd2bab7` 推送到
`origin/mi300-baseline`（`https://github.com/Joezia/ans-amd-bench.git`）。当前方向调整
与文档记录完成后也沿同一分支保存、推送。

## 阶段变更日志

- 2026-10-10：依据用户重新定义目标，旧“标准化+论文对照”任务归档；启动 S0–S3
  两阶段优化路线。保留上游只读、共享路径执行和本地 out-of-tree 构建约束。
