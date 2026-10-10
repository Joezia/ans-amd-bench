# 项目执行状态

最后更新：2026-10-11 CST

本文件与 [PLAN.md](PLAN.md) 的 S0–S3 一一对应。旧 P0–P5 与原始 Step 0–4
属于历史 demo 测量任务，入口见 [LEGACY_STEPS.md](LEGACY_STEPS.md)。

## 当前摘要

主任务为真实压缩权重推理的 AMD 性能优化。S0 的首个真实负载已确认；S1 的
首项独立交付“真实权重离线 artifact + 可验证 LDS baseline”已完成。当前进入
S1 优化前的实验扩展准备；ANS+GEMM 融合尚未实现。本次未使用 4-bit。
此前 637.57 GB/s、15.54% 及相关 counter 只描述独立 byte decoder 向全局内存
输出的历史路径，不能作为新场景已完成或已经定位瓶颈的证据。

论文已经从用户指定的本地 PDF 复查：离线按 GEMM tile 压缩权重、每层共享
codebook、独立 tile-offset/state 元数据；运行时直接解码到 shared memory，
producer/consumer 与双缓冲重叠 GEMM。论文包含 dense 与 MoE，以及多种数值
格式。依据和限制见 [PAPER_SCENARIO.md](../PAPER_SCENARIO.md)。

## 任务状态

| 编号 | 状态 | 已完成的实际进展 | 下一验收点 |
|---|---|---|---|
| S0 | 首个负载已完成 | 与 entropyflow-lab 对齐的 DeepSeek-V4-Flash-Base 原生 FP8 专家矩阵已提取，来源/shape/scale/hash 已记录；BF16 dense 作为辅助候选 | 后续扩展到多矩阵与缓存外真实工作集 |
| S1 | 进行中：baseline 已完成 | 持久化 artifact、独立 LDS kernel、逐位验证、raw/control 对照与 AMD profiling 均完成；未做性能优化 | 基于新路径证据建立受控优化实验，补齐 >256 MiB 真实工作集 |
| S2 | 未开始 | 已明确 LDS 数据必须在同一个 workgroup/kernel 中被 GEMM 消费 | 基于 S1 建立融合 kernel 和 producer/consumer 双缓冲 |
| S3 | 未开始 | 4-bit 权重与 dense/MoE 扩展列入目标 | 在同一数值格式和 workload 下衡量实际层级收益 |

## 可复用的历史资产

- `ans_bench`、HIP-event timing、roundtrip verification 与 out-of-tree 编译。
- rocprofv3 timeline、rocprof-compute counter 收集和不可用项记录。
- `results/peak_bw.json` 中 BabelStream Triad 为 4,102.71 GB/s。
- 历史 profiler 原始数据及 `results/profiling_summary.json`。
- `sepbase/` 的独立 GEMM 和串行 decode+GEMM，可作为后续比较参考。
- 保留 `ans_bench` 的现有 JSON API；新 LDS 路径使用独立接口以保留原字段含义。

## 已实现的独立交付（D0–D3，与交付计划对应）

| 节点 | 状态 | 实际证据 |
|---|---|---|
| D0 | 完成 | 只读检查 entropyflow-lab 的 registry、M2 parquet、格式和参考 rANS；同一 revision/tensor 的原生 FP8，无模拟转换 |
| D1 | 完成 | `inputs/lds/dsv4-flash-base-expert10-w1/` 保存源权重、scale、离线 ANS 包与两个 manifest；所有 streams 的 CPU roundtrip 通过 |
| D2 | 完成 | `bench/lds_bench.cpp` 从实际 LDS 读回逐位比对；11 项 CPU 回归、15 项 GPU 边界验证；四个计时 discard kernel 的 ISA 均保留 `ds_write_b8`，无 FLAT store |
| D3 | 完成 | MI300X job `109484` COMPLETED (0:0)，68 秒；4 组未插桩 timing、timeline、完整 counter 和 decoder-only 分析已落盘 |

实际负载：`deepseek-ai/DeepSeek-V4-Flash-Base@8855555...`，
`layers.0.ffn.experts.10.w1.weight`，2048×4096 `F8_E4M3`，8,388,608 bytes；
配套 `[16,32]` F32 scale 为 2,048 bytes。64×128 tiles、每 tile 64 streams，
每矩阵一张共享概率表。离线 payload 为 6,884,189 bytes，metadata 为 1,049,664
bytes，含 scale 的真实压缩比 1.05730×。这不是 hipANS archive 格式。

正式未优化 warm-cache baseline（10 warmup / 50 iterations）：LDS-table decode
93.565 GB/s / 89.655 us；global-LUT decode 123.905 GB/s / 67.702 us；raw-to-LDS
588.220 GB/s / 14.261 us；纯 LDS control 1,414.844 GB/s / 5.929 us（不是权重解码）。

正式 counter 在两次 warmup 后只分析 dispatch 3：wave occupancy 4.59%，VALU
6.98%，dependency wait / wave cycles 60.26%，LDS allocation 24 KiB，
insufficient-CU-LDS 36.19%，LDS instructions 294,912，L2-fabric read 90.92 GB/s、
write 0。完整证据见 [LDS_BASELINE_REPORT.md](../LDS_BASELINE_REPORT.md) 与
`results/lds_baseline.json`。旧 637.57 GB/s 不是同一 kernel、数据或输出路径。

## 已核实的剩余缺口与边界

- 当前 `ans_bench` 每次启动先 encode，再重复调用会重建 table 的全局输出 API；
  新独立路径已经直接加载离线 artifact；旧接口保留原语义，不混用结果。
- 现有 hipANS 字母表是 256 个 byte symbols；BF16 拆成 byte 不等于 BF16 原生
  符号编码。4-bit packed bytes 也不等于独立 4-bit 符号。
- 当前真实输入为上述原生 FP8 矩阵；Qwen BF16 辅助候选尚未提取。
- 当前保留 checkpoint 的 F8_E4M3 位模式，scale 已持久化但不参与计时 kernel；
  没有 FP8 数值运算。MI300X GEMM 的 FNUZ 兼容性与 scale 消费留待融合阶段。
- 当前 reference 使用独立 byte-rANS lane streams，借用上游 CPU 参考设计，不是
  已优化 hipANS kernel。表覆盖单个矩阵，不是整个 transformer 层的全部权重。
- 当前 8 MiB 单矩阵反复运行，不是 >256 MiB streaming；global LUT 不是跨 kernel
  保留的 resident LDS。LDS-table staging 已包含在 kernel timing 内。
- 本轮没有独立 MALL hit-rate、barrier 时间分解或新 roofline 微基准。L2-fabric
  流量不能无条件解释为 DRAM pin 带宽；BabelStream 仍为历史独立 HBM 参考。
- 本地论文正文没有明确给出“30%”的带宽分子和分母；该数值暂不作为验收目标。
- 旧 counter 的 dependency wait 不能直接外推到未来 LDS kernel；barrier 与数据
  依赖的各自成本也尚未分离。

## 当前活动与后续节点

当前没有本项目的活动 Slurm job。首轮 job `109479` 完成（67 秒），确认任务
`109484` 完成（68 秒），二者分别保留首次 dispatch / warm dispatch 的 profiler
条件；正式报告使用后者，不拿插桩 HIP-event 时间替代未插桩 baseline。
共享路径返回文件为 `results/slurm-lds-baseline-109484.out/.err`，原始证据目录
`results/profiling/lds-109484/`。如何运行和查看返回结果见 [LDS_CLI.md](../LDS_CLI.md)。

本次 [独立交付计划](DELIVERY_LDS_BASELINE.md) 的 D0–D3 已验收。下一项需要另立
范围：将 hipANS 接入同一真实预压缩权重→LDS 测量/验证场景，保留现有 scalar
reference 作为正确性与性能对照；扩展缓存外真实负载，并针对 occupancy、table
footprint 和 stream 访存做受控 A/B。可在 hipANS 内实施修改，不自动开始 GEMM
融合或 4-bit 扩展。本次权限澄清仅修正文档/构建提示，未修改任何 codec kernel。

此前 job `107394` 完整测量成功，job `107500` 已验证 ROCm provenance 为
`10.0.0`；两者均属于历史全局输出路径。

## 远端状态

既有工程状态与方向调整已推送到 `origin/mi300-baseline`，远端为
`e9f4709`（`https://github.com/Joezia/ans-amd-bench.git`）。本次新增独立交付已分步
本地提交；未自动扩大先前 push 的授权到此次新增实现。

## 阶段变更日志

- 2026-10-10：依据用户重新定义目标，旧“标准化+论文对照”任务归档；启动 S0–S3
  两阶段优化路线。当时保留上游只读、共享路径执行和本地 out-of-tree 构建约束。
- 2026-10-11：用户明确允许修改 hipANS，撤销其源码只读限制；entropyflow-lab
  仍只读，构建保持 out-of-tree，记录两仓库的独立提交及每次实验 provenance。
