# 项目执行状态

最后更新：2026-10-11。战略编号与 [PLAN.md](PLAN.md) 对齐。

## 当前摘要

当前目标：真实预压缩权重场景下的 AMD 解码优化，随后再融合 GEMM。
S1 baseline 正在重做，覆盖 MI300X 与 MI350X，必须使用当前未优化 hipANS。错误 scalar baseline 的
结果与报告被覆盖，不再作为对照或任务完成依据。源权重、scale 与来源记录保留。
本轮不做算法优化、GEMM 融合或 4-bit。

| 阶段 | 状态 | 实际进展 | 下一验收点 |
|---|---|---|---|
| S0：真实负载 | 首个负载完成 | entropyflow-lab 对齐的 DeepSeek-V4-Flash-Base 原生 FP8 专家矩阵，2048×4096，8 MiB，scale 2 KiB；来源与 hash 已核实 | 后续扩展真实多矩阵工作集 |
| S1：片上解码 | 双机器 baseline | 当前 hipANS clean revision 8cc8bc2；MI350X 的 GPU 验证、sweep、复测与 profiling 已完成；MI300X 等待资源 | 分机器解释 counters，MI300X 返回后对照；尚未开始优化 |
| S2：解码+GEMM | 未开始 | 场景要求同一 kernel/workgroup 消费 LDS | 基于 S1 构建融合路径 |
| S3：部署验证 | 未开始 | 后续格式及模型扩展另行规划 | 完整推理价值验证 |

## 当前独立交付：H0–H3

计划见 [DELIVERY_HIPANS_S1_BASELINE.md](DELIVERY_HIPANS_S1_BASELINE.md)。

| 节点 | 状态 | 证据/验收 |
|---|---|---|
| H0：固定真实实现与负载 | 完成 | 实际 hipANS revision、native FP8 tensor provenance 已固定 |
| H1：原生离线包与 LDS 验证 | MI350X 完成；MI300X 待运行 | 实际 hipANS 原生 archive；MI350X 70 项 GPU 验证、14 个 timed kernel ISA 审核通过 |
| H2：资源饱和实验 | MI350X 完成；MI300X 排队 | 53 组记录，含 8 MiB–2 GiB 工作量、部署/并发搜索与确认；大两档差异 2.33% |
| H3：确认与资源证据 | MI350X 证据已返回，归因待分析 | 三次确认 509.514/509.385/509.475 GB/s；两个配置的完整 counters/timeline；单队列 257 项工具指标已整理 |

## 活动任务与返回结果

Slurm job **109618** 已提交到 mi3001x，当前等待 GPU 资源。
共享路径可直接访问，无需节点间复制。

MI350X 独立任务 **109666**，已在 tn1-004-001 完成（0:0，耗时 3 分 5 秒）：
gfx950 原生库与 LDS adapter 构建、同一真实 FP8 矩阵的离线包/验证、部署 sweep 与 profiling。
实际设备名称已通过 rocminfo 确认 AMD Instinct MI350X，wave64；两类 profiler 已安装，
rocprofv3 与完整 rocprof-compute 采集均成功。MI300X 任务保留，不相互替代。

| 机器 | 作业 | 当前状态 | 正式结果/报告 |
|---|---|---|---|
| MI300X / gfx942 | 109618 | 等待资源 | results/lds_baseline.json；docs/LDS_BASELINE_REPORT.md |
| MI350X / gfx950 | 109666 | COMPLETED (0:0) | results/lds_baseline_mi350x.json；docs/LDS_BASELINE_REPORT_MI350X.md |

MI350X stdout/stderr 为 `results/slurm-hipans-lds-mi350x-109666.out/.err`，
原始数据为 `results/profiling/hipans-lds-mi350x-109666/`。
二者分别使用独立 binary、原生 build、ISA 和 artifact。两机任务不自动修改共享
STATUS，避免并发写入冲突；以 Slurm 状态和各自正式结果为实时证据。
gfx942 适配器已借用另一 MI350X 分配交叉编译通过，使 MI300X binary 与当前源码
一致；原重复排队构建 109667 已取消，没有取消 MI300X 测量任务。

MI350X 测试范围内最佳配置：384 threads、6 waves/workgroup、grid multiplier 8、
4 queues、256 独立地址副本；有效 ANS→LDS 509.475 GB/s，匹配 raw→LDS
669.924 GB/s。三次确认 spread 为 0.0252%，并非推理/GEMM 性能。
默认单队列（128 threads）counter 的 occupancy 为工具峰值的 45.55%、VALU 25.96%；
不能直接用这些 counter 解释最佳多队列配置。详细报告见
[MI350X 独立报告](../LDS_BASELINE_REPORT_MI350X.md)。

- 查询：`squeue -j 109618`；完成后：`sacct -j 109618 --format=JobID,State,ExitCode,Elapsed`。
- stdout：`results/slurm-hipans-lds-109618.out`；stderr：同名 `.err`。
- 逐项记录：`results/profiling/hipans-lds-109618/sweep/`。
- 正式结果：`results/lds_baseline.json`，当前为待测状态，旧数值已失效。
- 报告：[LDS_BASELINE_REPORT.md](../LDS_BASELINE_REPORT.md)，仅使用新测量。

## 测量边界

- 首个真实矩阵为 8 MiB。独立设备地址的副本用于资源饱和实验，不声称是新增模型参数。
- 有效吞吐是未填充原始权重 bytes × replicas / 所有队列共同完成的 GPU event 时间。
- table 构建、H2D、验证读回在计时外；每次 kernel 的 LDS table staging 包含在内。
- native FP8 位模式无转换；scale 保存但不参与此阶段 kernel，没有 FP8 GEMM。
- 当前允许修改 hipANS；本次 baseline 不修改其算法。entropyflow-lab 与缓存只读。
- BabelStream Triad 4102.71 GB/s 是独立 HBM 参考，不是 LDS 带宽上限。
- 该 BabelStream 数值仅来自 MI300X，不能用作 MI350X 分母；MI350X 尚未测峰值。
- 本次 gfx950 先复用相同初始部署搜索范围，不宣称穷尽 MI350X 可用配置或调优。
- 旧 global-output `ans_bench` API 保留，指标不可与 LDS baseline 混用。

## 版本管理

远端既有工程分支为 origin/mi300-baseline。本轮变更小步本地提交，不自动 push。
GPU 排队不表示验证完成；报告必须区分已编译、已提交、实测与饱和验收。
