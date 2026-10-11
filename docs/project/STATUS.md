# 项目执行状态

最后更新：2026-10-11。战略编号与 [PLAN.md](PLAN.md) 对齐。

## 当前摘要

当前目标：贴近真实预压缩权重计算场景的 AMD 解码优化，随后再融合 GEMM。
实际未优化 hipANS 的已有工程测量在两台机器均已返回，但此前 replicas 饱和
实验不再作为主要场景验收。错误 scalar baseline 已撤销，不作为对照。
新的 R0–R3 交付仅完成计划；没有启动新实验、修改 kernel 或提交新 Slurm 作业。

| 阶段 | 状态 | 实际进展 | 下一验收点 |
|---|---|---|---|
| S0：真实负载 | 工程输入已有；代表场景待确认 | 上游对应的原生 FP8 专家矩阵 2048×4096、8 MiB，scale 2 KiB；尚未确认目标 GEMM 的权重消费需求 | R0 提交经核实的候选矩阵及场景，用户确认 |
| S1：片上解码 | 工程路径已验证；真实场景 baseline 未完成 | hipANS clean revision 8cc8bc2；MI300X / MI350X LDS 验证、部署诊断和 profiling 已返回；未优化 | R1–R3 完成对应场景的 artifact、baseline 与归因 |
| S2：解码+GEMM | 未开始 | 尚无 GEMM 消费者和融合流水线 | 依据 S1 场景证据另行规划 |
| S3：部署验证 | 未开始 | 格式及模型扩展另行规划 | 完整推理价值验证 |

## 当前独立交付：R0–R3

计划见 [DELIVERY_REAL_SCENARIO_BASELINE.md](DELIVERY_REAL_SCENARIO_BASELINE.md)。

| 节点 | 状态 | 实际进展 / 下一验收点 |
|---|---|---|
| R0：确认权重与计算场景 | 已规划，待核实 | 候选为上游 DeepSeek-V4-Flash-Base / GLM-5.2-FP8 的实际矩阵；尚未核实新 tensor 名、shape 和适用计算；先提交 1–2 个候选给用户确认 |
| R1：场景 artifact 与正确性 | 未开始 | 现有 8 MiB artifact 可复用工程路径，不代表新场景已经完成；确认 R0 后再提取新权重 |
| R2：两机场景 baseline | 未开始 | 已有二机测量仅作诊断；需要按确认场景测量；优先 MI350X，MI300X 独立记录 |
| R3：瓶颈证据及建议 | 未开始 | 已有 counters 可帮助准备分析，不可直接替代新场景 profiling |

R0 需确定 prefill / decode 等推理阶段、输入规模、权重矩阵角色，以及目标
GEMM 的 tile 消费与复用需求。当前 CLI 不包含 activation、token 或 GEMM
消费者，因此尚不能证明其线性遍历就是目标推理的访问过程。
本轮优先原生 FP8、不使用 4-bit；BF16 进入实验前需单独明确可逆符号组织。
下一个反馈点是 R0 候选及理由确认，不自动开始融合或算法优化。

## 已有工程诊断：H0–H3

历史范围见 [DELIVERY_HIPANS_S1_BASELINE.md](DELIVERY_HIPANS_S1_BASELINE.md)。
保留有效实测，但不将 H2/H3 的 replicas 饱和结果称为真实推理场景验收。

| 节点 | 状态 | 证据 / 边界 |
|---|---|---|
| H0：固定真实实现与输入 | 完成 | hipANS revision 8cc8bc263bc943557682a5cc675a045cb9a86412；原生 FP8 tensor provenance 已固定 |
| H1：原生离线包与 LDS 验证 | 两机完成 | 两机分别完成 70 项 GPU 验证与 14 个 timed kernel ISA 审核 |
| H2：资源供给诊断 | 两机完成 | 各 53 组记录，含 8 MiB–2 GiB 解码产出、部署/并发搜索和重复确认 |
| H3：确认与资源证据 | 数据已返回，完整归因待完成 | 两机有 counters/timeline；默认单队列 counters 不能直接解释最佳并发配置 |

## 作业与已返回结果

2026-10-11 使用 sacct 核实：两个测量作业均 COMPLETED，ExitCode 0:0。
共享路径可从各节点访问，无需复制。当前没有为 R0–R3 提交作业。

| 机器 | 作业 | 状态 / 耗时 | 正式结果 / 报告 |
|---|---|---|---|
| MI300X / gfx942 | 109618 | COMPLETED (0:0)，3 分 20 秒，tn1-003-002 | results/lds_baseline.json；docs/LDS_BASELINE_REPORT.md |
| MI350X / gfx950 | 109666 | COMPLETED (0:0)，3 分 5 秒，tn1-004-001 | results/lds_baseline_mi350x.json；docs/LDS_BASELINE_REPORT_MI350X.md |

- 查询历史：`sacct -j 109618,109666 --format=JobID,State,ExitCode,Elapsed`。
- MI300X stdout/stderr：`results/slurm-hipans-lds-109618.out/.err`。
- MI350X stdout/stderr：`results/slurm-hipans-lds-mi350x-109666.out/.err`。
- 原始测量分别位于 `results/profiling/hipans-lds-109618/` 与 `results/profiling/hipans-lds-mi350x-109666/`。
- 两机使用独立 build、binary、ISA 和 artifact；作业不自动修改共享 STATUS。
- 原重复构建作业 109667 已取消；109618 的实际测量正常完成。
- 本次计划更新前，MI300X 结果 JSON 和报告已有作业生成的未提交改动；保留，不纳入此次计划文档提交。

## 已有测量摘要与解释边界

| 机器 | 搜索并确认的 2 GiB 产出吞吐 | 对应部署 | 匹配 raw→LDS |
|---|---|---|---|
| MI300X | 274.830 GB/s | 128 threads，grid multiplier 8，8 streams，256 replicas | 445.222 GB/s |
| MI350X | 509.475 GB/s | 384 threads，grid multiplier 8，4 streams，256 replicas | 669.924 GB/s |

上述是指定工作量及搜索范围内的有效 ANS→LDS 速率，不是所有工作量下的
绝对最大值，更不是推理或 GEMM 性能。默认部署的 128 MiB 产出记录分别为
318.720 / 567.465 GB/s，说明缓存与工作量条件会改变结果。
raw 对照不执行 ANS，资源占用及实际 grid 可能不同，不能把速率差直接全算作
纯 ANS 算术开销。详细证据见 [MI300X 报告](../LDS_BASELINE_REPORT.md)、
[MI350X 报告](../LDS_BASELINE_REPORT_MI350X.md)。

- 唯一源矩阵为 8 MiB；replicas 是实际独立设备地址的同内容副本，不是新增模型参数或 batch。
- 当前 64×128 FP8 tile 为 8192 bytes，共 1024 tiles；每 wave 解码一个 tile。
- LUT 为 4 KiB；每 workgroup 在 grid-stride loop 前加载一次，后续 tile 复用本组 LUT 和 LDS 输出区；不是每 CU 只有一份表。
- 当前有效吞吐分子为原始未填充 bytes × 实际处理 replicas，分母为所有队列共同完成的 HIP-event 时间。
- table 构建、H2D、验证读回在计时外；kernel 内 table staging 和 LDS 完成等待在计时内。
- native FP8 位模式未转换；scale 保存但未消费；计时 kernel 写 LDS 后丢弃，无 GEMM。
- MI350X 默认单队列工具 occupancy 为峰值的 45.55%、VALU 25.96%；不是最佳多队列配置的 counters。
- BabelStream Triad 4102.71 GB/s 是 MI300X 的 HBM 参考，不是 LDS 上限，也不能作为 MI350X 分母。
- hipANS 允许修改，但已有 baseline 未改算法；entropyflow-lab 及缓存只读。
- 旧 global-output ans_bench API 保留，与 LDS baseline 指标不能混用。

## 版本管理

远端既有工程分支为 origin/mi300-baseline。新计划小步本地提交，不自动 push。
区分已规划、已提交、已实测及已验收；已有工程诊断完成不代表新场景交付完成。
