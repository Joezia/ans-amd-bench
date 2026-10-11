# 项目执行状态

最后更新：2026-10-11。战略编号与 [PLAN.md](PLAN.md) 对齐。

## 当前摘要

当前目标：贴近真实预压缩权重计算场景的 AMD 解码优化，随后再融合 GEMM。
实际未优化 hipANS 的已有工程测量在两台机器均已返回，但此前 replicas 饱和
实验不作为主要场景验收。错误 scalar baseline 已撤销，不作为对照。
S1 的 R0–R3 已并入战略计划。用户已选定 GLM-5.2-FP8 dense projection；R1 的
MI350X artifact 与 LDS 逐位验证完成。R2 的 MI300X 单矩阵主测量及辅助 replica
诊断已完成；MI350X 作业仍在队列。尚未修改 decoder kernel。

| 阶段 | 状态 | 实际进展 | 下一验收点 |
|---|---|---|---|
| S0：真实负载 | 首个场景已选 | GLM-5.2-FP8 layer 0 dense MLP gate projection，native F8_E4M3 | 在 R2 测量确认场景解码表现 |
| S1：片上解码 | R1 完成；R2 部分完成 | GLM artifact 已在 MI350X 完成逐位 LDS 验证；MI300X 场景测量已返回，MI350X 在队列中 | 完成 MI350X 对照，随后完成两机 R3 profiling 归因 |
| S2：解码+GEMM | 未开始 | 尚无 GEMM 消费者和融合流水线 | 依据 S1 场景证据另行规划 |
| S3：部署验证 | 未开始 | 格式及模型扩展另行规划 | 完整推理价值验证 |

## 当前独立交付：R0–R3

计划见 [DELIVERY_REAL_SCENARIO_BASELINE.md](DELIVERY_REAL_SCENARIO_BASELINE.md)。

| 节点 | 状态 | 实际进展 / 下一验收点 |
|---|---|---|
| R0：确认权重与计算场景 | 完成 | 用户已选 GLM-5.2-FP8 layer 0 dense MLP `gate_proj`；候选依据见下表 |
| R1：场景 artifact 与正确性 | 完成 | GLM 72 MiB F8_E4M3 权重和 18 KiB scale；MI350X job 109840 实际 LDS 逐位验证通过 |
| R2：两机场景 baseline | MI300X 完成；MI350X 排队 | job 109848 已完成：单矩阵 decode→LDS 276.39 GB/s，raw→LDS 507.63 GB/s；replica 4/16/64 单独记录为辅助诊断；MI350X job 109847 仍排队 |
| R3：瓶颈证据及建议 | 两机作业已提交 | MI300X job 109852 排队；MI350X job 109853 依赖 R2 job 109847 成功完成后运行；均使用 GLM 单矩阵 R2 主配置 |

R0 只读取 entropyflow-lab 固定 revision 的配置/index，以及 GLM 对应 safetensors
shard 的有界 header range；未下载 checkpoint 权重 payload，也未写入 entropyflow-lab
或其 cache。

| 候选 | 已核实事实 | 场景适配判断 |
|---|---|---|
| 已选：GLM-5.2-FP8 layer 0 `model.layers.0.mlp.gate_proj.weight` | `zai-org/GLM-5.2-FP8@f33c6dc501ee5a2c7e35155653b1b1abbc320951`；F8_E4M3 `[12288,6144]`，75,497,472 bytes = 72 MiB；伴随 `weight_scale_inv` 为 F32 `[96,48]`（18 KiB）；配置给出 hidden 6144、intermediate 12288、前 3 层 dense、FP8 block 128×128 | batch 1、prompt 4096 的 prefill projection 可表示为 M=4096、K=6144、N=12288。真实 dense MLP 矩阵，规模比现有 8 MiB 输入更适合大层测量。4096-token prefill 取自论文评估范围；模型和权重不同于论文，不构成复现 |
| 替代：DeepSeek-V4-Flash-Base layer 0 expert 10 `layers.0.ffn.experts.10.w1.weight` | 已有 artifact 确认 F8_E4M3 `[2048,4096]`、8 MiB；256 routed experts，每 token top-6 | 与上游和现有工具链最直接对应，但只对应动态路由的一个 expert；是否被实际 token 选中取决于 router |

R1 已提取 GLM layer 0 的 `gate_proj`，不代表整个 MLP；scale 18 KiB 随 artifact
保留。checkpoint 使用 `weight_scale_inv`，提取器支持显式指定 companion 名称。
目标 prefill GEMM 中，完整矩阵作用于
M=4096 个 token；具体融合 kernel 对权重 tile 的复用/重复解码次数尚未确定，不能
据此声称每个 tile 在真实 GEMM 中只解码一次。独立 LDS 丢弃测量仅回答解码代理阶段。

用户已确认 GLM workload。R1 输入目录为
`inputs/lds/glm52-layer0-gate-proj/`（本地 ignored 数据）；解码 artifact 计划写入其
`mi350x/` 子目录。Slurm job 109840 在 tn1-004-001 完成（0:0，4 秒）；stdout/stderr 为
`results/slurm-lds-r1-glm-mi350x-109840.out/.err`，完整 GPU 结果位于
`results/profiling/lds-r1-mi350x-109840/prepare.json`。LDS 校验通过，artifact SHA-256
为 `4dac4637306d18f6a8fdf0873bd0613819aef39e8ca8231e22165fbd7f42f900`，压缩包
65,435,856 bytes，ANS metadata 2,433,568 bytes。prepare 阶段的 3 次短计时不是 R2
正式吞吐结果。本轮仍只用原生 FP8，不用 4-bit。

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

2026-10-11 使用 sacct 核实：两个历史测量作业均 COMPLETED，ExitCode 0:0。
共享路径可从各节点访问，无需复制。R1 的 MI350X 验证作业 109840 已完成；R2 的
MI300X job 109848 已完成，MI350X job 109847 仍因 GPU 资源占用而 PENDING (Resources)。
R2 stdout/stderr 路径分别为 `results/slurm-lds-r2-glm-mi350x-%j.out/.err` 和
`results/slurm-lds-r2-glm-mi300x-%j.out/.err`；结果位于
`results/profiling/lds-r2-glm-mi300x-109848/summary.json`，MI350X 结果待作业返回。
R3 MI300X profiling job 109852 已提交，目前 PENDING (Resources)，计划输出至
`results/profiling/lds-r3-glm-mi300x-109852/`。MI350X profiling job 109853 已提交，
依赖 R2 job 109847 成功完成后运行，计划输出至
`results/profiling/lds-r3-glm-mi350x-109853/`。两者插桩计时只用于诊断；R3 不把
replicas 当作主要场景。

MI300X R2 的主要配置为一个 72 MiB 权重（replicas=1）、128 threads、grid
multiplier 4、一个 stream、10 warmup / 50 timed iterations；median kernel 时间为
0.273152 ms，按解码后 75,497,472 bytes 计算有效 LDS 写入速率为 276.394 GB/s。
匹配的 raw→LDS 控制为 0.148724 ms / 507.635 GB/s。两者都只计 decoder 或 raw
写入 LDS 的 kernel，不包含 GEMM、H2D、ANS 查表准备；scale 未被消费，LDS 内容在
测量后丢弃。replica 结果仅作为辅助供给/缓存诊断：4、16、64 份分别为 303.605、
258.781、259.847 GB/s；它们不是实际模型参数数或 batch。详见上述 `summary.json`。

| 机器 | 作业 | 状态 / 耗时 | 正式结果 / 报告 |
|---|---|---|---|
| MI300X / gfx942 | 109848 | COMPLETED (0:0)，14 秒，tn1-003-002 | `results/profiling/lds-r2-glm-mi300x-109848/summary.json`；主测量 decode→LDS 276.394 GB/s，raw→LDS 507.635 GB/s |
| MI350X / gfx950 | 109847 | PENDING (Resources) | 排队中；stdout/stderr 为 `results/slurm-lds-r2-glm-mi350x-109847.out/.err` |
| MI300X / gfx942 | 109852 | PENDING (Resources) | R3 counters / timeline；stdout/stderr 为 `results/slurm-lds-r3-glm-mi300x-109852.out/.err` |
| MI350X / gfx950 | 109853 | PENDING (Dependency) | R3 counters / timeline，afterok:109847；stdout/stderr 为 `results/slurm-lds-r3-glm-mi350x-109853.out/.err` |
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
