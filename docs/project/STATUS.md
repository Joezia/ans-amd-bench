# 项目执行状态

最后更新：2026-10-07 CST

战略目标和成功标准见 [PLAN.md](PLAN.md)。本文使用相同的 P0–P5 编号记录实际
进展、证据、阻塞和近期动作。

## 当前摘要

MI300X 标准基线和完整 profiling campaign 已完成，项目从“等待资源”进入“结论
复核与后续实验收口”阶段。Slurm job `107394` 在节点 `tn1-003-002` 上以
`COMPLETED (0:0)` 返回；BabelStream、标准 benchmark、rocprofv3、
rocprof-compute、规模/tile/并发 sweep 和 launch sweep 均成功。rocprof-sys 未安装，
已按设计记录为 unavailable，而没有使 campaign 失败。

标准结果为 637.57 GB/s decoded-output throughput，对同机 BabelStream Triad
4,102.71 GB/s 的利用率为 15.54%。主要证据是约 79.9% wave cycle 处于 dependency
wait、vL1D 等待 L2 data 的中位比例为 70.85%、IPC 仅为峰值的 18.73%，而远端
XCD 流量几乎为零。当前判断是差距主要来自架构适配不足的依赖隐藏、同步、tile
调度和内存行为，而不是 ANS 解码的根本上限，也不是单纯的 HBM 饱和。

## 任务状态

| 编号 | 状态 | 当前结论 | 下一验收点 |
|---|---|---|---|
| P0 | 已完成 | 仓库、历史结果、指标口径和上游接口已经盘点 | 仅在事实基础变化时更新 |
| P1 | 已完成 | 标准 CLI、正确性校验、指标定义和 MI300X/BabelStream 基线均已取得 | GPU smoke-test ROCm 版本 provenance 修复 |
| P2 | 已完成 | timeline、完整 counter 和所有必需诊断维度已有证据；不可用工具已显式记录 | 复测一个异常 sweep 点，并可选补齐两个 launch 组合 |
| P3 | 已完成 | 已形成瓶颈排序、headroom 边界及非根本上限结论 | 下一阶段用受控 A/B 区分 barrier 与数据依赖占比 |
| P4 | 受阻 | 对照框架和论文报告值已记录；六个评估模型输入均缺失 | 获得合法模型输入，或保留明确的缺失状态 |
| P5 | 进行中 | 已按证据提出下一月方向；最终模型级优先级仍依赖 P4 | 将建议转成独立的优化实验计划 |

## 已完成和可验证证据

- P0：`docs/INVENTORY.md` 已记录初始仓库能力、历史结果和 hipANS 解码行为；提交
  `372484e`。
- P1：标准化 JSON benchmark、正确性校验、指标定义和 provenance 已完成；提交
  `36278d4`。正式结果见 `results/standard_mi300.json` 和
  `results/peak_bw.json`。
- P2：profiling/sweep 工作流提交为 `c870059`；job `107394` 已取得 SOL、HBM/L2、
  occupancy、wave wait、VALU、LDS、remote traffic 和 timeline 数据。
- P3：`docs/PROFILING_REPORT.md` 给出归因表；可机器读取的精选计数器与 sweep 结果
  保存在 `results/profiling_summary.json`。
- P4：论文六个模型的对照脚本和显式 `MISSING_INPUT` 表已完成；提交 `ba5c590`。
  当前输出见 `results/paper_reproduction.md`。
- campaign 可靠性修复包括 BabelStream pipefail/SIGPIPE（`fd44111`）以及共享路径下
  的 sbatch 工作目录解析（`f5e5a67`）。
- 上游 hipANS checkout 保持只读且未被修改；所有已提交改动均位于本仓库，未 push。

## 当前活动与阻塞

### 最近完成的任务

- Slurm job：`107394`，名称 `ans-mi300-campaign`。
- 状态：`COMPLETED`，exit code `0:0`，用时 1 分 30 秒。
- 时间：2026-10-06 23:46:40–23:48:10 CST。
- stdout/stderr：`results/slurm-mi300-campaign-107394.out` 和 `.err`。

### 阻塞与风险

- 论文评估模型权重未提供，因此 P4 不能生成真实 AMD 模型行；不会用合成数据替代。
- 192 MiB、64 KiB tile 的 429.68 GB/s 明显偏离相邻结果，需要复测后才能解释。
- 并发结果来自多个独立 process/queue 的 HIP-event 区间求和，时间窗未做全局同步；
  它只证明延迟隐藏潜力，不替代标准单流指标。
- gfx942 分析块没有给出独立的 256 MiB MALL hit-rate 字段；当前以 L2 hit、fabric
  traffic、remote traffic 和跨 256 MiB 的规模 sweep 作为证据边界。
- CLI 原先把 HIP runtime 整数 `71526333` 写成 ROCm 版本；代码已经改为从
  `hipconfig --path` 下的 `.info/version` 获取发行版版本，正式 JSON 已根据同节点
  记录校正为 `10.0.0`。CLI 已重建；interactive smoke job `107492` 因预计等待约
  12 小时而主动取消，GPU 运行验证尚待批处理完成。

## 紧接着的工作

1. 用小输入在 GPU 上验证已重建的 `ans_bench` provenance 修复，不重新执行完整
   campaign。
2. 对异常的 192 MiB/64 KiB tile 点做独立复测；若稳定再追加针对性 counter。
3. 可选构建 64-thread/grid-2 与 grid-8 变体，补齐 launch 全因子表；现有结果已经覆盖
   workgroup size 与 grid multiplier 的主效应。
4. 将当前报告作为 P5 输入，另行拆分 barrier、tile/work-queue、cache traffic 和
   decode-to-GEMM overlap 的下一阶段实验。
5. 获得合法模型输入后重新运行 `scripts/run_paper_reproduction.sh`，完成 P4 AMD 列。

## 阶段变更日志

- 2026-10-06：建立战略计划与战术状态双文档；当前阶段定位为 P1/P2 进行中，
  P3 等待实测证据，P4 因模型输入缺失受阻。
- 2026-10-06：job `107361` 暴露 batch 临时脚本路径问题；修复后以 job `107394`
  重新提交，等待 MI300X 资源。
- 2026-10-07：job `107394` 完整成功；P1、P2、P3 达到验收标准，形成 MI300X
  基线、counter 汇总和瓶颈归因。P4 继续受输入缺失阻塞，P5 开始。
