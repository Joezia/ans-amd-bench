# 独立交付：真实权重离线 artifact 与可验证 LDS baseline

本交付的旧 baseline 验收已撤销，错误结果由新 hipANS 结果覆盖，不作为参考。
下述输入仍作为工程验证负载保留；H0–H3 实际 hipANS 诊断已经返回。
当前交付转为 [真实场景 baseline](DELIVERY_REAL_SCENARIO_BASELINE.md) 的 R0–R3，
不再把该小矩阵或它的副本集合作为主要场景验收。

对应战略 S0 / S1；2026-10-10 启动。本次不涉及 4-bit，也不包含 GEMM 融合
或 decoder 性能优化。该交付执行时上游两个仓库均未修改，旧 `ans_bench` 接口
保持不变。2026-10-11 起用户允许后续优化修改 hipANS；entropyflow-lab 仍只读。

## 工作负载选择

主负载为官方 `deepseek-ai/DeepSeek-V4-Flash-Base`，固定 revision
`8855555deef230a27a21a8d6f294b7b7497759b6`，使用完整
`layers.0.ffn.experts.10.w1.weight` 及其 `.scale`。上游
`entropyflow-lab/configs/models.yaml` 的 `dsv4-flash-base` 和
`characterization-m2-v2.parquet` 已包含同一 tensor（8,388,608 元素）。
选择理由：原生 FP8 而非模拟转换、真实 MoE 专家计算矩阵、已有上游熵/压缩性
结果可对照；只提取目标 tensor 与 scale，不下载整个 checkpoint。

辅助候选为上游 T3 的官方 Qwen3.8-27B BF16 矩阵，用于后续 dense/格式对照。
它不阻塞此次 FP8 独立交付。Gemma E4B smoke 也可复用，但模拟 FP8 引入量化
变量且可能需要 gated 访问，因此不作为首选。上游 native FP4/MXFP4 不进入本轮。

## 交付节点与验收

| 节点 | 任务描述 | 成功标准 |
|---|---|---|
| D0 | 确认与上游对应的真实负载 | 来源、格式、场景边界清楚，排除本轮不使用的格式 |
| D1 | 提供可重复使用的离线压缩权重包 | 有可追溯来源、完整性证明和真实存储开销，不依赖运行时重新压缩 |
| D2 | 建立可验证的片上解码 baseline | 完整真实权重可正确恢复；性能路径和正确性路径的边界明确 |
| D3 | 形成 AMD 首轮测量证据包 | 结果可复查、对照条件清楚、局限显式记录，不夸大成完整推理性能 |

成功意味着有可复现的未优化 baseline，不意味着已经达到目标性能。
FP8 保留 checkpoint 编码；本轮没有浮点计算，后续 MI300X GEMM 的 FNUZ 兼容性
需另行处理。scale 持久化并计入 artifact，但本轮 decoder 不执行反量化。

## 边界与下一交付

8 MiB 单矩阵是工程验证输入。后续工作集由实际计算需求决定，不强制扩展到
大于 256 MiB，也不以跨越缓存容量作为贴近真实推理的证明。下一交付先完成
场景 baseline 与瓶颈归因，再独立规划优化及 S2 ANS+GEMM。
具体实现、命令、job 状态和验收证据记录在 STATUS 和 LDS 使用说明中；
本计划只记录任务布局，不记录工程实施步骤。
