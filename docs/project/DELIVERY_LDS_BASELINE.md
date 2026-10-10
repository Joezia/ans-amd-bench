# 独立交付：真实权重离线 artifact 与可验证 LDS baseline

对应战略 S0 / S1；2026-10-10 启动。本次不涉及 4-bit，也不包含 GEMM 融合
或 decoder 性能优化。上游两个仓库均只读，旧 `ans_bench` 接口保持不变。

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

## 本次验收

1. 真实 tensor 原始位模式与 scale、checkpoint 元数据和 hash 可查；离线保存
   共享 codebook、计算 tile 的独立 ANS streams、offset/state 和压缩字节统计。
2. 独立 benchmark 只加载现成 artifact；验证模式从实际 LDS 读回并逐位比较。
   性能模式恢复至 LDS 后丢弃；raw-to-LDS 对照使用相同 tile/layout/store。
3. MI300X HIP-event 结果、生成 ISA 的 LDS store 证据及可用 profiler 证据落盘。
   测不出的项目写明缺失；不把 cache-resident 小矩阵解释成 cold HBM streaming。

成功意味着有可复现的未优化 baseline，不意味着已经达到目标性能。
FP8 保留 checkpoint 编码；本轮没有浮点计算，后续 MI300X GEMM 的 FNUZ 兼容性
需另行处理。scale 持久化并计入 artifact，但本轮 decoder 不执行反量化。

## 边界与下一交付

8 MiB 单矩阵是缓存内实验。下一交付扩展到大于 256 MiB 的不同真实矩阵集合，
随后进行 counter-guided A/B 优化；通过后才进入 S2 ANS+GEMM。
具体实现、命令、job 状态和验收证据记录在 STATUS 和 LDS 使用说明中。
