# 第一阶段：真实预压缩权重解码到 LDS

这是第一阶段工程验收说明，服务于 [PLAN.md](project/PLAN.md) 的 S0/S1。
现有 `ans_bench` 仍用于历史全局输出基线；已新增原生 FP8 的未优化 LDS 参考
kernel 与离线 artifact，接口和已实现边界见 [LDS_CLI.md](LDS_CLI.md)。
完整第一阶段的多工作负载、large-working-set profiling 和优化尚未完成。

## 首个可独立验收的结果

加载一个有来源和数值配置记录的真实权重 tensor，其压缩 tile、每层 codebook、
offset/state、必要的 scale 元数据已离线持久化。GPU 的计时区域只执行按 tile
解码到 LDS，输出在 tile 生命周期结束后丢弃。验证模式从 LDS 读回完整 tile，
导出到 global output 并逐位与参考 tensor 比较。验证模式不参与性能结论。

首轮以 FP8 为工作格式，保留 BF16 的转换来源或 native-FP8 checkpoint provenance；
只有合成数据时可验收工程 smoke test，真实工作负载性能仍标为未测。

## 输入 artifact

离线准备工具与推理 benchmark 分开运行。artifact 至少记录：

- 模型/checkpoint revision、tensor name、原 shape 和内容 hash。
- 原始格式、ANS 符号格式、LDS 消费格式及每 weight 的字节数。
- 精确 FP8/低比特编码、量化方法和 scale/zero-point 的内容或文件 hash。
- 每层共享的 normalized probabilities/decode table 与 probability precision。
- GEMM-compatible tile dimensions、padding、layout/swizzle 和 tile 遍历顺序。
- 每 tile/substream 的 offset、长度、初始 state、符号数量和真实压缩总字节数。
- 负责正确性比较的原始 weight bit patterns，存于 local-only input 路径。

现有 hipANS archive 带每 tile 自己的概率表；不能仅将现有输出地址改为 LDS 就宣称
已经具有 layer codebook 或真实计算 tile 布局。准备工具必须明确这些差异，且基于
decoder 验证实现需要的 state/metadata，不只使用论文 offset-table 的简化大小。

## Kernel 与验证要求

- codebook/decode table 在计时前准备；不在每次 benchmark 启动重新编码全部权重。
  分别测 resident-table steady state 和包含 codebook staging 的完整 decode kernel。
- kernel 按工作 tile 从 global memory 读压缩流，在 workgroup 的 LDS 中恢复 tile。
  使用本仓库的 device helpers/kernel；上游 `HIPANS_ROOT` 继续只读。
- timed discard 模式必须保留真实 LDS stores。使用合适的 volatile/可观察消费方式，
  并检查生成 ISA 中的 `ds_write` 以及 profiler 的 LDS 操作数量，防止编译器删掉
  无用解码/输出。sink/checksum 若处于计时区域，必须记录并用对照测量其开销。
- 验证模式读取实际 LDS 内容再导出，不能用并行的 global store 代替 LDS 验证。
- 正确处理尾 tile、padding、多个 substream 和不同压缩长度；记录 LDS 字节、
  VGPR/SGPR、workgroup threads、wave/substream 映射。
- 同步与 table 访问先以正确性为准。优化用受控 A/B 比较，不依据旧 global-output
  kernel 的等待比例认定新路径的 barrier 成本。
- 与 raw-weight-to-LDS kernel 对照，保持相同 tensor、tile、布局、访问顺序、LDS
  stores 和验证/sink 配置。另行定义纯 LDS store/control microbenchmark。

## 首次测量与 profiling

报告 HIP-event latency 分布、实际解码次数、decoded tile bytes/s、weights/s、
compressed logical bytes/s，以及单独记录的 HBM read/write 和 LDS 流量。
若一个 weight 因 GEMM tile 的访问顺序被多次解码，输出 numerator 按实际生产量
计算，并同时给出 unique-weight count，不能用整层唯一字节数替代实际工作量。

分别准备可复用 cache 数据和大于 MALL 的真实有效工作集，记录缓存状态；反复
解码一小 tile 或重复合成同一内容不自动代表 cold inference streaming。

rocprofv3 检查 dispatch、host/device copies 和 kernel timeline；rocprof-compute
检查 dependency/issue wait、VALU、occupancy、LDS bandwidth/bank conflicts、L2
与 HBM traffic、remote traffic。counter 不可用项仍显式记录。
首轮只定位新路径的限制；之后逐项测试同步范围、独立 substream 数量、tile/layout、
memory request 行为和 table staging，并报告正确性、重复测量及硬件证据。

## 进入第二阶段的条件

真实 tensor artifact、LDS kernel、逐位验证、store-retention 证据和未优化 profiling
必须先通过。随后在同一个 kernel/workgroup 内增加 GEMM consumer，逐步建立
双缓冲和 swizzled tile layout。decode-only 的最优布局未必适合 MFMA；第二阶段
以同 dtype、shape、activation 与 accumulation 条件的 GEMM latency/throughput
为最终判断标准。
