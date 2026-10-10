# 真实 FP8 权重 → LDS：首个未优化 baseline

独立交付 D0–D3 已完成；这是 S1 的起点，不是优化完成或 ANS+GEMM 实现。
正式证据为 `results/lds_baseline.json`；原始文件位于
`results/profiling/lds-109484/`。运行方式见 [LDS_CLI.md](LDS_CLI.md)。

## 负载与上游对应

只读分析了 entropyflow-lab 的战略计划、11 个模型 registry、M2-v2 parquet、
native/模拟格式实现及 CPU rANS。选择官方 DeepSeek-V4-Flash-Base 的原生 FP8
专家矩阵，避免模拟量化变量，且与上游已有表征逐 tensor 对应。辅助候选为
Qwen3.8-27B BF16 dense 矩阵，尚未提取；此次没有 4-bit。

- 模型：`deepseek-ai/DeepSeek-V4-Flash-Base`，revision
  `8855555deef230a27a21a8d6f294b7b7497759b6`。
- Tensor：`layers.0.ffn.experts.10.w1.weight`，2048×4096 `F8_E4M3`，8 MiB；
  原始 SHA-256 `3438d6447dae9efc3bfa48d1c9a8aa54b8060883f0ab8e0d9219360d809cb94c`。
- 配套 F32 scale：16×32、2 KiB，原样保存，不在本阶段消费。
- 上游同一行：H=6.574583 bits/symbol，CPU `cr_total`=1.158008，
  tile_size=65,536，`bench_ratio=null`。它是 CPU 参考大小，不是 GPU 性能。
- 上游检查时 HEAD 为 `e5a2d3a1072c2544c138142bb5bf4a35641e7207`；其代码、
  cache、结果、已有未提交文件均未修改。

仅以受限 HTTP range 提取目标权重和 scale；没有下载完整 shard/checkpoint。
本地包位于 `inputs/lds/dsv4-flash-base-expert10-w1/`，不提交权重二进制。

## 真实存储与正确性

| 内容 | 字节 |
|---|---:|
| 原始 FP8 权重 | 8,388,608 |
| ANS payload | 6,884,189 |
| Header + 每矩阵概率表 + offset/length/state | 1,049,664 |
| 压缩 artifact（不含 scale） | 7,933,853 |
| 原样 scale | 2,048 |
| 含 scale 的总压缩比 | 1.057304× |

64×128 tile、每 tile 64 条独立 byte-rANS 流；每矩阵一个共享 codebook。
元数据中每 tile 的 64×16-byte descriptors 共占 1 MiB，解释了为什么本包的
真实总压缩比低于上游 65,536-symbol CPU 参考归档。两种大小不能互换。
这是本仓库的 **scalar byte-rANS 参考实现**，不是 hipANS archive decoder 的
已优化移植，也不是整个 transformer 层共用一张表。

11 项 CPU 回归通过；每个 GPU campaign 的 15 项边界验证通过，覆盖常量
frequency=4096、全字母表、多 tile、尾部 padding、单元素与偏斜分布。真实完整
tensor 在 LDS-table/global-LUT/raw 三条路径均从实际 LDS 导出逐位验证。
四个计时 discard kernel 的 ISA 审计均有 `ds_write_b8`，无 FLAT store；正式
decoder counter 为 294,912 LDS instructions，包含 staging、lookup 和 decoded
stores。验证 buffer 的 global 输出不在 timing/profiler decoder dispatch 内。

## MI300X 测量

任务 `109484` 在 `tn1-003-002` COMPLETED (0:0)，用时 68 秒，ROCm 10.0.0，
gfx942。实现提交 `3b52df3`，测量时 worktree clean；hipANS frozen reference 为
`8cc8bc263bc943557682a5cc675a045cb9a86412`，clean，未链接进新 codec。

以下为未插桩 HIP-event、10 warmup / 50 iterations；每 launch 生产一次完整矩阵。
这是 8 MiB cache-fitting 重复工作集的 **预热策略**，不保证所有 cache 命中。

| 路径 | median / p10 / p90 (us) | LDS 产出或 control-store GB/s |
|---|---|---:|
| ANS → LDS，LDS LUT（包含每 workgroup staging） | 89.655 / 89.415 / 89.935 | 93.565 |
| ANS → LDS，cached global LUT | 67.702 / 67.462 / 68.471 | 123.905 |
| 同 tile/layout 的 raw → LDS | 14.261 / 14.097 / 14.502 | 588.220 |
| 纯 LDS store control，不读取权重 | 5.929 / 5.849 / 6.094 | 1,414.844 |

首轮独立 campaign `109479` 得到 93.606 / 123.905 / 587.376 GB/s，主 decoder
与确认轮相差约 0.044%；该轮首次 dispatch counter 不作为正式 warm counter。
新路径和旧合成 byte→global 的 637.57 GB/s 不构成直接性能回归比较。

### 两个 baseline 的含义与差距

两者均用“恢复/搬入 LDS 的 8,388,608 个 FP8 字节 / kernel 时间”计算速率，
不是实测 HBM 带宽，也不是 LDS 峰值带宽或 GEMM 性能。
ANS 路径读取离线压缩流，逐符号查表、更新 rANS state、按需读取补充字节，再写
LDS；raw 路径直接读取未压缩权重并写 LDS，不做解码。两者 tile/layout/stores
相同，但 ANS-LDS-table 路径额外分配并 staging 16 KiB LUT，不能假设驻留条件
也相同。两者计时都不含离线压缩、H2D、正确性读回、scale 处理或 GEMM。

源码中每 lane 的 128 个符号有 state 依赖链；变长补充读取带来分支，64 lanes
读取各自独立 streams，地址不像 raw 路径相邻 lanes 的连续读取。低 occupancy
与较多 dependency wait 支持“当前实现缺乏有效延迟隐藏”的判断，24 KiB LDS
及 insufficient-CU-LDS counter 支持 table footprint 限制调度的判断。尚不能给
每项原因分配精确耗时比例。global-LUT 123.905 GB/s 对照表明移除 LDS-table
组织可改善本负载，但也不能把全部改善认定为 staging 拷贝开销。

因此 6.29× 差距是“未优化解码供给”与“直接搬运供给”的差距，不是单靠压缩
少读数据就应获得加速。每流串行依赖是算法特征，但这些结果不证明整个 GPU 上
的 ANS 吞吐存在这一固定上限，也不能直接推广到缓存外大模型或融合 GEMM。

## AMD profiling：当前证据，不是最终归因

rocprofv3 和 rocprof-compute 均成功。counter 在两次 warmup 后只分析最后
decoder **dispatch 3**，不包含 raw/control、验证 copy 或其他 runtime kernel。
未使用 counter-instrumented HIP-event 延迟作为 baseline。

| factor | evidence（正式 counter 或对照） | estimated headroom | 下一独立实验候选 |
|---|---|---|---|
| Memory bandwidth | L2-fabric read 90.92 GB/s、write 0；vL1D hit 98.58%、L2 hit 40.90%；remote read 0 | 当前没有 HBM 饱和证据；不可从小工作集直接推算 cold streaming 上限 | 不同真实矩阵构成 >256 MiB 工作集，检查有效负载与事务放大 |
| Compute / latency hiding | VALU 6.98%、IPC 7.86% of peak；dependency wait 54,776,320 / 90,906,760 wave cycles = 60.26%；occupancy 4.59% | raw 对照是 6.29× 速率差距，不是承诺的 decoder 加速；wait 百分比不能直接转为墙钟 headroom | 固定 codec 做更高有效驻留与 stream 访存对照 |
| LDS / table footprint | 24 KiB/workgroup、36 VGPR、32 SGPR、scratch 0；insufficient-CU-LDS 36.19%；LDS instructions 294,912，bank conflicts/access 1.00 | global-LUT 对照实测 1.324×；同时改变 footprint/staging/lookup，不能全归因于 staging | 区分 table footprint、随机 lookup 和 byte-store layout 的影响 |
| Synchronization | 实现仅在 LUT staging 后及 tile 完成时同步；insufficient-CU-barriers 0；没有单独 barrier 时间 counter | 尚未量化；不能套用旧 hipANS 的每符号 workgroup barrier 解释 | 在正确消费边界下单独测 wait/barrier 代价 |
| XCD / NUMA | remote read 0%；write=0，因此 remote write 比例 N/A | 本工作集未证明远端流量是主因；没有 placement A/B | 大工作集后再比较部署/地址局部性 |
| Pipeline design | 已在 timing 前离线压缩与上传；discard decoder 不写回 global；尚无 GEMM consumer/双缓冲 | 无端到端或 overlap headroom 估计 | 完成 S1 控制实验后，在 S2 增加同 dtype 的 GEMM 消费者 |

当前优先解释是这个未优化参考路径的 **低驻留、table footprint 和串行依赖/访存**，
而不是已证实的 ANS 基本极限。两个同步点不能证明同步无成本，但原 demo 的
“每符号 barrier”不适用于这里。LDS 理论流量指标 928.19 GB/s 包含查表与 staging，
不能把它当成 93.565 GB/s 的真实权重输出指标。

## 缺失项与下一边界

没有独立 MALL hit-rate、单独 barrier 耗时或新 roofline microbenchmark；分析器
明确记录缺少 `roofline.csv`。L2-fabric counters 虽标记 HBM read traffic 100%，
仍是 fabric 层分类/流量，不是独立 DRAM pin 或 MALL 计数。旧 BabelStream
4,102.71 GB/s 只作为单独 HBM 参考，不据此命名 LDS utilization。

本轮没有发生权限拒绝；零写流量导致 write latency/比例 N/A，MFMA=0 导致相关
表达式 N/A，这是 workload 不适用项而非缺失权限。工具失败的检测/fallback
逻辑保留，availability 和原始日志可查。

此次交付到此收口。下一项是缓存外真实负载与受控性能实验；BF16 dense 辅助对照
可在该阶段加入。FP8 bit-exact 验证不等于 MI300X FNUZ GEMM 数值兼容；融合、
scale 消费与转换需要各自验收。没有自动开始 4-bit 或 GEMM 实现。
