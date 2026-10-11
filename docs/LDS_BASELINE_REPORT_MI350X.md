# mi350x 当前 hipANS → LDS：S1 baseline

Slurm job 109666，真实 hipANS 原生 archive 与实际 LDS 验证通过。
错误 scalar baseline 已覆盖，不作为参考。

## 结果

测试范围内重复确认的最佳有效吞吐：**509.475 GB/s**。
threads=384，grid multiplier=8，
streams=4，replicas=256（2 GiB 有效解码产出）。
三次独立确认：[509.5136577, 509.3853751, 509.4752411] GB/s；
spread=0.03%。
最大两档工作量差异=2.33%；
5% 平台验收=True。
平台或重复性不达标时，不能宣称最大资源供给已充分饱和。
匹配 raw→LDS：669.924 GB/s，
不是解码吞吐，也不是 LDS 理论峰值。

| 测量 | replicas | 有效 GB/s | median ms |
|---|---|---|---|
| workload | 1 | 170.568 | 0.049180 |
| workload | 4 | 466.027 | 0.072001 |
| workload | 16 | 567.465 | 0.236521 |
| workload | 64 | 475.179 | 1.129828 |
| workload | 128 | 481.362 | 2.230635 |
| workload | 256 | 484.203 | 4.435092 |
| selected-workload | 64 | 480.229 | 1.117947 |
| selected-workload | 128 | 497.938 | 2.156374 |
| selected-workload | 256 | 509.536 | 4.214590 |
| confirmation | 256 | 509.514 | 4.214772 |
| confirmation | 256 | 509.385 | 4.215833 |
| confirmation | 256 | 509.475 | 4.215089 |
| raw-matched | 256 | 669.924 | 3.205563 |

## 实现与范围

- hipANS revision=8cc8bc263bc943557682a5cc675a045cb9a86412，dirty=false。
- native ansEncodeBatchPointer；原有 ANSDecodeWarpFullBlock/decodeOneWarp，
  state recurrence、16-bit refill、ballot/prefix 与同步没有改写。
- 必要适配仅为 LDS Writer、验证与部署循环。默认 128 threads；
  部署 sweep 包含单 wave，编译器可消除单 wave barrier，未修改原有代码。
- 真实 DeepSeek-V4-Flash-Base 原生 FP8 专家矩阵 2048×4096，8 MiB；
  scale 保存但不消费，无 GEMM、无 4-bit。
- 独立地址副本检查资源饱和，不代表新增真实模型参数或冷模型 streaming。
- 分子为真实未填充 bytes × replicas；分母为多队列共同完成的 HIP-event 时间。
  table 准备、H2D、验证在计时外，kernel 内 LUT staging 与 LDS completion 在计时内。
- 70 项 GPU 边界验证通过，14 个 timed kernel 保留 ds_write_b8，无全局输出。
- launch 理论资源不等于实测 occupancy；测试范围的最大值不等于数学极限。
- Profiler 插桩/重放会改变调度，正式吞吐只使用未插桩记录。
  counters/timeline 可用项见 lds_baseline_mi350x.json；硬件归因尚待分析。

证据目录：results/profiling/hipans-lds-mi350x-109666。所有部署案例、失败 stderr、hash 与 provenance 存于正式 JSON。

## 单队列资源证据（不是上述最佳多队列配置）

仅分析默认 128 threads、grid multiplier=4、streams=1 的末次 decoder dispatch；
不将重放 counters 当作未插桩吞吐，也不把这些值外推到最佳 384-thread/4-queue 配置。

| Metric ID | 指标 | Avg | 单位 | Percent of Peak |
|---|---|---|---|---|
| 2.1.8 | Active CUs (deprecated) | 252.0 | CUs | 98.44 |
| 2.1.10 | VALU Utilization | 25.96 | Percent | N/A |
| 2.1.16 | Wavefront Occupancy | 3731.13 | Wavefronts | 45.55 |
| 2.1.21 | L2 Cache Hit Rate | 1.71 | Percent | N/A |
| 6.2.8 | Insufficient CU LDS | 62.15 | Percent | — |
| 7.2.4 | Dependency Wait Cycles | 29588912000.00 | Cycles per kernel | — |
| 17.6.6 | Read Bandwidth - HBM | 431.42 | Gbps | — |
| 17.6.12 | Write Bandwidth - HBM | 0.00 | Gbps | — |

工具报告的 L2–HBM interface bandwidth 保留原单位标签，不等同于独立实测 HBM pin 峰值。
Active CUs 是工具标记 deprecated 的指标；occupancy 及调度拒绝比例不是算法上限的证明。
