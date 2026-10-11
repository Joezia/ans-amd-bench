# 当前 hipANS → LDS：S1 baseline

状态：新 GPU 测量待返回。旧 scalar baseline 结果已撤销并覆盖，不作为对照。
目前不能提供新的有效吞吐数值或瓶颈结论。

## 实现与负载

- 实际 hipANS：clean `8cc8bc263bc943557682a5cc675a045cb9a86412`。
- 离线编码调用原生 `ansEncodeBatchPointer`，默认 probability bits=10。
- 解码直接实例化原有 `ANSDecodeWarpFullBlock`，包含 `decodeOneWarp` 的原有
  state 更新、16-bit refill、ballot/prefix 与同步；没有替代 rANS 实现。
- 必要适配：LDS Writer、实际 LDS 验证读回，以及部署参数/独立地址 workload 分配。
- 默认 128 threads 保留原有部署；64–448 threads 属于部署 sweep。
  一个 wave 的原有 barrier 可被编译器消除，必须将此编译效果如实列为部署差异。
- 真实负载：DeepSeek-V4-Flash-Base 的 `layers.0.ffn.experts.10.w1.weight`，
  2048×4096，native F8_E4M3，8 MiB；scale 2 KiB，仅保存不计算。

## 已完成证据

out-of-tree 编译通过。14 个 timed discard kernel 的 ISA 检查通过：
均存在 `ds_write_b8`，均没有 FLAT/GLOBAL/BUFFER store。
此检查只证明计时路径保留 LDS 写入，不替代 GPU 字节验证或性能测量。

## 待返回测量

Slurm job `109618`：原生 artifact 准备和实际 LDS roundtrip，70 项边界验证；
workload 扩至 2 GiB；workgroup/grid/并发 sweep；选定部署三次复测；
工作量平台与匹配 raw→LDS 测量。结果只描述测试范围内的确认最大值，
不声称数学上的极限。平台未形成时必须继续扩大/分析，不能宣称资源已充分利用。

实际 bytes / 所有队列共同完成时间是有效吞吐；填充和尾部 dummy 不计入分子。
扩大工作量使用同一真实矩阵的独立设备地址副本，不模拟更多真实模型参数。
多队列不相加各自 GB/s。插桩运行不用于正式吞吐。

运行说明：[LDS_CLI.md](LDS_CLI.md)。状态：[project/STATUS.md](project/STATUS.md)。
