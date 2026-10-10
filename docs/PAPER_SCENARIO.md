# 论文场景核对与 AMD 项目边界

依据：用户提供的本地 ISCA 2026 PDF，
`../2026-Tan et al.-Approaching Shannon Bound with Lossless LLM Weight Compression.pdf`。
SHA-256：`448e3712b35bee7e45b03f287c6dfad52f44ef0dd8ec789ab1fdb3e50e0a6a9a`。
以下页码是 PDF 文件页序，不是正文印刷页码。PDF 保留在共享路径，未纳入 Git。

## 实际执行路径

Section IV 和 V（PDF 第 5–7 页）支持用户描述的两段流程：

1. 离线统计一个 layer/tensor 的权重分布，生成共享 ANS codebook；按 GEMM tile
   几何分割权重，分别编码 tile 并准备 offset、初始化 state 等寻址/解码信息。
2. 推理时，压缩权重与元数据已位于 GPU memory。解码 producer 从 global memory
   读压缩 tile，直接恢复到 shared-memory tile；同一 thread block 的 consumer
   warps 用该 tile 进行 GEMM。Ping/pong 缓冲重叠下一 tile 的解码与当前 tile 的计算。

Algorithm V.1 描述融合 producer/consumer；Algorithm V.2 描述每 lane 独立 state
的 warp-cooperative rANS；Algorithm V.3 描述 shared-memory GEMM。论文还讨论
swizzle/reuse，以及必要时将 evicted tile 暂存到小 decompression buffer 的策略。
因此，核心热路径没有完整权重的 global-memory 解压输出，但不能笼统说整个系统
永远不产生任何 global 临时 buffer。

AMD 上对应 shared memory 的资源是 LDS。LDS 属于 workgroup 生命周期：
独立 decode kernel 返回后，另一个 GEMM kernel 不能直接消费它的 LDS。
第二阶段必须让解码与 GEMM 在同一个融合 kernel/workgroup 中协作。

## 模型与格式

Section VI-A（PDF 第 8 页）明确列出：

| 模型 | 论文采用的架构类别 |
|---|---|
| Qwen2-1.5B、Mistral-7B-v0.3、Qwen-14B、DeepSeek-LLM-67B、Llama-3.1-405B | dense transformer |
| Mixtral-8x22B，约 176B 总参数 | MoE；端到端实验使用 expert parallelism |

同一节列出 BF16、FP8-E5M2、INT8、FP4-E2M1、INT4、SmoothQuant SQ8、AWQ4。
Section II/VI-B 的 entropy/compression 实验也覆盖这些格式。
因此，“模型全部 dense”和“数据全部 BF16”均不成立。但正文没有明确标注每一幅
GEMM 性能图的完整 dtype、scale、symbolization 和累加配置，不能反向断言每一幅
图都验证了上述全部格式。第 12 页 metadata 分析表使用 BF16，邻近正文也有 FP16
措辞；本项目应记录自己的精确配置，不从这些简写推断所有实验格式。

## 带宽口径需要纠正

Section VI-E（PDF 第 11 页）描述寄存器级解码吞吐处于 TB/s 量级，主要图表报告
融合 GEMM 的吞吐和相对加速。对本地 PDF 正文的搜索没有找到明确的“30%”
带宽分子、分母或单独的 LDS 写入带宽计数。因此：

- 解码到 shared memory 是明确的论文场景。
- 若“30%”来自另外的基准或交流，需要补充其原始定义，才可判断它是输出速率除
  以 HBM peak，还是 LDS 实际流量除以 LDS peak，或其他比例。
- 当前 637.57 GB/s 是 demo 向 global memory 输出的 byte throughput；15.54%
  使用 BabelStream HBM Triad 分母，不能被当作论文场景的实测结果。
- 第一阶段首先报告 decoded bytes/s、weights/s、实际 HBM/LDS counters 和
  raw-weight-to-LDS 对照；不把某个未经定义的百分比作为优化目标。

## 建议的格式推进顺序

首个 LDS 验收路径使用有真实来源的 FP8 权重，记录 E4M3/E5M2、FNUZ/OCP 和
scale。FP8 的 256-symbol 字母表便于复用现有 byte-codec 的状态机知识，但须改为
预压缩 artifact、layer codebook 与计算 tile 的数据组织。

MI300X/CDNA3 的原生 FP8 是 FNUZ（E4M3/E5M2）；OCP 与 FNUZ 数值编码不能直接
混用。参考 [AMD precision support](https://rocm.docs.amd.com/en/docs-7.14.0/reference/precision-support.html)
和 [HIP FP8 types](https://rocmdocs.amd.com/projects/HIP/en/develop/reference/low_fp_types.html)。
若源 tensor 是 BF16，转换成 FP8 是离线量化步骤；ANS 的 lossless 验证对象是转换
后的 FP8 bit pattern，不能声称它与原 BF16 模型数值相同。

BF16 可作为后续真实层的正确性/性能路径，但需要明确：原生 16-bit symbol codec，
还是可逆 byte-plane 等变换。后者可以逐位恢复 BF16，却有不同的 codebook 和
压缩分布，不应混称为论文的原生 BF16-symbol 方案。

最终 4-bit 目标必须选定 INT4/AWQ/FP4 等具体格式。分别记录 packed weight、
scale/zero point、解码后的 packed nibble 和 GEMM 真正消费的展开格式。
若 LDS 消费端使用 BF16，4-bit 到 BF16 的 unpack/dequantization 应纳入 runtime
计时，而不能只报告 entropy-symbol 解码速率。FP8/BF16 阶段结果不能直接预测
4-bit 的最终性能。
