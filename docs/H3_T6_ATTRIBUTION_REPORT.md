# H3 T6 — decoder bottleneck attribution and next steps

Status: T1–T6 evidence collection is complete for the current decode-to-LDS
stage. The report uses only current hipANS runs and the bench-only ablations;
no hipANS source or codec algorithm was changed in these tasks.

## What was measured

The reference workload is one native GLM-5.2-FP8 layer-0 `gate_proj` tensor
(72 MiB decoded, 62.4 MiB ANS archive). Its R2 `effective_LDS_GBps` is decoded
bytes written to LDS divided by the median HIP-event kernel time. It is an
effective output rate, not physical HBM bandwidth. H2D setup, table preparation,
and verification readback are outside that timing. The A1/A2/A2c/A3 kernels
are isolated `bench/` experiments: **all are INVALID as product-code
performance**, even where an output check passed. Their differences are
diagnostic bounds, not a shipped speedup.

T4 crossed 112 thread/grid/stream combinations on each GPU using the single
real matrix. T5 used nine distinct native GLM MLP matrices (layers 0–2 ×
gate/up/down), concatenated at tile-aligned byte boundaries and jointly
encoded into one p10 ANS artifact. The T5 composite is a decoder byte workload,
not one logical model matrix and not an ANS+GEMM execution. Its unique raw
working set is 648 MiB and compressed archive is 561.2 MiB, both larger than
the nominal 256 MiB MALL capacity. That is a capacity-class 1b-style test, not
proof that every request missed MALL; no direct MALL hit counter was found.

## A0–A5 scorecard

| Probe | MI300X | MI350X | What it says (and does not say) |
|---|---:|---:|---|
| A0 current hipANS, one-matrix R2 | 276.4 GB/s | 451.9 GB/s | Correctness-verified reference; effective LDS output rate. |
| A1 remove iteration barrier (INVALID) | 326.8 GB/s (+18.2%) | 533.7 GB/s (+18.1%) | A sizeable synchronization opportunity in this copied bench kernel; not yet a product-code result. |
| A2 stage compressed words in LDS (INVALID) | 178.3 GB/s | 517.2 GB/s | Staging raises LDS use to 38,208 B/WG and reduces resident WGs/CU from 3→1 on MI300X and 8→4 on MI350X. |
| A2c same extra LDS, retain global refills (INVALID) | 115.5 GB/s | 307.0 GB/s | Against A2, staging is +54.4% / +68.5%; the comparison controls for the larger LDS footprint, but both variants remain invalid ablations. |
| A3 barrier removal + staging (INVALID) | 183.0 GB/s (−33.8% vs A0) | 549.2 GB/s (+21.5% vs A0) | Effects interact with resource occupancy; do not add A1 and A2 gains. |
| A4 best single-matrix deployment | 275.3 GB/s, 128 threads / grid×2 / 1 stream | 540.0 GB/s, 384 threads / grid×1 / 1 stream | Versus the same sweep's 128/grid×4/1 baseline, +0.5% and +19.6%. MI300X has no meaningful launch headroom in this tested envelope; MI350X is materially launch-sensitive. |
| A5 raw→LDS control (R2) | 507.6 GB/s | 729.2 GB/s | Loose no-ANS control envelope: raw/current is 1.84× / 1.61×. It is not a pure arithmetic bound because the raw kernel has different resource use and occupancy. |

The separate T5 large-working-set result, using the same archive on each GPU
and the T4-selected configuration, was 263.2 GB/s on MI300X and 486.3 GB/s on
MI350X. Relative to the T4 single-matrix best, this is −4.4% / −10.0%. The
MI300X is close to its cache-sized result; MI350X loses more throughput as the
working set grows. This supports a working-set effect but cannot quantify MALL
hits without a direct counter.

## Bottleneck attribution

| Factor | Evidence | Estimated headroom | Proposed next-month work |
|---|---|---|---|
| Serial refill/dependency chain | MI300X R3 analysis attributes 78.3% of aggregated wave cycles to dependency wait; vL1D reports 57% stalled on L2 data. The hot loop has a dependent `global_load_ushort` followed by `s_waitcnt vmcnt(0)`. A2 vs A2c differs by +54.4% / +68.5%, but A2's absolute MI300X result falls below A0 after its occupancy loss. | Potential is substantial but not isolated as a valid product speedup. A2/A2c is an INVALID upper-bound probe; the realized result depends on recovering occupancy. | Prototype a bounded per-wave prefetch/window in a separate branch; compare register-window and small LDS-ring designs. Preserve A2c-style resource controls and recheck every output byte. |
| Workgroup synchronization | Removing the per-symbol barrier in A1 raises the bench-only result about 18% on both GPUs. The two waves in a workgroup decode independent tiles; source-level barrier cost is not separately exposed by rocprof counters. | Approximately +18% is the observed isolated upper bound for this barrier in the current test; not additive with refill changes and not yet validated in hipANS. | Test barrier removal in product code only after auditing all tail-wave/table/output paths; run native encoder/decoder roundtrip and sanitizer/correctness coverage. |
| Launch geometry and resource occupancy | T4 best is 128 threads, 6 theoretical resident waves/CU on MI300X, versus 384 threads, 18 theoretical resident waves/CU on MI350X. The MI350X best gains 19.6% over its matched default; MI300X gains only 0.5%. | Up to +19.6% was observed for MI350X in this one-matrix launch envelope; no meaningful MI300X gain was established. These are geometry choices, not algorithmic speedups. | Keep per-architecture launch defaults. Retest the winning geometries on the large T5 working set before adopting them for an HBM-fed path. |
| Compute / decode work | R3 reports VALU utilization 13.84% / 26.60% and VMEM utilization 0.72% / 1.39% (MI300X / MI350X); no MFMA is in this decode-only kernel. Raw→LDS controls are 507.6 / 729.2 GB/s. | The raw control gives only a loose 1.84× / 1.61× envelope versus A0; it includes different occupancy and does not isolate ANS arithmetic. Low VALU is not itself a speedup percentage. | After removing avoidable waits, inspect instruction mix and test symbol/output packing. Do not spend the first optimization cycle on instruction-count reduction before the dependency path is addressed. |
| HBM and cache supply | R3 L2 hit rate is 8.67% / 5.86%; L2-Fabric read is 262.75 / 443.47 GB/s. MI350X's raw HBM request cross-check yields 443.424 GB/s, resolving the `Gbps` label as bytes/s for this metric. MI300X BabelStream Triad is 4,102.7 GB/s; 262.75/4,102.7 ≈ 6.4%. T5 compressed input is 588.5 MB and its measured archive-read rate is 227.9 / 421.2 GB/s. | MI300X evidence rules out approaching its measured Triad limit in this kernel, but this is not a directly interchangeable read-only roofline. MI350X BabelStream Triad was not measured, so no utilization claim is made there. MALL headroom is not numerically measurable from current counters. | Add direct supported memory-byte metrics if available; otherwise retain the T5 capacity-class result and label it as such. Measure MI350X Triad before quoting a peak-relative percentage. |
| XCD locality / NUMA | Profiler attributes remote reads as 0% on MI300X and 0.01% on MI350X in selected R3 dispatches. No direct MALL/L3 hit counter appeared in either architecture's available list. Host NUMA placement was not varied. | No material remote-read penalty is visible in the sampled dispatch; host NUMA and per-XCD locality headroom remain unmeasured. | Do not prioritize XCD tuning yet. If later counters permit, pin host memory/process placement and compare local/remote traffic under a controlled multi-matrix run. |
| Pipeline and GEMM consumer | R3 timeline has two H2D setup copies (1.501 ms / 1.257 ms), a table kernel, then decoder dispatches; no GEMM or D2H. H2D and table setup are outside R2 event timing. | No end-to-end overlap or fused-consumer headroom can be estimated from decode-only LDS tests. | Build the next-stage ANS→LDS→MFMA path with a real activation, and measure the complete pipeline (transfer, decode, GEMM, synchronization) plus overlap—not only decode-only throughput. |

## Interpretation

The leading explanation is **latency-sensitive scheduling and memory-dependency
behavior, amplified by architecture-specific resource use**, not a demonstrated
fundamental ANS decoding limit. The strongest evidence is the large dependency
wait share, the dependent refill in the ISA, the similar A1 improvement on
both GPUs, and the strong MI350X launch-geometry effect. HBM is not close to
MI300X's measured Triad rate, while the large working set only modestly changes
MI300X throughput; this makes raw HBM capacity less likely to be the primary
limit than the serial request/response chain. The A2 controls also show that
simply staging more bytes in LDS can lose on MI300X when occupancy collapses.

This is a reasoned attribution, not a final causal proof: ablations are
bench-only and invalid as product performance, dependency-wait counters are
not source-level stall attribution, MALL hits are unavailable, and T5 does not
run GEMM. Headroom estimates above are isolated envelopes and **must not be
summed**. The T4 occupancy API's theoretical resident-wave count is also not
the same metric as rocprof-compute's time-averaged `Wavefront Occupancy`
(15.42% / 35.03% in R3).

## Evidence files

- [T2 ablations](H3_ABLATION_T2.md) and [T2 results](../results/h3_ablation_t2.json)
- [T3 MI350X profiler/unit report](H3_T3_PROFILER_AND_UNITS.md) and [MI300X follow-up](H3_T3_MI300X_FOLLOWUP.md)
- [T4 MI300X report](H3_T4_SWEEP_MI300X_110403.md), [T4 MI350X report](H3_T4_SWEEP_MI350X_110404.md)
- [T5 artifact](H3_T5_ARTIFACT_110410.md), [MI300X measurement](H3_T5_MEASURE_MI300X_110412.md), [MI350X measurement](H3_T5_MEASURE_MI350X_110411.md)
- [R2/R3 GLM profiling report](GLM_LDS_PROFILING_REPORT.md)

The 72 MiB R2 artifact, profiler replay timing, and 648 MiB T5 composite are
different working sets and should remain separate rows in future dashboards.
