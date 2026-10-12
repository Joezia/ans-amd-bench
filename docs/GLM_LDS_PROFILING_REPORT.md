# GLM FP8 decode-to-LDS: S1 profiling results

Updated: 2026-10-12. This is the GLM real-weight R2/R3 report. The older
[standalone decoder report](PROFILING_REPORT.md) profiles a different demo
kernel and must not be used as evidence for this workload.

## Scope and measurement authority

The workload is `zai-org/GLM-5.2-FP8`, revision
`f33c6dc501ee5a2c7e35155653b1b1abbc320951`, tensor
`model.layers.0.mlp.gate_proj.weight`, native F8_E4M3, shape `[12288, 6144]`
(75,497,472 bytes / 72 MiB). The companion F32 `weight_scale_inv` has shape
`[96,48]` (18 KiB), is preserved in the artifact, but is not consumed by this
decode-only measurement. The selected inference context is batch 1, prefill
4096 tokens (`M=4096, K=6144, N=12288`); no activation GEMM is run.

R2 is the performance authority: uninstrumented HIP-event timing, one tensor
(`replicas=1`), 128 threads, grid multiplier 4, one stream, 10 warmups and 50
timed iterations. Every R2 decode and raw-control run verified the actual LDS
output byte-exactly. The reported `effective_LDS_GBps` is decoded tensor bytes
written to LDS divided by median kernel time; it is not physical HBM bandwidth.
H2D, decode-table preparation, and verification readback are outside that
timing; kernel table staging and decode synchronization are inside it.

| GPU | ANS→LDS median (p10–p90) | Effective LDS output rate | Compressed archive read rate | raw→LDS control rate |
|---|---:|---:|---:|---:|
| MI300X / gfx942 | 0.273152 ms (0.271826–0.275723) | 276.394 GB/s | 239.558 GB/s | 507.635 GB/s |
| MI350X / gfx950 | 0.167082 ms (0.166278–0.169698) | 451.859 GB/s | 391.639 GB/s | 729.155 GB/s |

The MI350X single-tensor rate is about 1.64× the MI300X rate for this selected
deployment. The raw→LDS path is a useful control, not a measurement of pure
ANS arithmetic cost: the kernels have different resource use and occupancy.
Both artifacts have the same total ratio including scales, 1.15372.

Replicas are retained only as auxiliary resource/cache diagnostics. They are
repeated copies of the same tensor at independent device addresses, not model
parameters or batch:

| GPU | 4 copies | 16 copies | 64 copies |
|---|---:|---:|---:|
| MI300X | 303.605 GB/s | 258.781 GB/s | 259.847 GB/s |
| MI350X | 545.239 GB/s | 478.815 GB/s | 482.726 GB/s |

## R3 counter and timeline evidence

R3 used the R2 primary launch geometry and profiled a short replay
(`warmup=2`, `iters=1`, `--skip-verify`). `rocprof-compute` analysis selected
the last decoder dispatch; profiler replay timing is diagnostic only and is
not compared as a throughput result. `--skip-verify` explains why the R3 JSON
does not claim a fresh output verification; the corresponding R2 runs did
verify successfully.

| Counter family | MI300X / gfx942 | MI350X / gfx950 | Interpretation limit |
|---|---:|---:|---|
| VALU utilization | 13.84% | 26.60% | No MFMA work is expected in this decode-only kernel. A low VALU percentage is not directly convertible to speedup headroom. |
| Wavefront occupancy | 15.42% | 35.03% | Tool-reported average relative to its architecture maximum; suggests different residency, not a causal explanation by itself. |
| Active CUs | 255 / 304 (83.88%; deprecated metric) | 221 / 256 (86.33%; deprecated metric) | CU activity is not equivalent to full-CU utilization. |
| VMEM utilization | 0.72% | 1.39% | Does not by itself state achieved HBM bandwidth. |
| vL1D hit rate | 89.70% | 90.09% | Per rocprof-compute selected-dispatch metric. |
| L2 hit rate | 8.67% | 5.85% | Cache hit ratio only; does not supply MALL/Infinity Cache hit rate. |
| L2-Fabric read BW | 262.75 (unit labels conflict) | 444.86 (unit labels conflict) | One report section labels these GB/s, another labels the same values Gbps; do not use as a finalized achieved-bandwidth number. |
| HBM / remote read traffic | 100% / 0% | 100.01% / 0% | Profiler's traffic attribution for this dispatch; not an output-byte throughput metric. |
| Scheduler-pipe stall rate | 16.21% | 15.02% | Broad stall category, not a barrier-only measurement. |
| Dependency wait / issue wait | 644.25M / 7.66M cycles | 741.00M / 63.89M cycles | Aggregated per-kernel counts; not wall-clock fractions and not additive stall attribution. |
| Insufficient-CU-barrier metric | 0% | 0% | Does not establish zero cost for source-level barriers or synchronization. |

The `rocprofv3` trace records two H2D copies before kernel execution (total
copy durations 1.501 ms on MI300X and 1.257 ms on MI350X), one decode-table
kernel, then three decoder dispatches (two warmups and one timed replay) on a
single stream. No D2H copy or GEMM appears. There is therefore no producer /
consumer overlap to evaluate in this stage; these setup copies are outside the
R2 HIP-event timing.

## Attribution status and next evidence

| Factor | Evidence | Headroom estimate | Next action before choosing an optimization |
|---|---|---|---|
| Memory/cache | L2 hits 8.67% / 5.85%; profiler attributes essentially all selected-dispatch L2 read traffic to HBM. No MALL metric was collected. | Not quantifiable: the profile has no consistent achieved-HBM unit and no Infinity Cache hit counter. | Add supported MALL/Infinity Cache and byte-count metrics; reconcile read/write units and compare with the uninstrumented R2 timing. |
| Compute | VALU 13.84% / 26.60%, VMEM 0.72% / 1.39%; no MFMA. | No direct compute-only speedup estimate; these ratios are not idle-cycle fractions. | Inspect the selected ISA and correlate instruction/dependency metrics with a controlled launch sweep on the one-tensor case. |
| Synchronization/dependency | Scheduler-pipe stall 16.21% / 15.02%; dependency waits are large aggregate counts. The profiler does not isolate source barrier cycles. | Not separable from data dependency and memory wait with current data. | Do a barrier/dependency-focused A/B profile before attributing stalls to workgroup barriers. |
| Occupancy/deployment | Wave occupancy 15.42% / 35.03% with 128 threads, grid multiplier 4, one stream; MI350X output rate is 1.64× MI300X. | Correlation only; no defensible occupancy-to-speedup conversion. | Test bounded grid/block/stream variants while keeping the single GLM matrix and correctness fixed; keep replicas in a separate diagnostic table. |
| XCD/locality | Remote read traffic reports 0% on both selected dispatches. | No remote-read penalty is evident in this sample; broader placement effects are unmeasured. | Retain remote/XCD counters in a placement-controlled rerun if supported. |
| Pipeline | Single-stream decode-only trace, two setup H2D copies, no GEMM and no overlap. | Fusion/pipeline headroom is not measurable in S1. | Defer pipeline claims until the next stage has an activation GEMM consumer and a fused or overlapped path. |

The present evidence does **not** demonstrate a fundamental ANS decode limit,
and it is not yet sufficient to prove that architecture-unaware scheduling is
the dominant cause. Low VALU/VMEM activity, low wave occupancy on MI300X, and
low L2 hit ratios make scheduling and data-delivery behavior worth investigating;
the missing MALL counters, inconsistent fabric-BW units, and unisolated
dependency/barrier cost prevent stronger attribution. No kernel or codec
algorithm was changed in S1.

## Tool coverage and raw evidence

On both GPUs, rocprofv3 timeline and rocprof-compute counters completed without
elevated permissions. rocprof-sys/omnitrace was not installed. rocprof-compute
did not generate a roofline result because this LDS harness did not provide a
`roofline.csv`; its printed HBM bandwidth values (5,324.8 GB/s on MI300X and
8,192 GB/s on MI350X) are configured theoretical peaks, not achieved HBM rates.

Raw, ignored job outputs are under:

- `results/profiling/lds-r2-glm-mi300x-109848/`
- `results/profiling/lds-r2-glm-mi350x-109847/`
- `results/profiling/lds-r3-glm-mi300x-109852/rocprof/`
- `results/profiling/lds-r3-glm-mi350x-109853/rocprof/`
- `results/profiling/lds-s1-followup-109854/`

The R2 JSONs record hipANS commit `8cc8bc263bc943557682a5cc675a045cb9a86412`
(clean), ROCm 10.0.0, and architecture-specific build flags. The profiler
analysis text and availability files are retained with the raw outputs above;
this report preserves selected durable values because `results/profiling/` is
git-ignored.
