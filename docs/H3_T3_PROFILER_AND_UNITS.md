# H3 T3 — profiler counters, cache visibility, and bandwidth units

Status: MI350X discovery/profile completed (Slurm 110303 and corrected 110307);
the MI300X discovery probe (110302) is still pending for node resources.
Existing R3 counter captures on both GPUs remain usable.
This report does not treat instrumented profiler timing as a performance
result; uninstrumented R2 HIP-event results remain authoritative.

## Scope and available tooling

The workload is the single GLM-5.2-FP8 layer-0 `gate_proj` matrix from the R2
scenario, launched with one stream, 128 threads, grid multiplier 4, and the
R3 short replay (`warmup=2`, `iters=1`, `--skip-verify`). The profile is for
counter attribution only. R2 separately verified output byte-exactly.

| Device | Probe / profile | Tool result | PC sampling | Counter permissions |
|---|---|---|---|---|
| MI350X / gfx950 | T3 Slurm jobs 110303 and 110307 completed, exit 0; existing R3 job 109853 | ROCm 10.0.0, rocprofiler-SDK 1.3.5, rocprofiler-compute 3.8.0; `rocprofv3 --list-avail`, architecture-specific metric/block listing, L2 counter replay, and selected-dispatch analysis succeeded | Available configurations listed: `host_trap` / time (interval 1–1,048,576) and `stochastic` / cycle (interval 256–1,048,576, power-of-two interval). T3 checked availability; it did not run a PC-sampling capture. | No permission failure in the profile logs. |
| MI300X / gfx942 | Existing R3 job 109852; T3 discovery job 110302 pending | `rocprofv3` timeline and `rocprof-compute` counter profile succeeded in R3. New architecture-specific `--list-avail` probe is pending. | Not yet independently verified for gfx942 in this T3 run. | Existing R3 profile did not report a permission failure. |

`rocprofv3 --list-avail` on MI350X exposed L2 `TCC_HIT` / `TCC_MISS` and
HBM-directed `TCC_EA0_RDREQ_DRAM` request counters. No counter name containing
`MALL`, `L3`, or `Infinity` appeared in the captured list. The available TCC
counters are L2 counters, not an Infinity Cache hit counter: AMD documents
TCC as L2 and Infinity Cache as the last-level cache. Thus the current data
cannot directly report MALL hit rate or prove cache residency. [MI300/MI200
counter reference](https://rocm.docs.amd.com/en/docs-10.1.0/reference/gpu-arch/mi300-mi200-performance-counters.html),
[ROCm device-hardware glossary](https://rocm.docs.amd.com/en/latest/reference/glossary/device-hardware.html)

## Counter values from the selected decoder dispatch

| GPU | L2 hit rate | L2-Fabric read (17.1.3) | Interface read (17.2.0, raw label) | HBM read attribution | Remote read | Direct HBM read requests |
|---|---:|---:|---:|---:|---:|---:|
| MI300X | 8.67% | 262.75 GB/s | 262.75 Gbps | 100% | 0% | `TCC_EA0_RDREQ_DRAM_sum` = 524,523; this capture has no width-specific count to independently turn the requests into bytes |
| MI350X | 5.86% | 443.47 GB/s | 443.47 Gbps | 99.99%* | 0.01% | Re-run 110307: `TCC_EA0_RDREQ_DRAM_32B_sum` = 2,101,290 for the last decoder dispatch |

\* A value just above 100% is profiler rounding/aggregation; it is not a
physical fraction greater than 100%. The profiler's HBM-read attribution is
request-count based and is not a direct MALL-hit measurement. The MI350X R3
capture reports zero HBM writes for this decode-only kernel; output is written
to LDS.

### Reconciling `GB/s` and `Gbps`

Do not divide the `17.2.0` result by eight. In this profiler, the metric is
defined as bytes read by L2 from Infinity Fabric divided by duration, even
though the displayed unit for `17.2.0` is `Gbps`; the neighboring `17.1.3`
L2-Fabric metric labels the same numerical throughput `GB/s`. AMD's ROCm 10.0
changelog notes that the section-17.2 bandwidth unit changed to `Gbps`, and
the metric documentation describes it as bytes divided by time. [ROCprofiler-
compute L2 metric definitions](https://rocm.docs.amd.com/projects/rocprofiler-compute/en/docs-10.1.0/conceptual/cdna/l2-cache.html),
[ROCm changelog](https://rocm.docs.amd.com/en/develop/release/changelog.html)

The latest MI350X raw-counter cross-check gives the same byte-rate scale
without relying on the unit label: 2,101,290 32-byte HBM read transactions ×
32 bytes = 67,241,280 bytes; divided by the selected dispatch's 151,641 ns
duration gives **443.424 decimal GB/s**, within 0.02% of the profiler's
443.47 `17.1.3`/`17.2.0` value. Interpreting 443.47 as gigabits/s would imply
about 55.4 GB/s, inconsistent with the event-level HBM byte count. The
defensible reporting convention is therefore to preserve the raw 17.2.0 label
and value, but use bytes/s (decimal GB/s) for cross-tool bandwidth comparisons.

For MI300X, retain the raw values and this caveat; the T3 probe has not yet
captured width-specific HBM requests on gfx942. The saved `17.1.3` and `17.2.0`
values still agree numerically, while the R2 compressed-input read rate is a
separate end-to-end event metric and should not be substituted for physical
HBM traffic.

## What this says—and does not say

- L2 hit rate and read traffic are observable on both architectures through
  `rocprof-compute`; the new MI350X replay also exposes raw HBM request counts.
- A direct MALL/Infinity Cache hit counter was not found in the MI350X counter
  catalog. The small/large working-set comparison in T1 remains a capacity
  comparison, not proof of cache residency.
- PC sampling is available on MI350X, but only its supported methods and
  intervals were inventoried here; no sampled-PC attribution is claimed.
- HBM read attribution is essentially 100% in the selected dispatch, remote
  read is zero, and the decode kernel has no global output write. These are
  profiler attribution results, not a standalone physical-bandwidth roofline.
- `rocprof-compute` has no explicit HBM/L3 traffic view for the Infinity Cache
  in this run. A future controlled cache-residency experiment or a newer
  supported counter path is needed before claiming MALL hit rate.

Raw captures (git-ignored) are under `results/profiling/h3-t3-mi350x-110303/`,
`results/profiling/h3-t3-mi350x-110307/`, and the earlier R3 directories documented in
[`GLM_LDS_PROFILING_REPORT.md`](GLM_LDS_PROFILING_REPORT.md). The T3 probe
launcher is [`job_h3_t3_probe.sh`](../scripts/job_h3_t3_probe.sh).
