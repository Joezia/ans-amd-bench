# MI300X decoder profiling report

Scope update (2026-10-10): this report profiles the historical standalone
byte decoder writing to global memory. It is not a profile of precompressed
real model tiles decoded to LDS or of a fused ANS+GEMM kernel. Its bottleneck
attribution and headroom estimates apply only to the measured demo path.
The current project scope is in [PLAN.md](project/PLAN.md); the new native-FP8
LDS reference baseline now has its own correctness and hardware evidence in
[LDS_BASELINE_REPORT.md](LDS_BASELINE_REPORT.md). The supplied local
paper does not establish the previously cited 30% numerator/denominator; see
[PAPER_SCENARIO.md](PAPER_SCENARIO.md).

## Conclusion

For the measured standalone demo, the observations favor scheduling,
dependency, and memory-behavior limitations over simple HBM saturation.
They do not establish a fundamental ANS limit or the bottlenecks of a future
LDS/fused implementation. The standardized single-stream decoder produces 637.57 GB/s,
or 15.54% of the measured 4,102.71 GB/s BabelStream Triad rate.  At the same
time, the decoder kernel spends about 79.9% of wave cycles in dependency wait,
has only 18.7% of peak IPC, and leaves the HBM-facing fabric well below its
roofline.  Eight independent process/queue instances reach an approximate sum
of 2.86 TB/s, showing substantial latency-hiding headroom with the codec held
constant.

The most direct source-level explanation is the CUDA-shaped synchronization
and launch structure.  A full 8,192-byte rANS block executes 128 serial symbol
iterations on wave64.  `decodeOneWarp` uses a workgroup-wide
`__syncthreads()` before every wave ballot, so the two waves in a 128-thread
workgroup are coupled at least 128 times per full block.  The kernel also
launches an occupancy-derived grid independently for every tile, even when a
64 KiB tile contains only eight useful rANS blocks.  Counter data proves the
dependency stalls; the source establishes frequent barriers, but the current
counters do not isolate barrier cycles from data dependencies.  A barrier-only
A/B experiment is therefore required before assigning an exact fraction of
the stall to synchronization.

## Reproducible baseline

Campaign `107394` completed on `tn1-003-002` with one MI300X, ROCm 10.0.0, and
hipANS commit `8cc8bc263bc943557682a5cc675a045cb9a86412` (clean).  The input was
67,110,098 byte symbols, probability precision 10, 65,536-symbol tiles, and
8,192-byte rANS blocks.  Every benchmark and sweep performed byte-exact
roundtrip verification.

| Quantity | Result |
|---|---:|
| BabelStream HIP Triad | 4,102.71 GB/s |
| Decode median, p10, p90 | 0.10526, 0.10359, 0.11173 ms |
| Logical compressed-read throughput | 434.52 GB/s |
| Logical decoded-write throughput | 637.57 GB/s |
| Standardized utilization | 15.54% |
| Compression ratio, raw/compressed | 1.4673 |

The utilization numerator is decoded output bytes only, as defined in
[METRICS.md](METRICS.md). It is an output-normalized HBM reference, not literal
HBM traffic or the paper's fused decode-plus-GEMM throughput. The paper's
previously assumed 30% definition has not been verified.

## Counter and timeline evidence

The rows below select only `ansDecodeKernel<...,128,10,8192>` from the
rocprof-compute report.  Profiler replay slows the instrumented application,
so the unprofiled standardized run remains the performance authority.

| Area | Measured evidence |
|---|---|
| Compute/issue | VALU 40.29%, SALU 13.63%, VMEM 4.63%, IPC 0.9367 (18.73% of peak), 47.75/64 active VALU lanes |
| Residency | wave occupancy 59.96%, CU utilization 80.12%, 28 VGPR/thread, 32 SGPR/wave, 4 KiB LDS/workgroup, zero scratch |
| Wave stalls | 726.69M dependency-wait, 51.09M issue-wait, and 136.33M active cycles out of 909.30M wave cycles per kernel |
| Cache | vL1D hit 80.48%, coalescing 84.97%, median vL1D stall on L2 data 70.85%; L2 hit 45.33% |
| HBM-facing fabric | read 1,085.12 GB/s, write 891.33 GB/s; 1,976.45 GB/s combined, 37.12% of rocprof-compute's 5,324.8 GB/s theoretical rate and 48.17% of BabelStream Triad |
| Latency/locality | fabric read latency 484.18 cycles, write latency 246.19 cycles; remote reads 0.0104%, remote writes 0%; write-credit starvation 0% |
| LDS | 25.42% utilization, 14.85% theoretical bandwidth, 2.22% bank-conflict rate |
| Decode sequence | table kernel 16.18 us mean followed immediately by main decoder 73.61 us mean; table is 18.0% of those two kernels |
| Copies/overlap | no copy/kernel overlap in this single-queue trace; H2D input and D2H verification are outside the timed decode definition |

The physical fabric rate is much larger than the logical 637.57 GB/s output
rate.  This is expected because table loads, compressed reads, output writes,
cache-line amplification, and other internal traffic are not represented by
the output-only numerator.  Consequently, “15.54% utilization” means poor
useful decoded-byte production, not that 84.46% of all HBM transactions are
idle.

No direct 256 MiB MALL hit-rate field was exposed by the selected gfx942
analysis blocks.  L2 hit rate and HBM/remote-traffic counters are available.
The size sweep also shows no consistent throughput cliff between 192 MiB and
384 MiB, although the 192 MiB/64 KiB point is an outlier and should be repeated
before drawing a cache-capacity conclusion.

## Diagnostic sweeps

Tile size has much more effect than total input size.  Values are logical
decoded-output GB/s; every cell uses the same rANS state transition and block
size.

| Input size | 8 KiB tile | 64 KiB tile | 1 MiB tile |
|---:|---:|---:|---:|
| 64 MiB | 149.66 | 663.44 | 715.28 |
| 192 MiB | 142.03 | 429.68 | 764.69 |
| 384 MiB | 145.35 | 673.40 | 724.75 |

The 8 KiB case repeats per-tile table construction and launches far more
mostly-empty workgroups; it remains near 145 GB/s at every total size.  The
absence of a monotonic 256 MiB boundary effect argues against MALL capacity as
the primary limiter.  The 64 KiB outlier at 192 MiB is not used for headroom
estimates.

Launch-only variants are comparatively flat: 632.88 GB/s for 64 threads/grid
4, and 640.13, 615.32, and 636.60 GB/s for 128 threads/grid 2, 4, and 8.  The
observed range is about 4%, so workgroup size or grid oversubscription alone is
not a large fix.  The unavailable 64-thread/grid-2 and grid-8 binaries are
recorded in the launch manifest rather than silently omitted.

Independent process/queue concurrency produced approximate summed rates of
0.667, 1.320, 1.791, and 2.862 TB/s for 1, 2, 4, and 8 queues.  These are sums
of overlapping per-process HIP-event intervals, not a globally synchronized
end-to-end measurement, so they are evidence of latency-hiding potential rather
than a replacement for the standardized result.

## Attribution and next-month hypotheses

| factor | evidence (counter values) | estimated headroom | proposed optimization for next month |
|---|---|---|---|
| Memory bandwidth and cache behavior | Useful output is 15.54% of BabelStream, while HBM-facing read+write is already 1,976 GB/s; L2 hit is 45.33%, L2-data stall is 70.85%, and read latency is 484 cycles | No speedup follows directly from output/peak or fabric traffic; traffic per useful byte must be measured for each candidate | Improve coalescing and reuse of compressed state/table data; validate traffic bytes per decoded byte for each candidate |
| Compute and instruction issue | VALU 40.29%, IPC 18.73% of peak, active cycles about 15.0%, no MFMA work | Compute throughput is not the present roof; no defensible compute-only speedup is assigned | Shorten the serial state-update/ballot critical path and inspect generated ISA scheduling |
| Synchronization and dependency | Dependency wait is about 79.9%; the full-block source executes at least 128 workgroup barriers per rANS block and couples two waves | Aggregate wave-cycle ratios are not additive wall-time accounting; barrier-specific speedup is not isolated | Replace workgroup-wide coordination with wave-scoped primitives where correctness permits, then run a barrier-only A/B counter experiment |
| Occupancy and deployment | Occupancy 59.96%, CU use 80.12%; launch sweep changes only ~4%, but eight independent queues sum to ~2.86 TB/s | Simple grid/block tuning: about 4%; better independent-work scheduling may be much larger but is not yet a single-kernel result | Redesign tile-to-wave work assignment or use a persistent work queue so MI300X can hide per-stream dependency latency |
| XCD/NUMA placement | Remote read 0.0104%, remote write 0%, NPS1/SPX, no fabric write-credit starvation | Below measurement relevance for this run (<0.1%) | Keep placement controls and counters in regression tests; do not prioritize NUMA tuning now |
| Pipeline design | Table construction is 16.18 us versus 73.61 us main decode; copy/kernel overlap is zero and copies are excluded from the timed metric | For these two kernel durations alone, removing table work gives (16.18+73.61)/73.61 ≈ 1.22×; whole-call gain includes other overhead and must be remeasured | Cache/amortize decode tables where model semantics allow, and prototype decode-to-LDS producer/consumer overlap with GEMM |

## Tool coverage and limits

- rocprofv3 collected HIP runtime, kernel, allocation, and copy timelines.
- rocprof-compute collected SOL, wave, occupancy, LDS, vL1D, L2/fabric,
  per-channel, and remote-traffic counters without elevated privileges.
- rocprof-sys/omnitrace was not installed; its absence is recorded and did not
  stop the campaign.
- Counter replay results and the unprofiled HIP-event baseline are deliberately
  kept separate.  Raw profiling directories remain cluster-local; the durable
  selected values are in `results/profiling_summary.json`.

The evidence does not establish a fundamental ANS limit, and grid-only tuning
has little effect in the measured demo. The leading hypothesis is that a
CUDA-oriented warp/workgroup organization exposes serial rANS dependencies,
uses barriers broader than an AMD wave, and generates avoidable table/traffic
work.  That conclusion is strong at the category level; the exact split among
barrier latency, cache misses, and state dependency still needs controlled A/B
experiments in the next optimization phase.
