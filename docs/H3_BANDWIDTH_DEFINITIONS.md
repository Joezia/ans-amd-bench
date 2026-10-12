# H3 bandwidth definitions and R2 comparison

This table is generated from the verified R2 summaries listed in the source column. It separates the decoder's effective output rate from the compressed-input footprint.

## Definitions

- Effective LDS throughput = `effective_decoded_bytes / (median HIP-event ms × 1,000,000)`, in decimal GB/s. The numerator is decoded output bytes written to LDS across all replicas; the denominator is the median end-to-end decode event interval. It is not HBM bandwidth or measured LDS physical traffic.
- Compressed input footprint = replica count × per-replica compressed device allocation. The allocation includes ANS archive metadata and alignment padding; it excludes lookup data, output buffers, and unrelated cache occupants.
- The 256 MiB comparison is a capacity-only estimate. Being below it does not establish MALL residency; actual residency requires counters or a controlled cache experiment.
- Replicas are identical tensor contents at independent GPU addresses. They are auxiliary cache/scale diagnostics, not distinct model weights, batch items, or real inference throughput.

## Measurements

| GPU | Case | Effective LDS output | Numerator: decoded bytes | Denominator: median event | Compressed archive / replica | Allocated compressed working set | vs 256 MiB MALL | Verification |
|---|---|---:|---:|---:|---:|---:|---|---|
| MI300X | single matrix (single real matrix) | 276.4 GB/s | 75,497,472 B | 0.273 ms | 62.40 MiB | 62.40 MiB (24.4%) | below 256 MiB | passed |
| MI300X | 4 replicas (1a cache-resident approximation) | 303.6 GB/s | 301,989,888 B | 0.995 ms | 62.40 MiB | 249.62 MiB (97.5%) | near 256 MiB (slightly under) | passed |
| MI300X | 16 replicas (auxiliary cache/HBM sweep) | 258.8 GB/s | 1,207,959,552 B | 4.668 ms | 62.40 MiB | 998.47 MiB (390.0%) | above 256 MiB | passed |
| MI300X | 64 replicas (1b HBM-sourced approximation) | 259.8 GB/s | 4,831,838,208 B | 18.595 ms | 62.40 MiB | 3993.89 MiB (1560.1%) | above 256 MiB | passed |
| MI350X | single matrix (single real matrix) | 451.9 GB/s | 75,497,472 B | 0.167 ms | 62.40 MiB | 62.40 MiB (24.4%) | below 256 MiB | passed |
| MI350X | 4 replicas (1a cache-resident approximation) | 545.2 GB/s | 301,989,888 B | 0.554 ms | 62.40 MiB | 249.62 MiB (97.5%) | near 256 MiB (slightly under) | passed |
| MI350X | 16 replicas (auxiliary cache/HBM sweep) | 478.8 GB/s | 1,207,959,552 B | 2.523 ms | 62.40 MiB | 998.47 MiB (390.0%) | above 256 MiB | passed |
| MI350X | 64 replicas (1b HBM-sourced approximation) | 482.7 GB/s | 4,831,838,208 B | 10.009 ms | 62.40 MiB | 3993.89 MiB (1560.1%) | above 256 MiB | passed |

The 4-replica case is a cache-resident *approximation* (about 249.5 MiB allocated, just under 256 MiB); do not describe it as confirmed cache-resident. The 64-replica case is about 3.90 GiB and exceeds that capacity. The single matrix is about 62.4 MiB. MI300X and MI350X figures are separate hardware measurements; the MI350X case has no measured BabelStream denominator in this table.

## Sources and scope

- `results/profiling/lds-r2-glm-mi300x-109848/summary.json`
- `results/profiling/lds-r2-glm-mi350x-109847/summary.json`
- Both summaries mark `verified_actual_LDS: true`. The single-matrix case uses 10 warmups and 50 timed iterations; replica sweeps use 5 warmups and 20 timed iterations.
- Rates are the existing R2 values. The generator recomputes output bytes divided by median event time and rejects discrepancies over 0.1%.
