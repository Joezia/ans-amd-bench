# H3 T2 bench-only decode ablations

**All A1/A2/A2c/A3 measurements below are INVALID as product-code performance.** The kernels live only in `bench/`; the upstream hipANS source tree was not changed. Timing excludes the separate byte-exact verification launch.

## Results

| GPU | Variant | Effective LDS output | Median / p10–p90 | LDS per WG | Resident WG/CU | Output verification | Comparison purpose |
|---|---|---:|---:|---:|---:|---|---|
| AMD Instinct MI300X | A0 current hipANS | 276.4 GB/s | 0.273 ms | 20,480 B | — | R2 byte-exact passed | reference |
| AMD Instinct MI300X | A1 (INVALID) | 326.8 GB/s | 0.231 / 0.230–0.233 ms | 20,480 B | 3 | passed | barrier removed vs A0 |
| AMD Instinct MI300X | A2 (INVALID) | 178.3 GB/s | 0.423 / 0.423–0.424 ms | 38,208 B | 1 | passed | compressed words staged in LDS |
| AMD Instinct MI300X | A2c (INVALID) | 115.5 GB/s | 0.654 / 0.652–0.656 ms | 38,208 B | 1 | passed | A2 LDS allocation control; global reads retained |
| AMD Instinct MI300X | A3 (INVALID) | 183.0 GB/s | 0.413 / 0.412–0.413 ms | 38,208 B | 1 | passed | A1 + A2 combined |
| AMD Instinct MI350X | A0 current hipANS | 451.9 GB/s | 0.167 ms | 20,480 B | — | R2 byte-exact passed | reference |
| AMD Instinct MI350X | A1 (INVALID) | 533.7 GB/s | 0.141 / 0.140–0.143 ms | 20,480 B | 8 | passed | barrier removed vs A0 |
| AMD Instinct MI350X | A2 (INVALID) | 517.2 GB/s | 0.146 / 0.145–0.147 ms | 38,208 B | 4 | passed | compressed words staged in LDS |
| AMD Instinct MI350X | A2c (INVALID) | 307.0 GB/s | 0.246 / 0.245–0.248 ms | 38,208 B | 4 | passed | A2 LDS allocation control; global reads retained |
| AMD Instinct MI350X | A3 (INVALID) | 549.2 GB/s | 0.137 / 0.136–0.139 ms | 38,208 B | 4 | passed | A1 + A2 combined |

## Measurement and interpretation notes

- Workload: GLM-5.2-FP8 layer-0 gate projection, shape `[12288, 6144]`, F8_E4M3 bytes arranged as 9,216 independent 8 KiB tiles; one matrix, one stream, 128 threads (2 waves/WG), grid multiplier 4.
- Each variant uses 10 warmups and 50 timed iterations with HIP events. Throughput numerator is decoded output bytes written to LDS; denominator is median event time. Table output is decimal GB/s.
- A2/A2c must pass byte-exact verification or the collection job fails. A1/A3 verification status is recorded, but all ablation data remains invalid even if correct because these are isolated experiments, not supported codec builds.
- A2 stages each tile's compressed words to a per-wave LDS region once, then decodes from LDS. A2c allocates the same extra LDS footprint but keeps the original global reads, controlling for occupancy changes. A1 removes only the copied decoder's per-symbol workgroup barrier; the table-load and output-safety barriers remain.
- Compare A2 with A2c for the combined benefit of replacing dependent global refills with LDS refills after paying staging cost. Compare A1/A3 against A0 for barrier and combined effects. The effects need not add linearly.
- Raw per-job JSON is retained under `results/profiling/h3-t2-*`; the combined machine results and provenance are in `results/h3_ablation_t2.json`.
