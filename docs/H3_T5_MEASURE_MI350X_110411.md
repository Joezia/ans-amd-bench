# H3 T5 multi-matrix decode — MI350X

Slurm job `110411` measured the byte-exact nine-matrix composite. The launch was selected from the same GPU's T4 single-matrix sweep and uses one independent artifact address (no replicas).

- Decoded output: 679,477,248 bytes (648.0 MiB).
- Compressed ANS archive: 588,470,288 bytes (561.2 MiB), 2.19× nominal 256 MiB MALL capacity.
- Launch: 384 threads/WG, grid multiplier 1, 1 stream(s); LDS 53,248 B/WG.
- Decode median/p10/p90: 1.397209/1.393641/1.402393 ms.
- Effective LDS output throughput: **486.31 decimal GB/s**; compressed archive read rate: 421.18 decimal GB/s.
- Correctness: actual LDS output byte-exact verification passed; warmup=10, iters=50.

This is a capacity-class 1b-style large-working-set measurement, not proof of every load missing MALL: the hardware profiler does not expose a direct MALL-hit counter here. The decoder pseudo-matrix is a byte layout of distinct source matrices; it is not one logical model tensor and does not run GEMM. The nine tensor streams were jointly re-encoded with one p10 table to form one HIP-ANS artifact.

Artifact SHA-256: `2ac416ab458d9b3f69c5a69d2799dc80c2359cd8ed6f242a1637a0e6a23a3f1a`. Curated result: [`h3_t5_measure_mi350x_110411.json`](../results/h3_t5_measure_mi350x_110411.json).
