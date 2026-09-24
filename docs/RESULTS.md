# Curated MI210 baseline results

These tables summarize the CSV files under `results/published/`. The recorded
hipANS revision is `f3de74171de38daf15e56b05ad8beb99a6bd47fc` and all listed
correctness checks passed. Times are HIP-event medians.

## Raw decode

For the real-weight DF dataset, S4 uses 64 MiB per batch member. The table below
reports effective uncompressed-output throughput.

| probBits | batch members | decode median (ms) | throughput (GB/s) | compressed/raw |
|---:|---:|---:|---:|---:|
| 9 | 1 | 0.414 | 161.97 | 0.8083 |
| 9 | 8 | 2.818 | 190.48 | 0.8083 |
| 10 | 1 | 0.425 | 157.95 | 0.8103 |
| 10 | 8 | 2.880 | 186.41 | 0.8103 |
| 11 | 1 | 0.533 | 125.86 | 0.8069 |
| 11 | 8 | 4.090 | 131.26 | 0.8069 |

The complete CSV contains 108 cases spanning D1/D3/DF, six sizes, three
probability precisions, and batch counts 1 and 8. Every row is `PASS/OK`.

## Separated decode plus GEMM

The weight matrix is bf16 `K x N = 1024 x 1024` (2 MiB raw). GEMM uses fp32
accumulation and output through rocBLAS. `ratio` is
`median(t_decode+gemm) / median(t_gemm)`.

| M | cache | GEMM (ms) | decode (ms) | decode + GEMM (ms) | ratio |
|---:|:---:|---:|---:|---:|---:|
| 64 | cold | 0.0235 | 0.1138 | 0.1225 | 5.208 |
| 64 | hot | 0.0242 | 0.1131 | 0.1222 | 5.060 |
| 1024 | cold | 0.0363 | 0.1139 | 0.1395 | 3.841 |
| 1024 | hot | 0.0366 | 0.1130 | 0.1397 | 3.812 |
| 4096 | cold | 0.0972 | 0.1132 | 0.2023 | 2.081 |
| 4096 | hot | 0.0963 | 0.1129 | 0.2025 | 2.102 |

The raw-over-compressed ratio is 1.2348 for all six rows. This experiment is a
separated baseline; it does not overlap decode and GEMM and does not fuse the
decoder into the matrix kernel.

## Environment caveat

The runs recorded an MI210 (`gfx90a`) with ROCm 6.3.x. The attempt to set the
performance level to high was not permitted, and sampled SCLK remained at
800 MHz. These are baseline measurements rather than locked-clock peak results.
