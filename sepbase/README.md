# hipANS decode + vendor bf16 GEMM separated baseline

This directory contains an independent MI210 (`gfx90a`) benchmark. It leaves
the external `hipANS` source tree and `bench/harness` read-only.

## What is measured

For each row-major shape `(M,N,K)`, `A` is `M x K` bf16, the real-weight byte
stream `B` is `K x N` bf16, and `C` is `M x N` fp32. Computation is fp32
accumulation with `alpha=1` and `beta=0`.

The three reference shapes are:

- `(64, 1024, 1024)`
- `(1024, 1024, 1024)`
- `(4096, 1024, 1024)`

The timed intervals are `t_gemm` (vendor GEMM on raw `B`), `t_decode`
(hipANS decode only), and `t_dec_gemm` (decode then GEMM in one HIP stream).
The primary ratio is
`median(t_dec_gemm) / median(t_gemm)`. Each interval has 10 or more discarded
warmups and 50 or more measured iterations. The CSV stores median, p5, and p95.

Both cache modes are measured:

- `cold`: a 128 MiB buffer is written in the same stream before the start
  event. The flush is outside the timed interval.
- `hot`: no explicit cache flush.

hipBLASLt is tried first. The first successful result returned by its heuristic
API is used and its algorithm index, solution/kernel name, and workspace are
recorded. If that setup is unsupported, the program uses
`rocblas_gemm_ex` with `rocblas_gemm_algo_standard`.

## Build

From `hacchead`, run:

```bash
./build.sh
```

The script submits the build with `srun`; it refuses to build directly on the
login node. It prints the complete compiler command. That command explicitly
contains `--offload-arch=gfx90a` and `-DNDEBUG`.

Optional environment variables:

- `BUILD_PARTITION`: Slurm partition, default `mi210_u280_u55c`.
- `BUILD_TIME`: Slurm time request, default `00:15:00`.
- `HIPCC_BIN`: compiler path, default `/usr/bin/hipcc`.
- `HIPANS_REPO`: hipANS source tree; default `../hipANS` relative to this
  repository.
- `HIPANS_BUILD_DIR`: prebuilt hipANS directory; default
  `build-hipans-full` in this repository.

## Run all reference shapes

From `hacchead`, after building, run:

```bash
./run_all.sh
```

The launcher creates one Slurm allocation and runs all three shapes on the
same physical GPU. Stdout and stderr are both shown in the terminal and also
saved under `results/`. Two CSV rows (cold and hot) are appended per shape.

Optional environment variables:

- `GPU_PARTITION`: default `mi210_u280_u55c`.
- `GPU_INDEX`: physical GPU selected through `HIP_VISIBLE_DEVICES`, default `0`.
- `RUN_TIME`: default `02:00:00`.
- `WEIGHT_FILE`: default `inputs/qwen_bf16_256MiB.bin` in this repository.
- `ACTIVATION_SEED`: default `20260921`.
- `WARMUP_ITERS`: minimum 10, default 10.
- `MEASURE_ITERS`: minimum 50, default 50.
- `CPU_REL_TOLERANCE`: sampled fp32-reference scaled-error threshold, default
  `0.02`. The pass/fail denominator is
  `max(abs(reference), 0.01*sqrt(K))`, matching the reference benchmark's
  tolerance style; the ordinary maximum relative error is reported separately.
- `RESULTS_DIR`: output directory, default `sepbase/results`.
- `CSV_PATH`: explicit append-only CSV path.

The weight file's leading `K*N*2` bytes are used when it is large enough. A
smaller file is cyclically repeated. The path, source size, and exact fill rule
are printed and written to CSV. `A` uses a seeded normal distribution and is
rounded to bf16.

## Direct binary options

The binary is normally invoked by `run_all.sh` inside Slurm:

```text
--m <rows of A/C>
--n <columns of B/C>
--k <columns of A / rows of B>
--weight-file <binary bf16 byte stream>
--csv <append-only CSV path>
--seed <activation RNG seed>
--warmup <integer >= 10>
--measure <integer >= 50>
--gpu-index <physical GPU index also used by HIP_VISIBLE_DEVICES>
--cpu-rel-tolerance <sampled CPU fp32 relative-error threshold>
```

Correctness is outside all timed intervals: decoded `B` is compared byte for
byte, vendor GEMM outputs for raw and decoded `B` are compared bit for bit, and
nine output locations are checked with a CPU fp32 dot product. No correctness
check relies on `assert`.
