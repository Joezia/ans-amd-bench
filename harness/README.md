# hipANS raw decode benchmark harness

This directory contains an independent HIP/C++ harness for the raw
`ansDecodeBatchPointer` path. It does not build or modify the hipANS source
tree.

## Build

From the login node, provide the target GPU architecture and either one build
partition or a comma-separated list of eligible build partitions:

```bash
GPU_ARCH=gfx942 \
BUILD_PARTITIONS=devel \
HIPANS_BUILD_DIR="$PWD/build-hipans-mi300" \
./harness/build.sh
```

The script selects an immediately available partition when possible, obtains a
single-task Slurm allocation, prints the complete compiler command, and builds
`bin/hipans_raw_decode_bench`. The compiler command explicitly contains
`-DNDEBUG` and `--offload-arch=${GPU_ARCH}`. It links only the three requested
project archives: `libgpu_ans.a`, `libhipans_utils.a`, and `libglog.a`.

The executable rejects a runtime GPU whose base GFX architecture differs from
the architecture recorded at compile time. Rebuild after moving between, for
example, MI210 (`gfx90a`) and MI300X (`gfx942`).

Optional build environment variables:

- `GPU_ARCH`: required target passed to `--offload-arch`.
- `BUILD_PARTITION`: one explicit ROCm-enabled Slurm partition.
- `BUILD_PARTITIONS`: eligible partition list used only when
  `BUILD_PARTITION` is unset.
- `BUILD_TIME`: allocation time limit; default `00:15:00`.
- `HIPCC_BIN`: hipcc path; otherwise discovered from `PATH` in the allocation.
- `HIPANS_REPO`: hipANS source tree; default `../hipANS` relative to this
  repository.
- `HIPANS_BUILD_DIR`: prebuilt hipANS directory; default
  `build-hipans-full` in this repository.

## Run the required full matrix

From `hacchead`, run:

```bash
FILE_INPUT="$PWD/inputs/qwen_bf16_256MiB.bin" ./harness/run_all.sh
```

Also provide either `GPU_PARTITION` or `GPU_PARTITIONS`. Candidate partitions
must have compatible allocation semantics. Do not mix single-GPU and exclusive
whole-node partitions merely because they contain the same GPU model.

The script uses one `srun` allocation and pins the process to one physical card
with `HIP_VISIBLE_DEVICES`. It runs the full Cartesian product of:

- D1 uniform, D3 exponential, and DF file input;
- S1, S2, all three S3 tails, and S4 (64 MiB per batch member);
- probability bits 9, 10, and 11;
- batch counts 1 and 8.

This is 108 cases. Every case runs bit-exact validation, at least 10 warmups,
and at least 50 measured decodes. Do not lower those iteration counts.

Optional run environment variables:

- `GPU_PARTITION`: one explicit Slurm partition.
- `GPU_PARTITIONS`: comma-separated eligible partitions. The selector ranks
  them using current node state and free GPU TRES, without pinning a node.
- `GPU_COUNT`: requested GPU count; default `1`.
- `GPU_REQUEST_MODE`: `gres`, `gpus`, or `none`; default `gres`.
- `GPU_INDEX`: physical GPU index passed to `HIP_VISIBLE_DEVICES`; default `0`.
- `RUN_TIME`: Slurm time limit; default `12:00:00`.
- `FILE_INPUT`: DF binary source. It is repeated or truncated deterministically
  to the requested per-batch case size.
- `SEED`: D1/D3 fixed seed; default `20260920`.
- `LAMBDA`: D3 exponential-distribution lambda; default `100.0`.
- `WARMUP_ITERS`: warmup count, constrained to at least 10.
- `MEASURE_ITERS`: measurement count, constrained to at least 50.
- `RESULTS_DIR`: output directory under the harness; default `harness/results`.
- `CSV_PATH`: output CSV path; defaults to a Job-ID-named file in results.

The job writes stdout/stderr to the terminal and Job-ID-named log files. Use
the non-interactive launchers rather than holding an interactive GPU shell.

## Peak launcher

The focused launcher runs one smoke case followed by the large-workload search:

```bash
GPU_PARTITIONS=mi3001x \
GPU_COUNT=1 \
GPU_REQUEST_MODE=gres \
RUN_MODE=search \
FILE_INPUT="$PWD/inputs/expo20_u8_64MiB+1234.bin" \
./harness/run_mi300_peak.sh
```

An explicit `GPU_PARTITION` bypasses state-based partition selection. The
selected partition is passed into the allocation, but no node is pinned; Slurm
retains final placement control.

Command-line parameters:

- `--dataset d1|d3|df`: uniform bytes, skewed exponential bytes, or file data.
- `--size-bytes N`: uncompressed bytes in each independent batch member.
- `--probbits 9|10|11`: ANS probability precision.
- `--num-in-batch 1|8`: number of independent pointer-batch members.
- `--seed N`: deterministic seed printed in every run.
- `--lambda X`: exponential lambda used by D3.
- `--file PATH`: required for DF.
- `--warmup N`: discarded decode iterations; minimum 10.
- `--measure N`: measured decode iterations; minimum 50.
- `--gpu-index N`: physical index that must match `HIP_VISIBLE_DEVICES`.
- `--csv PATH`: append one schema-compliant row, creating the header if needed.
- `--case-label TEXT`: scale label added to stdout and CSV notes.
- `--verify-checksum-contract`: additionally validates supported checksum
  combinations outside timing. `run_all.sh` enables it on its first case.

## Measurement and validity rules

Input generation, H2D, encoding, correctness decode, and D2H comparison are
outside timing. Each timed sample contains only `ansDecodeBatchPointer`, bounded
by HIP events in the same stream. CPU `steady_clock` spans submission through
event synchronization.

The compressed buffers are immutable and disjoint from output buffers, so no
iteration consumes a prior output as input. After all samples, a separate GPU
kernel reads every output byte and reports a checksum; this is outside timing.

The harness captures C++ stderr. If it observes
`StackDeviceMemory: attempting to allocate`, the CSV row is `INVALID` and the
process exits nonzero. A failed or unparseable pre-measurement `rocm-smi`
utilization check marks the run `CONTENDED`. SCLK, MCLK, and temperature samples
are printed at measurement begin, midpoint, and end. Since the requested CSV
schema has no MCLK or midpoint-temperature columns, those values are preserved
in `notes`.

The benchmark does not request privileged clock or performance-level changes.
It records read-only clock, utilization, and temperature samples around the
measurement instead.

## Checksum configuration

Source inspection shows that `useChecksum=true` decode launches the checksum
kernel and reads the archive checksum, while `useChecksum=false` returns after
the raw table-build and decode path. A checksum-enabled decoder therefore needs
a checksum-enabled archive. A raw decoder can decode a checksum-enabled archive
without reading its checksum. The benchmark deliberately uses
`ANSCodecConfig{probBits, false}` for both encode and decode, and the optional
contract probe tests true/true and true/false archives outside timing.
