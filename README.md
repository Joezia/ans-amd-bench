# hipANS AMD GPU benchmarks

This repository contains reproducible MI210 (`gfx90a`) baselines for hipANS:

- `harness/`: bit-exact validation and raw ANS decode throughput.
- `sepbase/`: separated hipANS decode followed by vendor bf16 GEMM.
- `scripts/`: Slurm launchers, hipANS build helpers, and input generators.
- `results/published/`: curated CSV results with machine-local paths removed.

The repository intentionally does not contain model weights, generated binary
inputs, build products, raw logs, or private reference implementations.

## Dependencies

- HACC@NUS with Slurm and an MI210 partition.
- ROCm/HIP for `gfx90a`.
- A separate checkout of [hipANS](https://github.com/Joezia/hipANS), at the
  commit recorded in each result row.
- hipBLASLt and rocBLAS for `sepbase`.

By default, scripts expect the hipANS checkout at `../hipANS`. Override this
with `HIPANS_REPO=/path/to/hipANS`. All compute-heavy builds and benchmarks are
submitted through Slurm; the launchers refuse to run directly on `hacchead`.

## Quick start

Build the required static hipANS libraries first:

```bash
HIPANS_REPO=/path/to/hipANS ./scripts/job_build_full.sh
```

Then build and run either benchmark:

```bash
./harness/build.sh
FILE_INPUT="$PWD/inputs/qwen_bf16_256MiB.bin" ./harness/run_all.sh

./sepbase/build.sh
WEIGHT_FILE="$PWD/inputs/qwen_bf16_256MiB.bin" ./sepbase/run_all.sh
```

The build commands explicitly target `--offload-arch=gfx90a`. See the README
inside each benchmark directory for its complete configuration and output
schema.

## Inputs and results

Binary inputs are local-only. See `inputs/README.md` for the expected files and
generation process. Curated result CSVs are retained for comparison, while raw
Slurm stdout/stderr and job-numbered CSVs remain ignored.

The published CSV files use `MI210` as a non-identifying host label and relative
input paths. They preserve the measured values, software versions, and public
hipANS commit needed to interpret the experiment.

## Scope

`sepbase` is a separated baseline, not a fused decode-GEMM kernel: decoded
weights are written to HBM and then consumed by hipBLASLt or rocBLAS. No source
or history from the private reference repository is included.

See `docs/RESULTS.md` for a compact interpretation of the curated runs.

## License

No license has been selected for this benchmark repository yet. Add an explicit
license before making the repository public if reuse by others is intended.
