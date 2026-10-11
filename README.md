# hipANS AMD GPU benchmarks

This repository contains reproducible MI210 (`gfx90a`) baselines and a
standardized MI300X (`gfx942`) measurement/profiling workflow for hipANS:

The current objective is AMD optimization for precompressed model-weight
inference on MI300X and MI350X: first decode real weight tiles to LDS and profile/optimize that
stage, then fuse ANS decoding with pipelined GEMM. The existing `ans_bench`
measures a historical global-output byte-codec path. A separate, verified
native-FP8 decode-to-LDS adapter directly reuses the current unoptimized hipANS
decoder. Its resource-saturation baseline is being remeasured; the previous
replacement-codec baseline was withdrawn. Optimized/fused paths are not implemented yet.

- `ans_bench`: stable single-case JSON CLI with byte-exact verification.
- `bench/`: implementation of the standardized CLI.
- `harness/`: bit-exact validation and raw ANS decode throughput.
- `sepbase/`: separated hipANS decode followed by vendor bf16 GEMM.
- `profiling/`: rocprofv3/rocprof-compute runners and diagnostic sweeps.
- `scripts/`: Slurm launchers, hipANS build helpers, and input generators.
- `results/published/`: curated CSV results with machine-local paths removed.

The repository intentionally does not contain model weights, generated binary
inputs, build products, raw logs, or private reference implementations.

## Project tracking

- [战略计划](docs/project/PLAN.md)：记录长期目标、阶段关系和成功标准。
- [执行状态](docs/project/STATUS.md)：记录当前进展、证据、阻塞和近期动作。
- [原始 Step 0–4 产物索引](docs/project/LEGACY_STEPS.md)：定位已有代码、结果和完成边界。
- [论文场景与数值格式](docs/PAPER_SCENARIO.md)：本地论文核对结果。
- [第一阶段 LDS 工程验收](docs/LDS_STAGE1.md)：预压缩输入、正确性、计时和 profiling 要求。
- [当前独立交付计划](docs/project/DELIVERY_HIPANS_S1_BASELINE.md)：真实 hipANS LDS baseline 的范围。
- [LDS baseline 使用说明](docs/LDS_CLI.md)：离线包、独立 CLI、验证及 Slurm 返回结果。
- [MI350X 独立报告](docs/LDS_BASELINE_REPORT_MI350X.md)：单独 gfx950 构建、测量与 profiling，不覆盖 MI300X。

## Dependencies

- HACC@NUS with Slurm and an MI210 or MI300X partition.
- ROCm/HIP for `gfx90a`.
- A separate checkout of [hipANS](https://github.com/Joezia/hipANS), at the
  commit recorded in each result row.
- hipBLASLt and rocBLAS for `sepbase`.

By default, new scripts locate the hipANS checkout at `../hipANS`.
Override it with `HIPANS_ROOT=/path/to/hipANS`. Legacy harness scripts retain
the `HIPANS_REPO` spelling. All build products stay under this repository.
As authorized on 2026-10-11, hipANS sources may be modified for decoder and
fusion optimization. Keep each repository's changes/commits separate and
record codec commit, dirty state and build flags for every experiment.
entropyflow-lab and its existing data/cache remain read-only.

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

For the standardized MI300X path, reuse the out-of-tree library build and run:

```bash
GPU_ARCH=gfx942 HIPANS_BUILD_DIR="$PWD/build-hipans-mi300" \
  ./scripts/build_ans_bench.sh

./ans_bench --input inputs/expo20_u8_64MiB+1234.bin \
  --meta configs/smoke_meta.json --warmup 10 --iters 50 \
  --peak-GBps "$(.venv/bin/python -c \
    'import json; print(json.load(open("results/peak_bw.json"))["selected_peak_GBps"])')"
```

See `docs/CLI.md` for the machine-readable schema, `docs/METRICS.md` for byte
and utilization definitions, and `docs/PROFILING_REPORT.md` for the bottleneck
attribution. Build and measure the achievable HBM reference with
`scripts/run_peak_bw.sh`; run the profiler suite with
`profiling/run_all.sh INPUT META`.

The legacy harness build commands explicitly target `gfx90a`; the standardized
path takes `GPU_ARCH` and defaults to MI300X `gfx942`. See the README inside
each benchmark directory for its complete configuration and output schema.

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
