#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
bench_dir="$(cd -- "${script_dir}/.." && pwd)"
partition="${GPU_PARTITION:-mi3001x}"
input="${CAMPAIGN_INPUT:-${bench_dir}/inputs/expo20_u8_64MiB+1234.bin}"
meta="${CAMPAIGN_META:-${bench_dir}/configs/smoke_meta.json}"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  exec srun --job-name=ans-mi300-campaign --partition="${partition}" --gpus=1 \
    --nodes=1 --ntasks=1 --cpus-per-task=8 --time="${RUN_TIME:-04:00:00}" \
    "${BASH_SOURCE[0]}"
fi

export HIPANS_STACK_BYTES="${HIPANS_STACK_BYTES:-1073741824}"
"${script_dir}/run_peak_bw.sh"
peak="$("${bench_dir}/.venv/bin/python" -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["selected_peak_GBps"])' \
  "${bench_dir}/results/peak_bw.json")"
export PEAK_GBPS="${peak}"

"${bench_dir}/ans_bench" --input "${input}" --meta "${meta}" \
  --warmup 10 --iters 50 --peak-GBps "${peak}" \
  >"${bench_dir}/results/standard_mi300.json" \
  2>"${bench_dir}/results/standard_mi300.err"

PROFILE_WARMUP=1 PROFILE_ITERS=2 \
  "${bench_dir}/profiling/run_all.sh" "${input}" "${meta}"

SWEEP_SIZES_MIB="${SWEEP_SIZES_MIB:-64,192,384}" \
SWEEP_TILE_SIZES="${SWEEP_TILE_SIZES:-8192,65536,1048576}" \
SWEEP_WARMUP="${SWEEP_WARMUP:-3}" SWEEP_ITERS="${SWEEP_ITERS:-10}" \
  "${bench_dir}/profiling/run_sweeps.sh" "${input}"

if compgen -G "${bench_dir}/bin/ans_bench-t*-g*" >/dev/null; then
  "${bench_dir}/profiling/run_launch_sweep.sh" "${input}" "${meta}"
fi
echo "MI300 campaign complete"
