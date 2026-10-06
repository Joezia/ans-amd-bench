#!/usr/bin/env bash
set -euo pipefail
profiling_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
bench_dir="$(cd -- "${profiling_dir}/.." && pwd)"
partition="${GPU_PARTITION:-mi3001x}"
run_time="${RUN_TIME:-02:00:00}"

if (( $# != 2 )); then
  echo "Usage: $0 symbols.bin meta.json" >&2
  exit 2
fi
input="$(realpath "$1")"
meta="$(realpath "$2")"

if [[ "${1:-}" != "--inside-allocation" && -z "${SLURM_JOB_ID:-}" ]]; then
  exec srun --job-name=ans-profile --partition="${partition}" --gpus=1 \
    --nodes=1 --ntasks=1 --cpus-per-task=4 --time="${run_time}" \
    env GPU_PARTITION="${partition}" "${BASH_SOURCE[0]}" "${input}" "${meta}"
fi

root="${PROFILE_RESULTS_DIR:-${bench_dir}/results/profiling}/profile-${SLURM_JOB_ID}-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "${root}"

PROFILE_OUTPUT_DIR="${root}/timeline" "${profiling_dir}/run_rocprofv3.sh" "${input}" "${meta}" || true
PROFILE_OUTPUT_DIR="${root}/counters" "${profiling_dir}/run_rocprof_compute.sh" "${input}" "${meta}"
PROFILE_OUTPUT_DIR="${root}/system" "${profiling_dir}/run_rocprof_sys.sh" "${input}" "${meta}"
echo "all profiling output: ${root}"
