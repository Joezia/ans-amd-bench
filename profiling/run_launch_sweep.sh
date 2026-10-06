#!/usr/bin/env bash
set -euo pipefail
profiling_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
bench_dir="$(cd -- "${profiling_dir}/.." && pwd)"

if (( $# != 2 )); then
  echo "Usage: $0 symbols.bin meta.json" >&2
  exit 2
fi
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  exec srun --job-name=ans-launch-sweep --partition="${GPU_PARTITION:-mi3001x}" \
    --gpus=1 --nodes=1 --ntasks=1 --cpus-per-task=2 \
    --time="${RUN_TIME:-00:30:00}" "${BASH_SOURCE[0]}" \
    "$(realpath "$1")" "$(realpath "$2")"
fi

output="${LAUNCH_SWEEP_OUTPUT_DIR:-${bench_dir}/results/profiling/launch-${SLURM_JOB_ID}-$(date -u +%Y%m%dT%H%M%SZ)}"
mkdir -p "${output}"
printf 'threads\tgrid_multiplier\tstatus\tresult\n' >"${output}/manifest.tsv"

for threads in 64 128; do
  for grid in 2 4 8; do
    binary="${bench_dir}/bin/ans_bench-t${threads}-g${grid}"
    result="${output}/t${threads}-g${grid}.json"
    if [[ ! -x "${binary}" ]]; then
      printf '%s\t%s\tunavailable\t%s\n' "${threads}" "${grid}" \
        "run profiling/build_launch_variants.sh" >>"${output}/manifest.tsv"
      continue
    fi
    if "${binary}" --input "$1" --meta "$2" --warmup "${SWEEP_WARMUP:-5}" \
         --iters "${SWEEP_ITERS:-20}" >"${result}" 2>"${result%.json}.err"; then
      printf '%s\t%s\tok\t%s\n' "${threads}" "${grid}" "${result}" \
        >>"${output}/manifest.tsv"
    else
      printf '%s\t%s\tfailed\t%s\n' "${threads}" "${grid}" \
        "${result%.json}.err" >>"${output}/manifest.tsv"
    fi
  done
done
echo "launch sweep output: ${output}"
