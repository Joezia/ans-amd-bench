#!/usr/bin/env bash
set -euo pipefail

profiling_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
bench_dir="$(cd -- "${profiling_dir}/.." && pwd)"
bench_cli="${ANS_BENCH_CLI:-${bench_dir}/ans_bench}"

require_file() {
  if [[ ! -r "$1" ]]; then
    echo "ERROR: required readable file missing: $1" >&2
    exit 2
  fi
}

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "ERROR: required command unavailable: $1" >&2
    exit 2
  fi
}

new_output_dir() {
  local label="$1"
  local root="${PROFILE_RESULTS_DIR:-${bench_dir}/results/profiling}"
  local stamp
  stamp="$(date -u +%Y%m%dT%H%M%SZ)"
  local output="${root}/${label}-${SLURM_JOB_ID:-local}-${stamp}"
  mkdir -p "${output}"
  printf '%s\n' "${output}"
}

record_availability() {
  local output="$1"
  local component="$2"
  local status="$3"
  local detail="$4"
  if [[ ! -e "${output}/availability.tsv" ]]; then
    printf 'component\tstatus\tdetail\n' > "${output}/availability.tsv"
  fi
  printf '%s\t%s\t%s\n' "${component}" "${status}" "${detail//$'\t'/ }" \
    >> "${output}/availability.tsv"
}

bench_command() {
  local input="$1"
  local meta="$2"
  local warmup="${PROFILE_WARMUP:-2}"
  local iters="${PROFILE_ITERS:-5}"
  local peak="${PEAK_GBPS:-0}"
  BENCH_COMMAND=("${bench_cli}" --input "${input}" --meta "${meta}"
                 --warmup "${warmup}" --iters "${iters}")
  if [[ "${peak}" != 0 ]]; then
    BENCH_COMMAND+=(--peak-GBps "${peak}")
  fi
}
