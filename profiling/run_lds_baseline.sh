#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
manifest="${1:?Usage: run_lds_baseline.sh artifact.json output-directory}"
output="${2:?output directory required}"
mkdir -p "${output}"
command=("${bench_dir}/.venv/bin/python" "${bench_dir}/scripts/run_lds_bench.py"
  --manifest "${manifest}" --skip-verify --warmup "${PROFILE_LDS_WARMUP:-2}" --iters 1)
if command -v rocprofv3 >/dev/null 2>&1; then
  mkdir -p "${output}/timeline"
  if rocprofv3 --runtime-trace --stats --output-directory "${output}/timeline" \
      --output-format csv json -- "${command[@]}" >"${output}/timeline.json" 2>"${output}/timeline.log"; then
    record_availability "${output}" rocprofv3 available "LDS discard kernel, copies and host/device timeline"
  else
    record_availability "${output}" rocprofv3 unavailable "permission/tool failure: see timeline.log"
  fi
else
  record_availability "${output}" rocprofv3 unavailable "not installed"
fi
compute="$(command -v rocprof-compute || true)"
compute_command=("${compute}")
if [[ -x "${bench_dir}/.venv/rocprof-compute/bin/python" && -n "${compute}" ]]; then
  compute_command=("${bench_dir}/.venv/rocprof-compute/bin/python" "$(readlink -f "${compute}")")
fi
if [[ -n "${compute}" ]]; then
  if "${compute_command[@]}" profile --output-directory "${output}/counters" \
      --overwrite --block sol memchart spi wavefront cu_pipe lds vl1d l2 \
      -- "${command[@]}" >"${output}/counters.log" 2>&1; then
    record_availability "${output}" rocprof-compute available "SOL, memory, occupancy, waits, VALU, LDS and L2; inspect raw counters for missing metrics"
  else
    record_availability "${output}" rocprof-compute unavailable "see counters.log; counter permissions/tool failure"
    if "${compute_command[@]}" profile --output-directory "${output}/fallback" \
        --overwrite --no-roof --block topstats wavefront -- "${command[@]}" \
        >"${output}/fallback.log" 2>&1; then
      record_availability "${output}" rocprof-compute-fallback available "topstats and wavefront only"
    else
      record_availability "${output}" rocprof-compute-fallback unavailable "see fallback.log"
    fi
  fi
else
  record_availability "${output}" rocprof-compute unavailable "not installed"
fi
