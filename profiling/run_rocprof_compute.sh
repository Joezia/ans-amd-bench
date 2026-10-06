#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

if (( $# != 2 )); then
  echo "Usage: $0 symbols.bin meta.json" >&2
  exit 2
fi
require_file "$1"
require_file "$2"
require_file "${bench_cli}"
require_command rocprof-compute

output="${PROFILE_OUTPUT_DIR:-$(new_output_dir counters)}"
mkdir -p "${output}"
bench_command "$1" "$2"

blocks=(sol memchart spi wavefront cu_pipe lds vl1d l2 l2_per_channel)
set +e
rocprof-compute profile --output-directory "${output}/rocprof-compute" \
  --overwrite --block "${blocks[@]}" -- "${BENCH_COMMAND[@]}" \
  >"${output}/rocprof-compute.log" 2>&1
rc=$?
set -e

if (( rc == 0 )); then
  record_availability "${output}" rocprof-compute available \
    "SOL, HBM/L2, occupancy, wave, stalls, VALU, LDS, remote traffic, per-channel L2"
else
  record_availability "${output}" rocprof-compute unavailable \
    "rc=${rc}; permission/counter failure retained in rocprof-compute.log"
  echo "WARNING: full counter collection failed; collecting launch/top stats fallback" >&2
  set +e
  rocprof-compute profile --output-directory "${output}/fallback" --overwrite \
    --no-roof --block topstats wavefront -- "${BENCH_COMMAND[@]}" \
    >"${output}/fallback.log" 2>&1
  fallback_rc=$?
  set -e
  if (( fallback_rc == 0 )); then
    record_availability "${output}" rocprof-compute-fallback available \
      "topstats and wavefront only"
  else
    record_availability "${output}" rocprof-compute-fallback unavailable \
      "rc=${fallback_rc}; see fallback.log"
  fi
fi

echo "rocprof-compute output: ${output} (full rc=${rc})"
exit 0
