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
require_command rocprofv3

output="${PROFILE_OUTPUT_DIR:-$(new_output_dir timeline)}"
mkdir -p "${output}/trace"
bench_command "$1" "$2"

set +e
rocprofv3 --runtime-trace --stats --summary \
  --group-by-queue --output-directory "${output}/trace" \
  --output-file ans-decode --output-format csv json pftrace \
  -- "${BENCH_COMMAND[@]}" >"${output}/bench.json" 2>"${output}/rocprofv3.log"
rc=$?
set -e
if (( rc == 0 )); then
  record_availability "${output}" rocprofv3 available "runtime, kernel, copy, allocation timeline"
else
  record_availability "${output}" rocprofv3 unavailable "rc=${rc}; see rocprofv3.log"
fi
echo "rocprofv3 output: ${output} (rc=${rc})"
exit "${rc}"
