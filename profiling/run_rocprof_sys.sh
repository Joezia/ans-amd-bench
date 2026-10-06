#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

if (( $# != 2 )); then
  echo "Usage: $0 symbols.bin meta.json" >&2
  exit 2
fi
require_file "$1"
require_file "$2"
output="${PROFILE_OUTPUT_DIR:-$(new_output_dir system-timeline)}"
mkdir -p "${output}"
bench_command "$1" "$2"

tool=""
if command -v rocprof-sys >/dev/null 2>&1; then
  tool="rocprof-sys"
elif command -v omnitrace >/dev/null 2>&1; then
  tool="omnitrace"
fi

if [[ -z "${tool}" ]]; then
  record_availability "${output}" rocprof-sys unavailable \
    "neither rocprof-sys nor legacy omnitrace is installed"
  echo "rocprof-sys unavailable; recorded in ${output}/availability.tsv"
  exit 0
fi

set +e
"${tool}" -- "${BENCH_COMMAND[@]}" >"${output}/bench.json" 2>"${output}/${tool}.log"
rc=$?
set -e
if (( rc == 0 )); then
  record_availability "${output}" "${tool}" available "system timeline"
else
  record_availability "${output}" "${tool}" unavailable "rc=${rc}; see ${tool}.log"
fi
echo "${tool} output: ${output} (rc=${rc})"
exit 0
