#!/usr/bin/env bash
set -euo pipefail
bench_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
root="${1:?Usage: analyze_lds_profile.sh results/profiling/lds-JOBID}"
root="$(cd "${root}" && pwd)"
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  exec srun -p "${BUILD_PARTITION:-devel}" -N1 -n1 -c2 -t 00:05:00 bash "${BASH_SOURCE[0]}" "${root}"
fi
# Counter CSV dispatch IDs, not kernel IDs: select the last decoder after warmup.
dispatch="$("${bench_dir}/.venv/bin/python" -c '
import csv,sys
from pathlib import Path
ids={int(r["Dispatch_ID"]) for p in Path(sys.argv[1]).glob("results_pmc_perf_*.csv")
     for r in csv.DictReader(p.open()) if "ldsBaseline<false, true, 0>" in r["Kernel_Name"]}
if not ids: raise SystemExit("no decoder counter dispatch")
print(max(ids))' "${root}/profile/counters/0")"
compute="$(readlink -f "$(command -v rocprof-compute)")"
python="${PROFILE_COMPUTE_PYTHON:-${bench_dir}/.venv/rocprof-compute/bin/python}"
"${python}" "${compute}" analyze -p "${root}/profile/counters/0" \
  --dispatch "${dispatch}" --output-format stdout >"${root}/analysis.txt" 2>"${root}/analysis.err"
printf '%s\n' "${dispatch}" >"${root}/analysis-dispatch.txt"
