#!/usr/bin/env bash
set -euo pipefail
bench_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
root="${1:?Usage: analyze_lds_profile.sh new-profile-directory}"
root="$(cd "${root}" && pwd)"
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  exec srun -p "${BUILD_PARTITION:-devel}" -N1 -n1 -c2 -t 00:05:00 bash "${BASH_SOURCE[0]}" "${root}"
fi
# Single-queue saturated baseline only: last timed native decoder, not encode,
# table preparation, verification, warmup or concurrent replay averages.
dispatch="$("${bench_dir}/.venv/bin/python" -c '
import csv,re,sys
from pathlib import Path
ids={int(r["Dispatch_ID"]) for p in Path(sys.argv[1]).glob("results_pmc_perf_*.csv")
     for r in csv.DictReader(p.open())
     if re.search(r"hipansLDS<\s*\d+,\s*false,\s*false\s*>", r["Kernel_Name"])
        or re.search(r"hipansLDSILi\d+ELb0ELb0EE", r["Kernel_Name"])}
if not ids: raise SystemExit("no actual hipANS decoder counter dispatch")
print(max(ids))' "${root}/counters/0")"
compute="$(readlink -f "$(command -v rocprof-compute)")"
python="${PROFILE_COMPUTE_PYTHON:-${bench_dir}/.venv/rocprof-compute/bin/python}"
"${python}" "${compute}" analyze -p "${root}/counters/0" \
  --dispatch "${dispatch}" --output-format stdout >"${root}/analysis.txt" 2>"${root}/analysis.err"
printf '%s\n' "${dispatch}" >"${root}/analysis-dispatch.txt"
