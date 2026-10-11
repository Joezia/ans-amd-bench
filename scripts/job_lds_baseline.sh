#!/usr/bin/env bash
set -euo pipefail
bench_dir="${ANS_BENCH_ROOT:-${SLURM_SUBMIT_DIR:-}}"
if [[ -z "${bench_dir}" || ! -f "${bench_dir}/bench/hipans_lds_bench.cpp" ]]; then
  echo "Submit from ans-amd-bench or set ANS_BENCH_ROOT" >&2
  exit 2
fi
cd "${bench_dir}"
source_file="${LDS_SOURCE:-${bench_dir}/inputs/lds/dsv4-flash-base-expert10-w1/source.json}"
manifest="${source_file%/*}/artifact.json"
output="${bench_dir}/results/profiling/hipans-lds-${SLURM_JOB_ID}"
mkdir -p "${output}"
python="${bench_dir}/.venv/bin/python"
"${python}" scripts/prepare_lds_artifact.py --source "${source_file}" --replace >"${output}/prepare.json"
"${python}" scripts/sweep_hipans_lds.py --manifest "${manifest}" --output "${output}/sweep"
echo "Verified actual-hipANS baseline complete: ${output}"
