#!/usr/bin/env bash
set -euo pipefail

bench_dir="${ANS_BENCH_ROOT:-${SLURM_SUBMIT_DIR:?Submit from ans-amd-bench}}"
cd "${bench_dir}"
source_file="${LDS_SOURCE:?Set LDS_SOURCE to the selected workload source.json}"
artifact_dir="${LDS_ARTIFACT_DIR:-${source_file%/*}/mi350x}"
output="${bench_dir}/results/profiling/lds-r1-mi350x-${SLURM_JOB_ID}"
mkdir -p "${output}"

export LDS_MACHINE=mi350x GPU_ARCH=gfx950 BUILD_PARTITION=mi3501x
export HIPANS_ROOT="${HIPANS_ROOT:-${bench_dir}/../hipANS}"
export HIPANS_BUILD_DIR="${bench_dir}/build-hipans-mi350"
export HIPANS_LDS_BUILD_DIR="${bench_dir}/build-hipans-lds-mi350"
export HIPANS_LDS_BINARY="${bench_dir}/bin/hipans_lds_bench_mi350"

echo "R1 GLM LDS verification: job=${SLURM_JOB_ID} node=$(hostname) start=$(date -Iseconds)"
rocminfo >"${output}/rocminfo.txt"
bash scripts/build_hipans_lds.sh >"${output}/adapter-build.log" 2>&1
"${bench_dir}/.venv/bin/python" scripts/prepare_lds_artifact.py \
  --source "${source_file}" --output-dir "${artifact_dir}" --replace \
  >"${output}/prepare.json"
"${bench_dir}/.venv/bin/python" - "${output}/prepare.json" <<'PY'
import json
import sys

result = json.load(open(sys.argv[1]))
if result.get("verified_actual_LDS") is not True:
    raise SystemExit("actual LDS byte-exact verification did not pass")
print(json.dumps({
    "verified_actual_LDS": result["verified_actual_LDS"],
    "unique_weight_bytes": result["unique_weight_bytes"],
    "artifact_bytes": result["artifact_bytes"],
    "artifact_sha256": result["artifact_sha256"],
    "decode_ms": result["decode_ms"],
    "provenance": result["provenance"],
}, sort_keys=True))
PY
echo "R1 actual LDS byte-exact verification complete: ${output}"
