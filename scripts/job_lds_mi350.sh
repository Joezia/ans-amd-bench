#!/usr/bin/env bash
# Independent gfx950 build, artifact and profiling results; never overwrite MI300X.
set -euo pipefail
bench_dir="${ANS_BENCH_ROOT:-${SLURM_SUBMIT_DIR:?Submit from ans-amd-bench}}"
cd "${bench_dir}"
export LDS_MACHINE=mi350x GPU_ARCH=gfx950 BUILD_PARTITION=mi3501x
export HIPANS_ROOT="${HIPANS_ROOT:-${bench_dir}/../hipANS}"
export HIPANS_BUILD_DIR="${bench_dir}/build-hipans-mi350"
export HIPANS_LDS_BUILD_DIR="${bench_dir}/build-hipans-lds-mi350"
export HIPANS_LDS_BINARY="${bench_dir}/bin/hipans_lds_bench_mi350"
export HIPANS_LDS_ISA="${HIPANS_LDS_BUILD_DIR}/hipans_lds_bench-hip-amdgcn-amd-amdhsa-gfx950.s"
export LDS_RESULT_FILE="${bench_dir}/results/lds_baseline_mi350x.json"
export LDS_REPORT_FILE="${bench_dir}/docs/LDS_BASELINE_REPORT_MI350X.md"
source_file="${LDS_SOURCE:-${bench_dir}/inputs/lds/dsv4-flash-base-expert10-w1/source.json}"
artifact_dir="${source_file%/*}/mi350x"
output="${bench_dir}/results/profiling/hipans-lds-mi350x-${SLURM_JOB_ID}"
mkdir -p "${output}"
echo "MI350X job ${SLURM_JOB_ID}; node $(hostname); $(date -Iseconds)"
rocminfo >"${output}/rocminfo.txt"
if [[ ! -f "${HIPANS_BUILD_DIR}/lib/libgpu_ans.a" ]]; then
  HIPANS_REPO="${HIPANS_ROOT}" BUILD_DIR="${HIPANS_BUILD_DIR}" BUILD_CPUS=4 \
    bash scripts/job_build_full.sh --inside-allocation >"${output}/native-build.log" 2>&1
else
  cmake --build "${HIPANS_BUILD_DIR}" --parallel 4 --target glog hipans_utils gpu_ans \
    >"${output}/native-build.log" 2>&1
fi
bash scripts/build_hipans_lds.sh >"${output}/adapter-build.log" 2>&1
python="${bench_dir}/.venv/bin/python"
"${python}" scripts/prepare_lds_artifact.py --source "${source_file}" \
  --output-dir "${artifact_dir}" --replace >"${output}/prepare.json"
"${python}" scripts/sweep_hipans_lds.py --manifest "${artifact_dir}/artifact.json" --output "${output}/sweep"
echo "MI350X verified measurement/profiling finished: ${output}"
