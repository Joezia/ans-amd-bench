#!/usr/bin/env bash
#SBATCH --job-name=h3-t4
#SBATCH --partition=mi3001x
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=results/slurm-h3-t4-%j.out
#SBATCH --error=results/slurm-h3-t4-%j.err
set -euo pipefail

bench_dir="${ANS_BENCH_ROOT:-${SLURM_SUBMIT_DIR:?Submit from ans-amd-bench}}"
cd "${bench_dir}"
machine="${LDS_MACHINE:?Set LDS_MACHINE=mi300x or mi350x}"
output="${bench_dir}/results/profiling/h3-t4-${machine}-${SLURM_JOB_ID}"
mkdir -p "${output}"

case "${machine}" in
  mi300x)
    export GPU_ARCH=gfx942 BUILD_PARTITION=mi3001x
    export HIPANS_BUILD_DIR="${bench_dir}/build-hipans-mi300"
    export HIPANS_LDS_BUILD_DIR="${bench_dir}/build-hipans-lds-mi300"
    export HIPANS_LDS_BINARY="${bench_dir}/bin/hipans_lds_bench"
    ;;
  mi350x)
    export GPU_ARCH=gfx950 BUILD_PARTITION=mi3501x
    export HIPANS_BUILD_DIR="${bench_dir}/build-hipans-mi350"
    export HIPANS_LDS_BUILD_DIR="${bench_dir}/build-hipans-lds-mi350"
    export HIPANS_LDS_BINARY="${bench_dir}/bin/hipans_lds_bench_mi350"
    ;;
  *) echo "Unsupported LDS_MACHINE=${machine}" >&2; exit 2 ;;
esac
export HIPANS_ROOT="${HIPANS_ROOT:-${bench_dir}/../hipANS}"
manifest="${bench_dir}/inputs/lds/glm52-layer0-gate-proj/mi350x/artifact.json"
if [[ ! -r "${manifest}" ]]; then
  echo "Missing device-independent GLM R2 artifact: ${manifest}" >&2
  exit 2
fi

echo "H3 T4 launch sweep: job=${SLURM_JOB_ID} machine=${machine} node=$(hostname) start=$(date -Iseconds)"
if [[ -f "${HIPANS_BUILD_DIR}/lib/libgpu_ans.a" ]]; then
  cmake --build "${HIPANS_BUILD_DIR}" --parallel 4 --target glog hipans_utils gpu_ans \
    >"${output}/native-build.log" 2>&1
else
  HIPANS_REPO="${HIPANS_ROOT}" BUILD_DIR="${HIPANS_BUILD_DIR}" BUILD_CPUS=4 \
    bash scripts/job_build_full.sh --inside-allocation >"${output}/native-build.log" 2>&1
fi
bash scripts/build_hipans_lds.sh >"${output}/adapter-build.log" 2>&1

python="${bench_dir}/.venv/bin/python"
"${python}" -m py_compile scripts/run_h3_t4_sweep.py scripts/run_lds_bench.py
"${python}" scripts/run_h3_t4_sweep.py \
  --manifest "${manifest}" \
  --output-dir "${output}/points" \
  --result "${bench_dir}/results/h3_t4_${machine}_${SLURM_JOB_ID}.json" \
  --report "${bench_dir}/docs/H3_T4_SWEEP_${machine^^}_${SLURM_JOB_ID}.md" \
  --job-id "${SLURM_JOB_ID}" --machine "${machine}"
echo "H3 T4 ${machine} complete: ${output}"
