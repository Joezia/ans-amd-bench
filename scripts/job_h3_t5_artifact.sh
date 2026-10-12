#!/usr/bin/env bash
#SBATCH --job-name=h3-t5-artifact
#SBATCH --partition=mi3501x
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=results/slurm-h3-t5-artifact-%j.out
#SBATCH --error=results/slurm-h3-t5-artifact-%j.err
set -euo pipefail

bench_dir="${ANS_BENCH_ROOT:-${SLURM_SUBMIT_DIR:?Submit from ans-amd-bench}}"
cd "${bench_dir}"
export GPU_ARCH=gfx950 BUILD_PARTITION=mi3501x
export HIPANS_BUILD_DIR="${bench_dir}/build-hipans-mi350"
export HIPANS_LDS_BUILD_DIR="${bench_dir}/build-hipans-lds-mi350"
export HIPANS_LDS_BINARY="${bench_dir}/bin/hipans_lds_bench_mi350"
export HIPANS_ROOT="${HIPANS_ROOT:-${bench_dir}/../hipANS}"
output_root="${bench_dir}/inputs/lds/glm52-first3-dense-stack/job-${SLURM_JOB_ID}"
mkdir -p "${bench_dir}/results/profiling/h3-t5-artifact-${SLURM_JOB_ID}"
logs="${bench_dir}/results/profiling/h3-t5-artifact-${SLURM_JOB_ID}"

echo "H3 T5 artifact preparation: job=${SLURM_JOB_ID} node=$(hostname) start=$(date -Iseconds)"
if [[ -f "${HIPANS_BUILD_DIR}/lib/libgpu_ans.a" ]]; then
  cmake --build "${HIPANS_BUILD_DIR}" --parallel 4 --target glog hipans_utils gpu_ans \
    >"${logs}/native-build.log" 2>&1
else
  HIPANS_REPO="${HIPANS_ROOT}" BUILD_DIR="${HIPANS_BUILD_DIR}" BUILD_CPUS=4 \
    bash scripts/job_build_full.sh --inside-allocation >"${logs}/native-build.log" 2>&1
fi
bash scripts/build_hipans_lds.sh >"${logs}/adapter-build.log" 2>&1
python="${bench_dir}/.venv/bin/python"
"${python}" -m py_compile scripts/prepare_h3_t5_multimatrix.py scripts/fetch_lds_weights.py scripts/prepare_lds_artifact.py
"${python}" scripts/prepare_h3_t5_multimatrix.py \
  --output-root "${output_root}" --job-id "${SLURM_JOB_ID}" \
  >"${logs}/artifact-preparation.json"
echo "H3 T5 artifact preparation complete: ${output_root}"
