#!/usr/bin/env bash
#SBATCH --job-name=h3-t2
#SBATCH --partition=mi3001x
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=results/slurm-h3-t2-%j.out
#SBATCH --error=results/slurm-h3-t2-%j.err
set -euo pipefail

bench_dir="${ANS_BENCH_ROOT:-${SLURM_SUBMIT_DIR:?Submit from ans-amd-bench}}"
cd "${bench_dir}"
machine="${LDS_MACHINE:-mi300x}"
output="${bench_dir}/results/profiling/h3-t2-${machine}-${SLURM_JOB_ID}"
mkdir -p "${output}"
case "${machine}" in
  mi300x)
    export GPU_ARCH=gfx942 BUILD_PARTITION=mi3001x
    export HIPANS_BUILD_DIR="${bench_dir}/build-hipans-mi300"
    ;;
  mi350x)
    export GPU_ARCH=gfx950 BUILD_PARTITION=mi3501x
    export HIPANS_BUILD_DIR="${bench_dir}/build-hipans-mi350"
    ;;
  *) echo "Unsupported LDS_MACHINE=${machine}" >&2; exit 2 ;;
esac
export HIPANS_ROOT="${HIPANS_ROOT:-${bench_dir}/../hipANS}"
artifact_dir="${bench_dir}/inputs/lds/glm52-layer0-gate-proj/mi350x"
artifact="${artifact_dir}/weights.ans"
reference="${bench_dir}/inputs/lds/glm52-layer0-gate-proj/weights.bin"
if [[ ! -r "${artifact}" || ! -r "${reference}" ]]; then
  echo "Missing verified R2 artifact or reference input" >&2
  exit 2
fi
echo "H3 T2 ablations: job=${SLURM_JOB_ID} machine=${machine} node=$(hostname) start=$(date -Iseconds)"
if [[ -f "${HIPANS_BUILD_DIR}/lib/libgpu_ans.a" ]]; then
  cmake --build "${HIPANS_BUILD_DIR}" --parallel 4 --target glog hipans_utils gpu_ans \
    >"${output}/native-build.log" 2>&1
else
  HIPANS_REPO="${HIPANS_ROOT}" BUILD_DIR="${HIPANS_BUILD_DIR}" BUILD_CPUS=4 \
    bash scripts/job_build_full.sh --inside-allocation >"${output}/native-build.log" 2>&1
fi
LDS_MACHINE="${machine}" HIPANS_ROOT="${HIPANS_ROOT}" \
  bash scripts/build_hipans_lds_ablation.sh >"${output}/ablation-build.log" 2>&1
binary="${bench_dir}/bin/hipans_lds_ablation_${machine}"
"${binary}" --artifact "${artifact}" --reference "${reference}" \
  --rows 12288 --cols 6144 --warmup 10 --iters 50 --grid-mult 4 \
  >"${output}/benchmark.json"
echo "H3 T2 ${machine} complete: ${output}"

# The second machine job collects both raw results into the shared T2 report.
if [[ "${machine}" == mi350x ]]; then
  mi300_raw="${H3_T2_MI300X_RAW:?Set H3_T2_MI300X_RAW to the completed MI300X benchmark JSON}"
  if [[ ! -r "${mi300_raw}" ]]; then
    echo "MI300X result is not readable: ${mi300_raw}" >&2
    exit 2
  fi
  "${bench_dir}/.venv/bin/python" scripts/collect_h3_t2_results.py \
    --mi300x "${mi300_raw}" --mi350x "${output}/benchmark.json" \
    --mi300x-job "${H3_T2_MI300X_JOB:?Set H3_T2_MI300X_JOB}" \
    --mi350x-job "${SLURM_JOB_ID}"
fi
