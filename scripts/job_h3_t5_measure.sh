#!/usr/bin/env bash
#SBATCH --job-name=h3-t5-measure
#SBATCH --partition=mi3001x
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=results/slurm-h3-t5-measure-%j.out
#SBATCH --error=results/slurm-h3-t5-measure-%j.err
set -euo pipefail

bench_dir="${ANS_BENCH_ROOT:-${SLURM_SUBMIT_DIR:?Submit from ans-amd-bench}}"
cd "${bench_dir}"
machine="${LDS_MACHINE:?Set LDS_MACHINE=mi300x or mi350x}"
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
manifest="${H3_T5_ARTIFACT_DIR:?Set H3_T5_ARTIFACT_DIR}/artifact/artifact.json"
if [[ ! -r "${manifest}" ]]; then
  echo "Missing completed T5 artifact manifest: ${manifest}" >&2
  exit 2
fi
output="${bench_dir}/results/profiling/h3-t5-measure-${machine}-${SLURM_JOB_ID}"
mkdir -p "${output}"
echo "H3 T5 multi-matrix measurement: job=${SLURM_JOB_ID} machine=${machine} node=$(hostname) start=$(date -Iseconds)"
if [[ -f "${HIPANS_BUILD_DIR}/lib/libgpu_ans.a" ]]; then
  cmake --build "${HIPANS_BUILD_DIR}" --parallel 4 --target glog hipans_utils gpu_ans \
    >"${output}/native-build.log" 2>&1
else
  HIPANS_REPO="${HIPANS_ROOT}" BUILD_DIR="${HIPANS_BUILD_DIR}" BUILD_CPUS=4 \
    bash scripts/job_build_full.sh --inside-allocation >"${output}/native-build.log" 2>&1
fi
bash scripts/build_hipans_lds.sh >"${output}/adapter-build.log" 2>&1
python="${bench_dir}/.venv/bin/python"
"${python}" -m py_compile scripts/measure_h3_t5_multimatrix.py scripts/run_lds_bench.py
"${python}" scripts/measure_h3_t5_multimatrix.py \
  --manifest "${manifest}" --machine "${machine}" --job-id "${SLURM_JOB_ID}" \
  --threads "${H3_T5_THREADS:?Set T4-selected thread count}" \
  --grid "${H3_T5_GRID:?Set T4-selected grid multiplier}" \
  --streams "${H3_T5_STREAMS:?Set T4-selected stream count}" \
  --warmup 10 --iters 50 >"${output}/measurement.json"
echo "H3 T5 ${machine} measurement complete: ${output}"
