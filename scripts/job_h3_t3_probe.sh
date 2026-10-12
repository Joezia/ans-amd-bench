#!/usr/bin/env bash
#SBATCH --job-name=h3-t3-probe
#SBATCH --partition=mi3001x
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=results/slurm-h3-t3-%j.out
#SBATCH --error=results/slurm-h3-t3-%j.err
set -uo pipefail

bench_dir="${ANS_BENCH_ROOT:-${SLURM_SUBMIT_DIR:?Submit from ans-amd-bench}}"
cd "${bench_dir}"
machine="${LDS_MACHINE:?Set LDS_MACHINE=mi300x or mi350x}"
output="${bench_dir}/results/profiling/h3-t3-${machine}-${SLURM_JOB_ID}"
mkdir -p "${output}"

case "${machine}" in
  mi300x)
    partition=mi3001x
    export GPU_ARCH=gfx942 BUILD_PARTITION="${partition}"
    export HIPANS_BUILD_DIR="${bench_dir}/build-hipans-mi300"
    export HIPANS_LDS_BUILD_DIR="${bench_dir}/build-hipans-lds-mi300"
    export HIPANS_LDS_BINARY="${bench_dir}/bin/hipans_lds_bench"
    artifact_dir="${bench_dir}/inputs/lds/glm52-layer0-gate-proj/mi300x"
    ;;
  mi350x)
    partition=mi3501x
    export GPU_ARCH=gfx950 BUILD_PARTITION="${partition}"
    export HIPANS_BUILD_DIR="${bench_dir}/build-hipans-mi350"
    export HIPANS_LDS_BUILD_DIR="${bench_dir}/build-hipans-lds-mi350"
    export HIPANS_LDS_BINARY="${bench_dir}/bin/hipans_lds_bench_mi350"
    artifact_dir="${bench_dir}/inputs/lds/glm52-layer0-gate-proj/mi350x"
    ;;
  *) echo "Unsupported LDS_MACHINE=${machine}" >&2; exit 2 ;;
esac
export HIPANS_ROOT="${HIPANS_ROOT:-${bench_dir}/../hipANS}"
manifest="${artifact_dir}/artifact.json"

echo "H3 T3 profiler probe: job=${SLURM_JOB_ID} machine=${machine} partition=${partition} node=$(hostname) start=$(date -Iseconds)"

{
  echo "machine=${machine}"
  echo "node=$(hostname)"
  echo "ROCm_VISIBLE_DEVICES=${ROCR_VISIBLE_DEVICES:-unset}"
  rocminfo | sed -n '1,100p'
} >"${output}/device.txt" 2>&1 || true

probe() {
  local name="$1"
  shift
  set +e
  "$@" >"${output}/${name}.txt" 2>&1
  local rc=$?
  set -e
  printf '%s\t%s\t%s\n' "${name}" "$((rc == 0 ? 1 : 0))" "rc=${rc}; see ${name}.txt" \
    >>"${output}/tool-availability.tsv"
}

printf 'probe\tavailable\tdetail\n' >"${output}/tool-availability.tsv"
probe rocprofv3-version rocprofv3 --version
probe rocprofv3-help rocprofv3 --help
probe rocprofv3-list-avail rocprofv3 --list-avail
probe rocprof-compute-version rocprof-compute --version
probe rocprof-compute-help rocprof-compute --help
probe rocprof-compute-profile-help rocprof-compute profile --help
probe rocprof-compute-list-metrics rocprof-compute --list-metrics "${GPU_ARCH}"
probe rocprof-compute-list-blocks rocprof-compute --list-blocks "${GPU_ARCH}"

# Re-run the known one-matrix R3 case with the existing counter blocks. These
# counters are diagnostic only; uninstrumented R2 HIP-event measurements remain
# the performance authority. R3 artifacts and R2 inputs are left untouched.
if [[ -r "${manifest}" && -x "${HIPANS_LDS_BINARY}" && -x "${bench_dir}/.venv/bin/python" ]]; then
  export PROFILE_LDS_REPLICAS=1 PROFILE_LDS_THREADS=128 PROFILE_LDS_GRID_MULT=4
  export PROFILE_LDS_STREAMS=1 PROFILE_LDS_WARMUP=2 PROFILE_LDS_ITERS=1
  if bash profiling/run_lds_baseline.sh "${manifest}" "${output}/r3" \
      >"${output}/profile-launch.log" 2>&1; then
    printf 'r3-counter-replay\t1\tprofile output in r3/\n' >>"${output}/tool-availability.tsv"
    if PROFILE_COMPUTE_PYTHON="${bench_dir}/.venv/rocprof-compute/bin/python" \
        bash scripts/analyze_lds_profile.sh "${output}/r3" \
        >"${output}/analysis-launch.log" 2>&1; then
      printf 'r3-counter-analysis\t1\tselected decoder analysis in r3/analysis.txt\n' \
        >>"${output}/tool-availability.tsv"
    else
      rc=$?
      printf 'r3-counter-analysis\t0\trc=%s; see analysis-launch.log\n' "${rc}" \
        >>"${output}/tool-availability.tsv"
    fi
  else
    rc=$?
    printf 'r3-counter-replay\t0\trc=%s; see profile-launch.log\n' "${rc}" \
      >>"${output}/tool-availability.tsv"
  fi
else
  if [[ ! -r "${manifest}" ]]; then
    printf 'r3-counter-replay\t0\tskipped: missing R2 manifest %s; use existing R3 capture\n' \
      "${manifest}" >>"${output}/tool-availability.tsv"
  elif [[ ! -x "${HIPANS_LDS_BINARY}" ]]; then
    printf 'r3-counter-replay\t0\tmissing existing LDS binary %s\n' \
      "${HIPANS_LDS_BINARY}" >>"${output}/tool-availability.tsv"
  else
    printf 'r3-counter-replay\t0\tmissing local venv Python %s\n' \
      "${bench_dir}/.venv/bin/python" >>"${output}/tool-availability.tsv"
  fi
fi

echo "H3 T3 profiler probe complete: ${output}"
