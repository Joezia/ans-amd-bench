#!/usr/bin/env bash
set -euo pipefail

bench_dir="${ANS_BENCH_ROOT:-${SLURM_SUBMIT_DIR:?Submit from ans-amd-bench}}"
cd "${bench_dir}"
machine="${LDS_MACHINE:?Set LDS_MACHINE=mi300x or mi350x}"
artifact_dir="${LDS_ARTIFACT_DIR:-${bench_dir}/inputs/lds/glm52-layer0-gate-proj/mi350x}"
manifest="${artifact_dir}/artifact.json"
output="${bench_dir}/results/profiling/lds-r3-glm-${machine}-${SLURM_JOB_ID}"
mkdir -p "${output}"

case "${machine}" in
  mi350x)
    export GPU_ARCH=gfx950 BUILD_PARTITION=mi3501x
    export HIPANS_BUILD_DIR="${bench_dir}/build-hipans-mi350"
    export HIPANS_LDS_BUILD_DIR="${bench_dir}/build-hipans-lds-mi350"
    export HIPANS_LDS_BINARY="${bench_dir}/bin/hipans_lds_bench_mi350"
    ;;
  mi300x)
    export GPU_ARCH=gfx942 BUILD_PARTITION=mi3001x
    export HIPANS_BUILD_DIR="${bench_dir}/build-hipans-mi300"
    export HIPANS_LDS_BUILD_DIR="${bench_dir}/build-hipans-lds-mi300"
    export HIPANS_LDS_BINARY="${bench_dir}/bin/hipans_lds_bench"
    ;;
  *) echo "Unsupported LDS_MACHINE=${machine}" >&2; exit 2 ;;
esac
export HIPANS_ROOT="${HIPANS_ROOT:-${bench_dir}/../hipANS}"
export HIPANS_LDS_ISA="${HIPANS_LDS_BUILD_DIR}/hipans_lds_bench-hip-amdgcn-amd-amdhsa-${GPU_ARCH}.s"

if [[ ! -r "${manifest}" ]]; then
  echo "Missing scenario artifact manifest: ${manifest}" >&2
  exit 2
fi

echo "R3 GLM scenario profile: job=${SLURM_JOB_ID} machine=${machine} node=$(hostname) start=$(date -Iseconds)"
rocminfo >"${output}/rocminfo.txt"
if [[ -f "${HIPANS_BUILD_DIR}/lib/libgpu_ans.a" ]]; then
  cmake --build "${HIPANS_BUILD_DIR}" --parallel 4 --target glog hipans_utils gpu_ans \
    >"${output}/native-build.log" 2>&1
else
  HIPANS_REPO="${HIPANS_ROOT}" BUILD_DIR="${HIPANS_BUILD_DIR}" BUILD_CPUS=4 \
    bash scripts/job_build_full.sh --inside-allocation >"${output}/native-build.log" 2>&1
fi
bash scripts/build_hipans_lds.sh >"${output}/adapter-build.log" 2>&1

# R3 profiles the verified R2 primary case. Instrumented times are diagnostic only.
export PROFILE_LDS_REPLICAS=1 PROFILE_LDS_THREADS=128 PROFILE_LDS_GRID_MULT=4
export PROFILE_LDS_STREAMS=1 PROFILE_LDS_WARMUP=2 PROFILE_LDS_ITERS=1
bash profiling/run_lds_baseline.sh "${manifest}" "${output}/rocprof"

# rocprof-sys/Omnitrace is optional; retain its failure as availability metadata.
timeline_tool=""
if command -v rocprof-sys >/dev/null 2>&1; then
  timeline_tool=rocprof-sys
elif command -v omnitrace >/dev/null 2>&1; then
  timeline_tool=omnitrace
fi
if [[ -n "${timeline_tool}" ]]; then
  mkdir -p "${output}/system-timeline"
  set +e
  "${timeline_tool}" -- "${bench_dir}/.venv/bin/python" \
    scripts/run_lds_bench.py --manifest "${manifest}" --mode decode --replicas 1 \
    --threads 128 --grid-mult 4 --streams 1 --skip-verify --warmup 2 --iters 1 \
    >"${output}/system-timeline/bench.json" \
    2>"${output}/system-timeline/${timeline_tool}.log"
  timeline_rc=$?
  set -e
  if (( timeline_rc == 0 )); then
    printf 'rocprof-sys\tavailable\t%s system timeline\n' "${timeline_tool}" \
      >>"${output}/rocprof/availability.tsv"
  else
    printf 'rocprof-sys\tunavailable\t%s failed with rc=%s; see system-timeline log\n' \
      "${timeline_tool}" "${timeline_rc}" >>"${output}/rocprof/availability.tsv"
  fi
else
  printf 'rocprof-sys\tunavailable\tneither rocprof-sys nor omnitrace is installed\n' \
    >>"${output}/rocprof/availability.tsv"
fi

printf 'R3 scenario profiling complete: %s\n' "${output}"
