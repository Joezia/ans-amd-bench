#!/usr/bin/env bash
set -euo pipefail

sepbase_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
bench_dir="$(cd -- "${sepbase_dir}/.." && pwd)"
repo_dir="${HIPANS_REPO:-${bench_dir}/../hipANS}"
hipans_build_dir="${HIPANS_BUILD_DIR:-${bench_dir}/build-hipans-full}"
library_dir="${hipans_build_dir}/lib"
glog_generated="${hipans_build_dir}/third_party/glog"
partition="${BUILD_PARTITION:-mi210_u280_u55c}"
build_time="${BUILD_TIME:-00:15:00}"

if [[ "${1:-}" != "--inside-allocation" ]]; then
  exec srun \
    --job-name=hipans-sepbase-build \
    --partition="${partition}" \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=2 \
    --time="${build_time}" \
    "${BASH_SOURCE[0]}" --inside-allocation
fi

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "ERROR: refusing to compile outside a Slurm allocation" >&2
  exit 2
fi
if [[ "$(hostname)" == hacchead* ]]; then
  echo "ERROR: refusing to compile on hacchead" >&2
  exit 2
fi

hipcc_bin="${HIPCC_BIN:-/usr/bin/hipcc}"
output_dir="${sepbase_dir}/bin"
output_bin="${output_dir}/sepbase_bench"
mkdir -p "${output_dir}"

required=(
  "${hipcc_bin}"
  "${sepbase_dir}/sepbase_bench.cpp"
  "${library_dir}/libgpu_ans.a"
  "${library_dir}/libhipans_utils.a"
  "${library_dir}/libglog.a"
  /opt/rocm/include/hipblaslt/hipblaslt.h
  /opt/rocm/include/rocblas/rocblas.h
  /opt/rocm/lib/libhipblaslt.so
  /opt/rocm/lib/librocblas.so
)
for path in "${required[@]}"; do
  if [[ ! -e "${path}" ]]; then
    echo "ERROR: required build input is missing: ${path}" >&2
    exit 2
  fi
done

compile=(
  "${hipcc_bin}"
  -std=c++17
  -O3
  -DNDEBUG
  --offload-arch=gfx90a
  -DGFLAGS_IS_A_DLL=0
  "-DHIPANS_SOURCE_DIR=\"${repo_dir}\""
  "-DBENCH_SOURCE_DIR=\"${bench_dir}\""
  -I"${repo_dir}"
  -I"${glog_generated}"
  -I"${repo_dir}/third_party/glog/src"
  -I/opt/rocm/include
  "${sepbase_dir}/sepbase_bench.cpp"
  -x none
  "${library_dir}/libgpu_ans.a"
  "${library_dir}/libhipans_utils.a"
  "${library_dir}/libglog.a"
  -L/opt/rocm/lib
  -Wl,-rpath,/opt/rocm/lib
  -lhipblaslt
  -lrocblas
  -pthread
  -o "${output_bin}"
)

echo "hostname: $(hostname)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"
echo "COMPLETE BUILD COMMAND:"
printf '%q ' "${compile[@]}"
printf '\n'
"${compile[@]}"

if [[ ! -x "${output_bin}" ]]; then
  echo "ERROR: compiler returned success but binary is missing: ${output_bin}" >&2
  exit 1
fi
echo "built: ${output_bin}"
