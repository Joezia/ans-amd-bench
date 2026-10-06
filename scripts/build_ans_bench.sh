#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
bench_dir="$(cd -- "${script_dir}/.." && pwd)"
hipans_root="${HIPANS_ROOT:-${bench_dir}/../hipANS}"
hipans_build_dir="${HIPANS_BUILD_DIR:-${bench_dir}/build-hipans-mi300}"
gpu_arch="${GPU_ARCH:-gfx942}"
partition="${BUILD_PARTITION:-mi3001x}"
build_time="${BUILD_TIME:-00:15:00}"

if [[ "${1:-}" != "--inside-allocation" && -z "${SLURM_JOB_ID:-}" ]]; then
  exec srun --job-name=ans-bench-build --partition="${partition}" \
    --nodes=1 --ntasks=1 --cpus-per-task=2 --time="${build_time}" \
    env BUILD_PARTITION="${partition}" "${BASH_SOURCE[0]}" --inside-allocation
fi

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "ERROR: build must run in a Slurm allocation" >&2
  exit 2
fi

hipcc_bin="${HIPCC_BIN:-$(command -v hipcc || true)}"
library_dir="${hipans_build_dir}/lib"
glog_generated="${hipans_build_dir}/third_party/glog"
output_dir="${bench_dir}/bin"
output="${output_dir}/ans_bench"
mkdir -p "${output_dir}"

required=(
  "${hipcc_bin}"
  "${hipans_root}/hipans/ans/GpuANSCodec.h"
  "${library_dir}/libgpu_ans.a"
  "${library_dir}/libhipans_utils.a"
  "${library_dir}/libglog.a"
)
for path in "${required[@]}"; do
  if [[ ! -e "${path}" ]]; then
    echo "ERROR: missing build input: ${path}" >&2
    exit 2
  fi
done

build_flags="-std=c++17 -O3 -DNDEBUG --offload-arch=${gpu_arch} -pthread"
compile=(
  "${hipcc_bin}" -std=c++17 -O3 -DNDEBUG --offload-arch="${gpu_arch}"
  "-DHIPANS_GPU_ARCH=\"${gpu_arch}\""
  "-DHIPANS_SOURCE_DIR=\"${hipans_root}\""
  "-DANS_BENCH_BUILD_FLAGS=\"${build_flags}\""
  -DGFLAGS_IS_A_DLL=0
  -I"${hipans_root}" -I"${glog_generated}"
  -I"${hipans_root}/third_party/glog/src"
  "${bench_dir}/bench/ans_bench.cpp" -x none
  "${library_dir}/libgpu_ans.a" "${library_dir}/libhipans_utils.a"
  "${library_dir}/libglog.a" -pthread -o "${output}"
)

echo "hipANS source (read-only): ${hipans_root}"
echo "hipANS out-of-tree build: ${hipans_build_dir}"
printf 'BUILD COMMAND: '
printf '%q ' "${compile[@]}"
printf '\n'
"${compile[@]}"
echo "built ${output}"
