#!/usr/bin/env bash
set -euo pipefail

profiling_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
bench_dir="$(cd -- "${profiling_dir}/.." && pwd)"
hipans_root="${HIPANS_ROOT:-${bench_dir}/../hipANS}"
base_build="${HIPANS_BUILD_DIR:-${bench_dir}/build-hipans-mi300}"
gpu_arch="${GPU_ARCH:-gfx942}"
partition="${BUILD_PARTITION:-mi3001x}"
variants="${LAUNCH_VARIANTS:-64x2 64x4 128x2 128x4 128x8}"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  exec srun --job-name=ans-launch-build --partition="${partition}" \
    --nodes=1 --ntasks=1 --cpus-per-task="${BUILD_CPUS:-8}" \
    --time="${BUILD_TIME:-01:00:00}" "${BASH_SOURCE[0]}"
fi

hipcc_bin="${HIPCC_BIN:-$(command -v hipcc)}"
cmake_bin="${CMAKE_BIN:-$(command -v cmake)}"
glog_generated="${base_build}/third_party/glog"
mkdir -p "${bench_dir}/bin"
variant_dir="${bench_dir}/build-launch-variants"
overlay="${variant_dir}/overlay/hipans/ans"
mkdir -p "${overlay}"
cp "${hipans_root}/hipans/ans/GpuANSDecode.h" "${overlay}/GpuANSDecode.h"

"${cmake_bin}" --fresh -S "${hipans_root}" -B "${variant_dir}/build" \
  -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER="${hipcc_bin}" \
  -DCMAKE_HIP_COMPILER="$(readlink -f "${hipcc_bin}" | sed 's@/bin/hipcc$@/lib/llvm/bin/clang++@')" \
  -DCMAKE_HIP_ARCHITECTURES="${gpu_arch}" \
  "-DCMAKE_CXX_FLAGS=-I${variant_dir}/overlay --offload-arch=${gpu_arch}" \
  -DBUILD_SHARED_LIBS=OFF -DHIPANS_LIBRARY_TYPE=STATIC \
  -DHIPANS_BUILD_FULL_LIBRARY=OFF

for variant in ${variants}; do
  threads="${variant%x*}"
  grid="${variant#*x}"
  if [[ "${threads}" != 64 && "${threads}" != 128 ]]; then
    echo "ERROR: diagnostic threads must be 64 or 128, got ${threads}" >&2
    exit 2
  fi
  if [[ ! "${grid}" =~ ^[1248]$ ]]; then
    echo "ERROR: diagnostic grid multiplier must be 1,2,4,8, got ${grid}" >&2
    exit 2
  fi

  cp "${hipans_root}/hipans/ans/GpuANSDecode.h" "${overlay}/GpuANSDecode.h"

  # Only launch geometry changes.  The rANS block size, state transition,
  # archive format, table construction, and all memory operations are intact.
  perl -0pi -e "s/__global__ __launch_bounds__\(128\) void ansDecodeKernel/__global__ __launch_bounds__(${threads}) void ansDecodeKernel/" \
    "${overlay}/GpuANSDecode.h"
  perl -0pi -e "s/constexpr int kThreads = 128;/constexpr int kThreads = ${threads};/" \
    "${overlay}/GpuANSDecode.h"
  perl -0pi -e "s/divUp\(maxGrid, numInBatch\) \* 4/divUp(maxGrid, numInBatch) * ${grid}/" \
    "${overlay}/GpuANSDecode.h"

  grep -q "__launch_bounds__(${threads}) void ansDecodeKernel" "${overlay}/GpuANSDecode.h"
  grep -q "constexpr int kThreads = ${threads};" "${overlay}/GpuANSDecode.h"
  grep -q "divUp(maxGrid, numInBatch) \* ${grid}" "${overlay}/GpuANSDecode.h"

  # The decoder implementation includes this header through an overlay.  Some
  # generated Make dependency files retain the original source-tree header,
  # so an ordinary incremental build can silently reuse the previous variant.
  # Clean the isolated diagnostic tree to make each geometry unambiguous.
  "${cmake_bin}" --build "${variant_dir}/build" --target clean
  "${cmake_bin}" --build "${variant_dir}/build" --parallel "${BUILD_CPUS:-8}" \
    --target gpu_ans

  dependency_hits="$(grep -Rlc "${variant_dir}/overlay/hipans/ans/GpuANSDecode.h" \
    "${variant_dir}/build/hipans/ans/CMakeFiles/gpu_ans.dir" || true)"
  if [[ "${dependency_hits}" == 0 ]]; then
    echo "ERROR: compiler dependency files do not prove overlay use for ${variant}" >&2
    exit 1
  fi

  build_flags="-std=c++17 -O3 -DNDEBUG --offload-arch=${gpu_arch} decode_threads=${threads} grid_multiplier=${grid}"
  output="${bench_dir}/bin/ans_bench-t${threads}-g${grid}"
  "${hipcc_bin}" -std=c++17 -O3 -DNDEBUG --offload-arch="${gpu_arch}" \
    "-DHIPANS_GPU_ARCH=\"${gpu_arch}\"" \
    "-DHIPANS_SOURCE_DIR=\"${hipans_root}\"" \
    "-DANS_BENCH_BUILD_FLAGS=\"${build_flags}\"" \
    "-DHIPANS_DECODE_THREADS=${threads}" \
    "-DHIPANS_GRID_OVERSUBSCRIPTION=${grid}" -DGFLAGS_IS_A_DLL=0 \
    -I"${hipans_root}" -I"${glog_generated}" \
    -I"${hipans_root}/third_party/glog/src" \
    "${bench_dir}/bench/ans_bench.cpp" -x none \
    "${variant_dir}/build/lib/libgpu_ans.a" \
    "${base_build}/lib/libhipans_utils.a" "${base_build}/lib/libglog.a" \
    -pthread -o "${output}"
  echo "built diagnostic launch variant: ${output}"
done
