#!/usr/bin/env bash
# Configure + build hipANS on a ROCm compute node, in a build dir OUTSIDE the repo.
# Does not modify any file inside the hipANS source tree.
#
# By default, build only the static libraries required by the raw decode harness.
# Set BUILD_FULL_LIBRARY=ON explicitly when the PyTorch/float targets are needed.
set -uo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
bench_dir="$(cd -- "${script_dir}/.." && pwd)"
repo_dir="${HIPANS_REPO:-${bench_dir}/../hipANS}"
build_dir="${BUILD_DIR:-${bench_dir}/build-hipans-full}"
build_cpus="${BUILD_CPUS:-8}"
gpu_arch="${GPU_ARCH:?GPU_ARCH must be set explicitly}"
partition="${BUILD_PARTITION:-}"
build_time="${BUILD_TIME:-00:20:00}"
full_mode="${BUILD_FULL_LIBRARY:-OFF}"

if [[ "${1:-}" != "--inside-allocation" && -z "${SLURM_JOB_ID:-}" ]]; then
  if [[ -z "${partition}" ]]; then
    build_partitions="${BUILD_PARTITIONS:?set BUILD_PARTITION or BUILD_PARTITIONS}"
    partition="$("${script_dir}/select_slurm_partition.sh" "${build_partitions}" 0)"
  fi
  echo "selected build partition: ${partition}"
  exec srun \
    --job-name=hipans-full-build \
    --partition="${partition}" \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task="${build_cpus}" \
    --time="${build_time}" \
    env BUILD_PARTITION="${partition}" \
    "${BASH_SOURCE[0]}" --inside-allocation
fi

if [[ -z "${SLURM_JOB_ID:-}" || "$(hostname)" == hacchead* ]]; then
  echo "ERROR: refusing to build outside a Slurm compute-node allocation" >&2
  exit 2
fi

partition="${BUILD_PARTITION:?BUILD_PARTITION missing inside allocation}"
cmake_bin="${CMAKE_BIN:-$(command -v cmake || true)}"
hipcc_bin="${HIPCC_BIN:-$(command -v hipcc || true)}"
if [[ -z "${cmake_bin}" || ! -x "${cmake_bin}" ]]; then
  echo "ERROR: cmake not found; set CMAKE_BIN" >&2
  exit 2
fi
if [[ -z "${hipcc_bin}" || ! -x "${hipcc_bin}" ]]; then
  echo "ERROR: hipcc not found; set HIPCC_BIN" >&2
  exit 2
fi

if [[ -n "${HIP_COMPILER_BIN:-}" ]]; then
  hip_compiler_bin="${HIP_COMPILER_BIN}"
else
  hipcc_real="$(readlink -f "${hipcc_bin}")"
  rocm_root="$(cd -- "$(dirname -- "${hipcc_real}")/.." && pwd)"
  hip_compiler_bin=""
  for candidate in \
    "${rocm_root}/lib/llvm/bin/clang++" \
    "${rocm_root}/llvm/bin/clang++"; do
    if [[ -x "${candidate}" ]]; then
      hip_compiler_bin="${candidate}"
      break
    fi
  done
fi
if [[ -z "${hip_compiler_bin}" || ! -x "${hip_compiler_bin}" ]]; then
  echo "ERROR: HIP clang++ not found; set HIP_COMPILER_BIN" >&2
  exit 2
fi

echo "##### build job #####"
echo "node: $(hostname)  job: ${SLURM_JOB_ID:-unset}  date: $(date -Iseconds)"
echo "repo:      ${repo_dir}"
echo "build dir: ${build_dir}"
echo "cmake:     ${cmake_bin} ($(${cmake_bin} --version | head -1))"
echo "hipcc:     ${hipcc_bin}"
echo "HIP clang: ${hip_compiler_bin}"
echo "arch:      ${gpu_arch}"
echo "torch cmake prefix: $(python3 -c 'import torch; print(torch.utils.cmake_prefix_path)' 2>/dev/null)"
echo

configure() {
  local full="$1"
  echo "##### cmake configure (HIPANS_BUILD_FULL_LIBRARY=${full}) #####"
  set -x
  "${cmake_bin}" \
    --fresh \
    -S "${repo_dir}" \
    -B "${build_dir}" \
    -G "Unix Makefiles" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_COMPILER="${hipcc_bin}" \
    -DCMAKE_HIP_COMPILER="${hip_compiler_bin}" \
    -DCMAKE_HIP_ARCHITECTURES="${gpu_arch}" \
    "-DCMAKE_CXX_FLAGS=--offload-arch=${gpu_arch}" \
    -DBUILD_SHARED_LIBS=OFF \
    -DHIPANS_LIBRARY_TYPE=STATIC \
    -DHIPANS_BUILD_FULL_LIBRARY="${full}"
  local rc=$?
  set +x
  echo "configure rc=${rc}"
  return ${rc}
}

configure "${full_mode}" || { echo "configure failed"; exit 1; }
echo "FULL_LIBRARY mode actually used: ${full_mode}"
echo

echo "##### cache: key variables #####"
grep -E "^(CMAKE_HIP_ARCHITECTURES|CMAKE_CXX_COMPILER|CMAKE_HIP_COMPILER|CMAKE_BUILD_TYPE|BUILD_SHARED_LIBS|HIPANS_LIBRARY_TYPE|HIPANS_BUILD_FULL_LIBRARY|HIP_DIR|HIP_PATH|Torch_DIR|TORCH_LIBRARY)" "${build_dir}/CMakeCache.txt"
echo

echo "##### cmake --build (required static-library targets) #####"
set -x
"${cmake_bin}" --build "${build_dir}" --parallel "${build_cpus}" \
  --target glog hipans_utils gpu_ans
build_rc=$?
set +x
echo "build rc=${build_rc}"
echo

echo "##### verify offload arch flags actually used by the compiler #####"
grep -RhoE -- "--offload-arch=[a-z0-9:+-]+" "${build_dir}" 2>/dev/null | sort | uniq -c
echo "-- HIP_FLAGS / CMAKE_HIP_FLAGS in flags.make (first file) --"
grep -E "HIP_FLAGS|CXX_FLAGS" "${build_dir}/hipans/ans/CMakeFiles/gpu_ans.dir/flags.make"
echo "-- code objects in libgpu_ans.a (offload bundle targets) --"
if [ -f "${build_dir}/lib/libgpu_ans.a" ]; then
  strings "${build_dir}/lib/libgpu_ans.a" | grep -oE "hip-amdgcn-amd-amdhsa--gfx[0-9a-z:+-]+" | sort | uniq -c
fi
echo

echo "##### build products #####"
echo "-- bin --"; ls -l "${build_dir}/bin" 2>/dev/null
echo "-- lib --"; ls -l "${build_dir}/lib" 2>/dev/null
echo "-- all targets known to the generator --"
"${cmake_bin}" --build "${build_dir}" --target help | grep -vE "^\.\.\.|^The following|^$" | grep -vE "\.(o|i|s)$" | sort
echo
echo "##### done (build rc=${build_rc}, full_mode=${full_mode}) #####"
exit ${build_rc}
