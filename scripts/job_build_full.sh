#!/usr/bin/env bash
# Configure + build hipANS on a ROCm compute node, in a build dir OUTSIDE the repo.
# Does not modify any file inside the hipANS source tree.
#
# Attempt 1: HIPANS_BUILD_FULL_LIBRARY=ON  (float codec + torch lib + benchmark)
# Attempt 2 (fallback if attempt 1 fails to configure): HIPANS_BUILD_FULL_LIBRARY=OFF
set -uo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
bench_dir="$(cd -- "${script_dir}/.." && pwd)"
repo_dir="${HIPANS_REPO:-${bench_dir}/../hipANS}"
build_dir="${BUILD_DIR:-${bench_dir}/build-hipans-full}"
build_cpus="${BUILD_CPUS:-8}"
gpu_arch="${GPU_ARCH:-gfx90a}"
partition="${BUILD_PARTITION:-mi210_u280_u55c}"
build_time="${BUILD_TIME:-00:30:00}"
cmake_bin="${CMAKE_BIN:-/usr/local/bin/cmake}"
hipcc_bin="${HIPCC_BIN:-/usr/bin/hipcc}"
hip_compiler_bin="${HIP_COMPILER_BIN:-/opt/rocm/llvm/bin/clang++}"

if [[ "${1:-}" != "--inside-allocation" && -z "${SLURM_JOB_ID:-}" ]]; then
  exec srun \
    --job-name=hipans-full-build \
    --partition="${partition}" \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task="${build_cpus}" \
    --time="${build_time}" \
    "${BASH_SOURCE[0]}" --inside-allocation
fi

if [[ -z "${SLURM_JOB_ID:-}" || "$(hostname)" == hacchead* ]]; then
  echo "ERROR: refusing to build outside a Slurm compute-node allocation" >&2
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
    -DBUILD_SHARED_LIBS=OFF \
    -DHIPANS_LIBRARY_TYPE=STATIC \
    -DHIPANS_BUILD_FULL_LIBRARY="${full}"
  local rc=$?
  set +x
  echo "configure rc=${rc}"
  return ${rc}
}

full_mode=ON
if ! configure ON; then
  echo
  echo "!!!!! configure with FULL_LIBRARY=ON failed; falling back to OFF !!!!!"
  echo
  full_mode=OFF
  configure OFF || { echo "configure failed in both modes"; exit 1; }
fi
echo "FULL_LIBRARY mode actually used: ${full_mode}"
echo

echo "##### cache: key variables #####"
grep -E "^(CMAKE_HIP_ARCHITECTURES|CMAKE_CXX_COMPILER|CMAKE_HIP_COMPILER|CMAKE_BUILD_TYPE|BUILD_SHARED_LIBS|HIPANS_LIBRARY_TYPE|HIPANS_BUILD_FULL_LIBRARY|HIP_DIR|HIP_PATH|Torch_DIR|TORCH_LIBRARY)" "${build_dir}/CMakeCache.txt"
echo

echo "##### cmake --build (all targets, keep going on errors) #####"
set -x
"${cmake_bin}" --build "${build_dir}" --parallel "${build_cpus}" -- -k
build_rc=$?
set +x
echo "build rc=${build_rc}"
echo

echo "##### verify offload arch flags actually used by the compiler #####"
grep -rhoE -- "--offload-arch=[a-z0-9:+-]+" "${build_dir}"/hipans/*/CMakeFiles/*.dir/flags.make "${build_dir}"/hipans/CMakeFiles/*.dir/flags.make 2>/dev/null | sort | uniq -c
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
