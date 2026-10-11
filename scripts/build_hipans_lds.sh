#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  exec srun -p "${BUILD_PARTITION:-devel}" -N1 -n1 -c2 -t 00:15:00 bash "${BASH_SOURCE[0]}"
fi
hipans="$(cd "${HIPANS_ROOT:-${root}/../hipANS}" && pwd)"
build="${HIPANS_BUILD_DIR:-${root}/build-hipans-mi300}"
arch="${GPU_ARCH:-gfx942}"
case "${arch}" in gfx942|gfx950) ;; *) echo "Unsupported baseline architecture: ${arch}" >&2; exit 2;; esac
adapter_build="${HIPANS_LDS_BUILD_DIR:-${root}/build-hipans-lds}"
binary="${HIPANS_LDS_BINARY:-${root}/bin/hipans_lds_bench}"
commit="$(git -C "${hipans}" rev-parse HEAD)"
if [[ -n "$(git -C "${hipans}" status --porcelain)" ]]; then
  echo "ERROR: baseline requires the clean current unoptimized hipANS checkout" >&2
  exit 2
fi
mkdir -p "$(dirname "${binary}")" "${adapter_build}"
cd "${adapter_build}"
hipcc -std=c++17 -O3 -DNDEBUG --offload-arch="${arch}" --save-temps \
  "-DHIPANS_ROOT_BUILT=\"${hipans}\"" "-DHIPANS_COMMIT_BUILT=\"${commit}\"" \
  "-DHIPANS_GPU_ARCH_BUILT=\"${arch}\"" \
  "-DHIPANS_LDS_BUILD_FLAGS=\"-std=c++17 -O3 -DNDEBUG --offload-arch=${arch}\"" \
  -DGFLAGS_IS_A_DLL=0 -I"${hipans}" -I"${build}/third_party/glog" \
  -I"${hipans}/third_party/glog/src" "${root}/bench/hipans_lds_bench.cpp" -x none \
  "${build}/lib/libgpu_ans.a" "${build}/lib/libhipans_utils.a" "${build}/lib/libglog.a" \
  -pthread -o "${binary}"
echo "Built actual hipANS ${commit} LDS adapter"
