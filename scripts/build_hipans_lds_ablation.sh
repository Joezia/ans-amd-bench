#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "ERROR: ablation compilation must run inside a Slurm job" >&2
  exit 2
fi
hipans="$(cd "${HIPANS_ROOT:-${root}/../hipANS}" && pwd)"
machine="${LDS_MACHINE:-mi300x}"
case "${machine}" in
  mi300x) arch=gfx942; build="${root}/build-hipans-mi300" ;;
  mi350x) arch=gfx950; build="${root}/build-hipans-mi350" ;;
  *) echo "Unsupported LDS_MACHINE=${machine}" >&2; exit 2 ;;
esac
adapter_build="${root}/build-hipans-lds-ablation-${machine}"
binary="${root}/bin/hipans_lds_ablation_${machine}"
commit="$(git -C "${hipans}" rev-parse HEAD)"
if [[ -n "$(git -C "${hipans}" status --porcelain)" ]]; then
  echo "ERROR: expected read-only, clean hipANS source tree" >&2
  exit 2
fi
if [[ ! -f "${build}/lib/libgpu_ans.a" ]]; then
  echo "Missing out-of-tree hipANS libraries: ${build}" >&2
  exit 2
fi
mkdir -p "${adapter_build}" "${root}/bin"
rocm_root="$(hipconfig --rocmpath)"
if [[ -r "${rocm_root}/.info/version" ]]; then
  rocm_version="$(tr -d '\n' <"${rocm_root}/.info/version")"
else
  rocm_version=unknown
fi
flags="-std=c++17 -O3 -DNDEBUG --offload-arch=${arch}"
cd "${adapter_build}"
hipcc -std=c++17 -O3 -DNDEBUG --offload-arch="${arch}" \
  "-DHIPANS_COMMIT_BUILT=\"${commit}\"" \
  "-DHIPANS_GPU_ARCH_BUILT=\"${arch}\"" \
  "-DHIPANS_ROCM_VERSION=\"${rocm_version}\"" \
  "-DHIPANS_ABLATION_FLAGS=\"${flags}\"" \
  -DGFLAGS_IS_A_DLL=0 -I"${hipans}" -I"${build}/third_party/glog" \
  -I"${hipans}/third_party/glog/src" "${root}/bench/hipans_lds_ablation.cpp" -x none \
  "${build}/lib/libgpu_ans.a" "${build}/lib/libhipans_utils.a" "${build}/lib/libglog.a" \
  -pthread -o "${binary}"
echo "Built bench-only ablation ${binary} for ${arch}, hipANS ${commit}"
