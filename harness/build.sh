#!/usr/bin/env bash
set -euo pipefail

harness_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
bench_dir="$(cd -- "${harness_dir}/.." && pwd)"
repo_dir="${HIPANS_REPO:-${bench_dir}/../hipANS}"
hipans_build_dir="${HIPANS_BUILD_DIR:-${bench_dir}/build-hipans-full}"
library_dir="${hipans_build_dir}/lib"
glog_generated="${hipans_build_dir}/third_party/glog"
partition="${BUILD_PARTITION:-}"
build_time="${BUILD_TIME:-00:15:00}"
gpu_arch="${GPU_ARCH:?GPU_ARCH must be set explicitly}"

if [[ "${1:-}" != "--inside-allocation" ]]; then
  if [[ -z "${partition}" ]]; then
    build_partitions="${BUILD_PARTITIONS:?set BUILD_PARTITION or BUILD_PARTITIONS}"
    partition="$("${bench_dir}/scripts/select_slurm_partition.sh" "${build_partitions}" 0)"
  fi
  echo "selected build partition: ${partition}"
  exec srun \
    --job-name=hipans-harness-build \
    --partition="${partition}" \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=2 \
    --time="${build_time}" \
    env BUILD_PARTITION="${partition}" \
    "${BASH_SOURCE[0]}" --inside-allocation
fi

partition="${BUILD_PARTITION:?BUILD_PARTITION missing inside allocation}"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "ERROR: refusing to compile outside a Slurm allocation" >&2
  exit 2
fi
if [[ "$(hostname)" == hacchead* ]]; then
  echo "ERROR: refusing to compile on hacchead" >&2
  exit 2
fi

hipcc_bin="${HIPCC_BIN:-$(command -v hipcc || true)}"
output_dir="${harness_dir}/bin"
output_bin="${output_dir}/hipans_raw_decode_bench"
mkdir -p "${output_dir}"

required=(
  "${hipcc_bin}"
  "${library_dir}/libgpu_ans.a"
  "${library_dir}/libhipans_utils.a"
  "${library_dir}/libglog.a"
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
  --offload-arch="${gpu_arch}"
  "-DHIPANS_GPU_ARCH=\"${gpu_arch}\""
  -DGFLAGS_IS_A_DLL=0
  "-DHIPANS_SOURCE_DIR=\"${repo_dir}\""
  "-DBENCH_SOURCE_DIR=\"${bench_dir}\""
  -I"${repo_dir}"
  -I"${glog_generated}"
  -I"${repo_dir}/third_party/glog/src"
  "${harness_dir}/hipans_raw_decode_bench.cpp"
  -x none
  "${library_dir}/libgpu_ans.a"
  "${library_dir}/libhipans_utils.a"
  "${library_dir}/libglog.a"
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
