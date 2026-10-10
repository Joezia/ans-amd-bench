#!/usr/bin/env bash
set -euo pipefail
bench_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  exec srun --partition="${BUILD_PARTITION:-devel}" --nodes=1 --ntasks=1 \
    --cpus-per-task=2 --time=00:10:00 bash "${BASH_SOURCE[0]}"
fi
mkdir -p "${bench_dir}/bin" "${bench_dir}/build-lds"
"${CXX:-g++}" -std=c++17 -O3 -Wall -Wextra "${bench_dir}/bench/lds_pack.cpp" -o "${bench_dir}/bin/lds_pack"
arch="${GPU_ARCH:-gfx942}"
cd "${bench_dir}/build-lds"
hipcc -std=c++17 -O3 --offload-arch="${arch}" -Wall -Wextra \
  "-DLDS_BUILD_FLAGS=\"-std=c++17 -O3 --offload-arch=${arch}\"" \
  --save-temps "${bench_dir}/bench/lds_bench.cpp" -o "${bench_dir}/bin/lds_bench"
# hipcc temporary ISA is saved in build-lds; build products stay local.
echo "Built bin/lds_pack and bin/lds_bench"
