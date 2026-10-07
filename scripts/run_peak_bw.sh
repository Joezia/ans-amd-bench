#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
bench_dir="$(cd -- "${script_dir}/.." && pwd)"
source_dir="${bench_dir}/build-babelstream/source"
build_dir="${bench_dir}/build-babelstream/build"
commit="f6ae48de899408cf50c24079417dc71a03dbb5a8"
partition="${GPU_PARTITION:-mi3001x}"
gpu_arch="${GPU_ARCH:-gfx942}"
array_elements="${BABELSTREAM_ARRAY_ELEMENTS:-134217728}"
iterations="${BABELSTREAM_ITERS:-20}"
output="${PEAK_BW_OUTPUT:-${bench_dir}/results/peak_bw.json}"
raw_csv="${bench_dir}/results/peak_bw.csv"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  exec srun --job-name=babelstream-peak --partition="${partition}" --gpus=1 \
    --nodes=1 --ntasks=1 --cpus-per-task=2 --time="${RUN_TIME:-00:30:00}" \
    "${BASH_SOURCE[0]}"
fi

if [[ ! -d "${source_dir}/.git" ]]; then
  mkdir -p "$(dirname -- "${source_dir}")"
  git clone https://github.com/UoB-HPC/BabelStream.git "${source_dir}"
fi
git -C "${source_dir}" checkout --detach "${commit}"

hipcc_bin="${HIPCC_BIN:-$(command -v hipcc)}"
cmake -S "${source_dir}" -B "${build_dir}" -DMODEL=hip \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER="${hipcc_bin}" \
  "-DCXX_EXTRA_FLAGS=--offload-arch=${gpu_arch}"
cmake --build "${build_dir}" --parallel 2

command=("${build_dir}/hip-stream" --arraysize "${array_elements}"
         --numtimes "${iterations}" --csv)
mkdir -p "$(dirname -- "${output}")"
"${command[@]}" >"${raw_csv}"

if [[ ! -x "${bench_dir}/.venv/bin/python" ]]; then
  python3 -m venv --system-site-packages "${bench_dir}/.venv"
fi
# Consume complete producer output under `set -o pipefail`; early awk exits can
# otherwise make rocminfo report SIGPIPE (141) after a successful measurement.
gpu_name="$(rocminfo | awk -F: '!found && /Marketing Name:.*(AMD Instinct|Radeon)/{gsub(/^[[:space:]]+|[[:space:]]+$/, "", $2); print $2; found=1}')"
gpu_arch_runtime="$(rocminfo | awk -F: '!found && /^[[:space:]]*Name:[[:space:]]*gfx/{gsub(/^[[:space:]]+|[[:space:]]+$/, "", $2); print $2; found=1}')"
rocm_version="$(rocprofv3 --version 2>&1 | awk -F: '!found && /rocm_version/{gsub(/[[:space:]]/, "", $2); print $2; found=1}')"
command_text="$(printf '%q ' "${command[@]}")"
"${bench_dir}/.venv/bin/python" "${script_dir}/parse_babelstream.py" \
  --csv "${raw_csv}" --output "${output}" --commit "${commit}" \
  --command "${command_text}" --gpu "${gpu_name:-unknown}" \
  --gpu-arch "${gpu_arch_runtime:-unknown}" --rocm "${rocm_version:-unknown}" \
  --hostname "$(hostname)"
echo "peak bandwidth result: ${output}"
