#!/usr/bin/env bash
set -euo pipefail
profiling_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
bench_dir="$(cd -- "${profiling_dir}/.." && pwd)"
python_bin="${bench_dir}/.venv/bin/python"
bench_cli="${ANS_BENCH_CLI:-${bench_dir}/ans_bench}"

if (( $# != 1 )); then
  echo "Usage: $0 seed-input.bin" >&2
  exit 2
fi
if [[ ! -x "${python_bin}" ]]; then
  echo "ERROR: create the local venv first: python3 -m venv --system-site-packages .venv" >&2
  exit 2
fi
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  partition="${GPU_PARTITION:-mi3001x}"
  exec srun --job-name=ans-sweeps --partition="${partition}" --gpus=1 \
    --nodes=1 --ntasks=1 --cpus-per-task=8 --time="${RUN_TIME:-02:00:00}" \
    "${BASH_SOURCE[0]}" "$(realpath "$1")"
fi

output="${SWEEP_OUTPUT_DIR:-${bench_dir}/results/profiling/sweeps-${SLURM_JOB_ID}-$(date -u +%Y%m%dT%H%M%SZ)}"
cases="${output}/cases"
mkdir -p "${output}/single" "${output}/concurrency"
"${python_bin}" "${profiling_dir}/make_sweep_cases.py" --input "$1" \
  --output-dir "${cases}" --sizes-mib "${SWEEP_SIZES_MIB:-64,192,384}" \
  --tile-sizes "${SWEEP_TILE_SIZES:-8192,65536,1048576}"

tail -n +2 "${cases}/cases.tsv" | while IFS=$'\t' read -r name input meta _bytes _tile; do
  command=("${bench_cli}" --input "${input}" --meta "${meta}"
           --warmup "${SWEEP_WARMUP:-5}" --iters "${SWEEP_ITERS:-20}")
  if [[ -n "${PEAK_GBPS:-}" ]]; then
    command+=(--peak-GBps "${PEAK_GBPS}")
  fi
  "${command[@]}" >"${output}/single/${name}.json"
done

# Each process contributes one HIP stream/queue.  This deployment sweep avoids
# codec changes and records wall time plus each independently verified result.
base_input="${cases}/input-64MiB.bin"
base_meta="${cases}/size-64MiB_tile-65536.json"
for concurrency in ${SWEEP_CONCURRENCY:-1 2 4 8}; do
  run_dir="${output}/concurrency/streams-${concurrency}"
  mkdir -p "${run_dir}"
  start_ns="$(date +%s%N)"
  pids=()
  for (( stream=0; stream<concurrency; ++stream )); do
    command=("${bench_cli}" --input "${base_input}" --meta "${base_meta}"
             --warmup "${SWEEP_WARMUP:-5}" --iters "${SWEEP_ITERS:-20}")
    if [[ -n "${PEAK_GBPS:-}" ]]; then
      command+=(--peak-GBps "${PEAK_GBPS}")
    fi
    HIPANS_STACK_BYTES="${HIPANS_STACK_BYTES:-536870912}" \
      "${command[@]}" \
      >"${run_dir}/stream-${stream}.json" 2>"${run_dir}/stream-${stream}.err" &
    pids+=("$!")
  done
  failures=0
  for pid in "${pids[@]}"; do
    wait "${pid}" || failures=$((failures + 1))
  done
  end_ns="$(date +%s%N)"
  printf '{"concurrent_hip_streams":%d,"wall_ns":%d,"failures":%d}\n' \
    "${concurrency}" "$((end_ns - start_ns))" "${failures}" >"${run_dir}/deployment.json"
done

cat >"${output}/launch-parameter-availability.tsv" <<'EOF'
parameter	baseline	variants	status
grid oversubscription	4	2,4,8	requires out-of-tree diagnostic library variants; see build_launch_variants.sh
workgroup threads / rANS streams	128	64,128	requires out-of-tree diagnostic library variants; see build_launch_variants.sh
rANS block bytes	8192	fixed	not swept: changing it changes the codec/archive algorithm
EOF
echo "sweep output: ${output}"
