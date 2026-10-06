#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
bench_dir="$(cd -- "${script_dir}/.." && pwd)"
input_dir="${PAPER_INPUT_DIR:-${bench_dir}/inputs/paper}"
partition="${GPU_PARTITION:-mi3001x}"

# Missing licensed model inputs can be reported without occupying a GPU.  Once
# at least one case is present, allocate the MI300X before invoking the bench.
if [[ -z "${SLURM_JOB_ID:-}" ]] && compgen -G "${input_dir}/*.bin" >/dev/null; then
  exec srun --job-name=paper-repro --partition="${partition}" --gpus=1 \
    --nodes=1 --ntasks=1 --cpus-per-task=2 --time="${RUN_TIME:-02:00:00}" \
    "${BASH_SOURCE[0]}"
fi
if [[ ! -x "${bench_dir}/.venv/bin/python" ]]; then
  python3 -m venv --system-site-packages "${bench_dir}/.venv"
fi

peak_args=()
if [[ -r "${bench_dir}/results/peak_bw.json" ]]; then
  peak="$("${bench_dir}/.venv/bin/python" -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["selected_peak_GBps"])' \
    "${bench_dir}/results/peak_bw.json")"
  peak_args=(--peak-gbps "${peak}")
fi

"${bench_dir}/.venv/bin/python" "${script_dir}/reproduce_paper.py" \
  --bench "${bench_dir}/ans_bench" --input-dir "${input_dir}" \
  --config "${bench_dir}/configs/paper_models.json" \
  --output "${bench_dir}/results/paper_reproduction.md" \
  --json-output "${bench_dir}/results/paper_reproduction.json" \
  --warmup "${WARMUP_ITERS:-10}" --iters "${MEASURE_ITERS:-50}" \
  "${peak_args[@]}"
echo "paper reproduction table: ${bench_dir}/results/paper_reproduction.md"
