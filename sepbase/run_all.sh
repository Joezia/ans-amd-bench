#!/usr/bin/env bash
set -euo pipefail

sepbase_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
bench_dir="$(cd -- "${sepbase_dir}/.." && pwd)"
binary="${sepbase_dir}/bin/sepbase_bench"
results_dir="${RESULTS_DIR:-${sepbase_dir}/results}"
partition="${GPU_PARTITION:-mi210_u280_u55c}"
gpu_index="${GPU_INDEX:-0}"
run_time="${RUN_TIME:-02:00:00}"
weight_file="${WEIGHT_FILE:-${bench_dir}/inputs/qwen_bf16_256MiB.bin}"
seed="${ACTIVATION_SEED:-20260921}"
warmup="${WARMUP_ITERS:-10}"
measure="${MEASURE_ITERS:-50}"
cpu_rel_tolerance="${CPU_REL_TOLERANCE:-0.02}"

if [[ "${1:-}" != "--inside-allocation" ]]; then
  mkdir -p "${results_dir}"
  exec srun \
    --job-name=hipans-sepbase \
    --partition="${partition}" \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=2 \
    --time="${run_time}" \
    env HIP_VISIBLE_DEVICES="${gpu_index}" \
    "${BASH_SOURCE[0]}" --inside-allocation
fi

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "ERROR: refusing to run outside a Slurm allocation" >&2
  exit 2
fi
if [[ "$(hostname)" == hacchead* ]]; then
  echo "ERROR: refusing to run on hacchead" >&2
  exit 2
fi
if [[ ! -x "${binary}" ]]; then
  echo "ERROR: missing binary; run ${sepbase_dir}/build.sh first" >&2
  exit 2
fi
if [[ ! -r "${weight_file}" ]]; then
  echo "ERROR: weight input is not readable: ${weight_file}" >&2
  exit 2
fi
if (( warmup < 10 || measure < 50 )); then
  echo "ERROR: WARMUP_ITERS must be >=10 and MEASURE_ITERS must be >=50" >&2
  exit 2
fi

mkdir -p "${results_dir}"
csv="${CSV_PATH:-${results_dir}/sepbase-${SLURM_JOB_ID}.csv}"
stdout_log="${results_dir}/sepbase-${SLURM_JOB_ID}.out"
stderr_log="${results_dir}/sepbase-${SLURM_JOB_ID}.err"
exec > >(tee "${stdout_log}") 2> >(tee "${stderr_log}" >&2)

echo "============================================================"
echo "hipANS decode + vendor bf16 GEMM separated baseline"
echo "hostname: $(hostname)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"
echo "partition: ${SLURM_JOB_PARTITION:-${partition}}"
echo "HIP_VISIBLE_DEVICES: ${HIP_VISIBLE_DEVICES:-unset}"
echo "physical GPU index: ${gpu_index}"
echo "weight file: ${weight_file}"
echo "activation seed: ${seed}"
echo "warmup: ${warmup}"
echo "measure: ${measure}"
echo "CPU sampled relative tolerance: ${cpu_rel_tolerance}"
echo "shapes: (64,1024,1024), (1024,1024,1024), (4096,1024,1024)"
echo "cache modes per shape: cold and hot"
echo "CSV: ${csv}"
echo "stdout log: ${stdout_log}"
echo "stderr log: ${stderr_log}"
echo "============================================================"

shapes=(
  "64 1024 1024"
  "1024 1024 1024"
  "4096 1024 1024"
)

failures=0
case_number=0
for shape in "${shapes[@]}"; do
  read -r m n k <<< "${shape}"
  case_number=$((case_number + 1))
  echo
  echo "######## SHAPE ${case_number}/${#shapes[@]}: M=${m} N=${n} K=${k} ########"
  command=(
    "${binary}"
    --m "${m}"
    --n "${n}"
    --k "${k}"
    --weight-file "${weight_file}"
    --seed "${seed}"
    --warmup "${warmup}"
    --measure "${measure}"
    --gpu-index "${gpu_index}"
    --cpu-rel-tolerance "${cpu_rel_tolerance}"
    --csv "${csv}"
  )
  printf 'COMMAND: '
  printf '%q ' "${command[@]}"
  printf '\n'
  if "${command[@]}"; then
    echo "SHAPE RESULT: PASS"
  else
    rc=$?
    echo "SHAPE RESULT: FAIL rc=${rc}"
    failures=$((failures + 1))
  fi
done

echo
echo "============================================================"
echo "all shapes complete: total=${#shapes[@]} failures=${failures}"
echo "CSV: ${csv}"
echo "stdout log: ${stdout_log}"
echo "stderr log: ${stderr_log}"
echo "============================================================"

if (( failures != 0 )); then
  exit 1
fi
