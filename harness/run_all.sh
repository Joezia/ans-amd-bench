#!/usr/bin/env bash
set -euo pipefail

harness_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
bench_dir="$(cd -- "${harness_dir}/.." && pwd)"
binary="${harness_dir}/bin/hipans_raw_decode_bench"
results_dir="${RESULTS_DIR:-${harness_dir}/results}"
partition="${GPU_PARTITION:-}"
gpu_count="${GPU_COUNT:-1}"
gpu_request_mode="${GPU_REQUEST_MODE:-gres}"
gpu_index="${GPU_INDEX:-0}"
run_time="${RUN_TIME:-12:00:00}"
seed="${SEED:-20260920}"
lambda="${LAMBDA:-100.0}"
warmup="${WARMUP_ITERS:-10}"
measure="${MEASURE_ITERS:-50}"
file_input="${FILE_INPUT:-${bench_dir}/inputs/expo20_u8_64MiB+1234.bin}"

if [[ "${1:-}" != "--inside-allocation" ]]; then
  mkdir -p "${results_dir}"
  if [[ -z "${partition}" ]]; then
    gpu_partitions="${GPU_PARTITIONS:?set GPU_PARTITION or GPU_PARTITIONS}"
    partition="$("${bench_dir}/scripts/select_slurm_partition.sh" "${gpu_partitions}" "${gpu_count}")"
  fi
  case "${gpu_request_mode}" in
    gres) gpu_request=("--gres=gpu:${gpu_count}") ;;
    gpus) gpu_request=("--gpus=${gpu_count}") ;;
    none) gpu_request=() ;;
    *) echo "ERROR: GPU_REQUEST_MODE must be gres, gpus, or none" >&2; exit 2 ;;
  esac
  echo "selected GPU partition: ${partition}"
  exec srun \
    --job-name=hipans-raw-decode-matrix \
    --partition="${partition}" \
    "${gpu_request[@]}" \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=2 \
    --time="${run_time}" \
    env GPU_PARTITION="${partition}" HIP_VISIBLE_DEVICES="${gpu_index}" \
    "${BASH_SOURCE[0]}" --inside-allocation
fi

partition="${GPU_PARTITION:?GPU_PARTITION missing inside allocation}"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "ERROR: refusing to run outside a Slurm allocation" >&2
  exit 2
fi
if [[ "$(hostname)" == hacchead* ]]; then
  echo "ERROR: refusing to run on hacchead" >&2
  exit 2
fi
if [[ ! -x "${binary}" ]]; then
  echo "ERROR: missing harness binary; run ${harness_dir}/build.sh first" >&2
  exit 2
fi
if [[ ! -r "${file_input}" ]]; then
  echo "ERROR: FILE_INPUT is not readable: ${file_input}" >&2
  exit 2
fi
if (( warmup < 10 || measure < 50 )); then
  echo "ERROR: WARMUP_ITERS must be >=10 and MEASURE_ITERS must be >=50" >&2
  exit 2
fi

mkdir -p "${results_dir}"
csv="${CSV_PATH:-${results_dir}/raw-decode-${SLURM_JOB_ID}.csv}"
stdout_log="${results_dir}/raw-decode-${SLURM_JOB_ID}.out"
stderr_log="${results_dir}/raw-decode-${SLURM_JOB_ID}.err"
exec > >(tee "${stdout_log}") 2> >(tee "${stderr_log}" >&2)

echo "============================================================"
echo "hipANS raw decode full matrix"
echo "hostname: $(hostname)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"
echo "partition: ${SLURM_JOB_PARTITION:-${partition}}"
echo "HIP_VISIBLE_DEVICES: ${HIP_VISIBLE_DEVICES:-unset}"
echo "physical GPU index: ${gpu_index}"
echo "seed: ${seed}"
echo "lambda: ${lambda}"
echo "warmup iterations: ${warmup}"
echo "measurement iterations: ${measure}"
echo "DF input: ${file_input}"
echo "CSV: ${csv}"
echo "stdout log: ${stdout_log}"
echo "stderr log: ${stderr_log}"
echo "matrix: 3 datasets x 6 sizes x 3 probBits x 2 batch counts = 108 cases"
echo "============================================================"

datasets=(d1 d3 df)
size_labels=(S1 S2 S3_r1 S3_r4096 S3_r8191 S4)
size_values=(
  8192
  $((256 * 8192))
  $((256 * 8192 + 1))
  $((256 * 8192 + 4096))
  $((256 * 8192 + 8191))
  $((64 * 1024 * 1024))
)
probbits=(9 10 11)
batch_counts=(1 8)

failures=0
case_number=0
total_cases=$(( ${#datasets[@]} * ${#size_values[@]} * ${#probbits[@]} * ${#batch_counts[@]} ))

for dataset in "${datasets[@]}"; do
  for size_index in "${!size_values[@]}"; do
    label="${size_labels[$size_index]}"
    size="${size_values[$size_index]}"
    for probbits_value in "${probbits[@]}"; do
      for batch in "${batch_counts[@]}"; do
        case_number=$((case_number + 1))
        echo
        echo "######## CASE ${case_number}/${total_cases}: dataset=${dataset} scale=${label} size=${size} probbits=${probbits_value} batch=${batch} ########"
        command=(
          "${binary}"
          --dataset "${dataset}"
          --size-bytes "${size}"
          --probbits "${probbits_value}"
          --num-in-batch "${batch}"
          --seed "${seed}"
          --lambda "${lambda}"
          --warmup "${warmup}"
          --measure "${measure}"
          --gpu-index "${gpu_index}"
          --csv "${csv}"
          --case-label "${label}"
        )
        if [[ "${dataset}" == df ]]; then
          command+=(--file "${file_input}")
        fi
        if (( case_number == 1 )); then
          command+=(--verify-checksum-contract)
        fi
        printf 'COMMAND: '
        printf '%q ' "${command[@]}"
        printf '\n'
        if "${command[@]}"; then
          echo "CASE RESULT: PASS"
        else
          rc=$?
          echo "CASE RESULT: FAIL rc=${rc}"
          failures=$((failures + 1))
        fi
      done
    done
  done
done

echo
echo "============================================================"
echo "matrix complete: total=${total_cases} failures=${failures}"
echo "CSV: ${csv}"
echo "stdout log: ${stdout_log}"
echo "stderr log: ${stderr_log}"
echo "============================================================"

if (( failures != 0 )); then
  exit 1
fi
