#!/usr/bin/env bash
set -euo pipefail

harness_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
bench_dir="$(cd -- "${harness_dir}/.." && pwd)"
binary="${harness_dir}/bin/hipans_raw_decode_bench"
results_dir="${RESULTS_DIR:-${harness_dir}/results}"
partition="${GPU_PARTITION:-}"
gpu_count="${GPU_COUNT:-1}"
gpu_request_mode="${GPU_REQUEST_MODE:-gres}"
run_time="${RUN_TIME:-00:30:00}"
gpu_index="${GPU_INDEX:-0}"
warmup="${WARMUP_ITERS:-10}"
measure="${MEASURE_ITERS:-50}"
seed="${SEED:-20260920}"
file_input="${FILE_INPUT:-${bench_dir}/inputs/expo20_u8_64MiB+1234.bin}"
mode="${RUN_MODE:-search}"

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
    --job-name="hipans-raw-decode-${mode}" \
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
if [[ "${SLURM_JOB_PARTITION:-}" != "${partition}" ]]; then
  echo "ERROR: expected partition ${partition}, got ${SLURM_JOB_PARTITION:-unset}" >&2
  exit 2
fi
if [[ ! -x "${binary}" ]]; then
  echo "ERROR: missing benchmark binary: ${binary}" >&2
  exit 2
fi
if [[ ! -r "${file_input}" ]]; then
  echo "ERROR: missing DF input: ${file_input}" >&2
  exit 2
fi
if (( warmup < 10 || measure < 50 )); then
  echo "ERROR: warmup must be >=10 and measure must be >=50" >&2
  exit 2
fi

mkdir -p "${results_dir}"
csv="${CSV_PATH:-${results_dir}/raw-decode-${mode}-${SLURM_JOB_ID}.csv}"
stdout_log="${results_dir}/raw-decode-${mode}-${SLURM_JOB_ID}.out"
stderr_log="${results_dir}/raw-decode-${mode}-${SLURM_JOB_ID}.err"
exec > >(tee "${stdout_log}") 2> >(tee "${stderr_log}" >&2)

echo "hostname: $(hostname)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"
echo "partition: ${SLURM_JOB_PARTITION}"
echo "SLURM_STEP_GPUS: ${SLURM_STEP_GPUS:-unset}"
echo "HIP_VISIBLE_DEVICES: ${HIP_VISIBLE_DEVICES:-unset}"
echo "ROCR_VISIBLE_DEVICES: ${ROCR_VISIBLE_DEVICES:-unset}"
echo "mode: ${mode}"
echo "DF input: ${file_input}"
echo "CSV: ${csv}"

run_case() {
  local dataset="$1"
  local size_bytes="$2"
  local probbits="$3"
  local batch="$4"
  local label="$5"
  shift 5
  local command=(
    "${binary}"
    --dataset "${dataset}"
    --size-bytes "${size_bytes}"
    --probbits "${probbits}"
    --num-in-batch "${batch}"
    --seed "${seed}"
    --warmup "${warmup}"
    --measure "${measure}"
    --gpu-index "${gpu_index}"
    --csv "${csv}"
    --case-label "${label}"
  )
  if [[ "${dataset}" == "df" ]]; then
    command+=(--file "${file_input}")
  fi
  command+=("$@")
  printf 'COMMAND: '
  printf '%q ' "${command[@]}"
  printf '\n'
  if ! "${command[@]}"; then
    echo "ERROR: benchmark failed for ${label}" >&2
    exit 1
  fi

  local correctness
  local run_status
  correctness="$(tail -n 1 "${csv}" | awk -F, '{print $16}')"
  run_status="$(tail -n 1 "${csv}" | awk -F, '{print $17}')"
  if [[ "${correctness}" != "PASS" || "${run_status}" != "OK" ]]; then
    echo "ERROR: rejected ${label}: correctness=${correctness} run_status=${run_status}" >&2
    exit 1
  fi
}

if [[ "${mode}" == "search" ]]; then
  run_case d1 8192 10 1 smoke --verify-checksum-contract
  run_case df $((64 * 1024 * 1024)) 9 8 size-64MiB-p9
  run_case df $((128 * 1024 * 1024)) 9 8 size-128MiB-p9
  run_case df $((256 * 1024 * 1024)) 9 8 size-256MiB-p9
  run_case df $((256 * 1024 * 1024)) 10 8 size-256MiB-p10
  run_case df $((256 * 1024 * 1024)) 11 8 size-256MiB-p11
elif [[ "${mode}" == "repeat" ]]; then
  peak_size="${PEAK_SIZE_BYTES:?PEAK_SIZE_BYTES is required for repeat mode}"
  peak_probbits="${PEAK_PROBBITS:?PEAK_PROBBITS is required for repeat mode}"
  run_case df "${peak_size}" "${peak_probbits}" 8 peak-repeat-2
  run_case df "${peak_size}" "${peak_probbits}" 8 peak-repeat-3
else
  echo "ERROR: RUN_MODE must be search or repeat" >&2
  exit 2
fi

echo "all cases passed: mode=${mode} csv=${csv}"
