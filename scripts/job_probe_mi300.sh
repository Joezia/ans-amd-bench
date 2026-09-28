#!/usr/bin/env bash
set -euo pipefail

echo "hostname: $(hostname)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-unset}"
echo "SLURM_JOB_PARTITION: ${SLURM_JOB_PARTITION:-unset}"
echo "SLURM_JOB_GPUS: ${SLURM_JOB_GPUS:-unset}"
echo "SLURM_STEP_GPUS: ${SLURM_STEP_GPUS:-unset}"
echo "HIP_VISIBLE_DEVICES: ${HIP_VISIBLE_DEVICES:-unset}"
echo "ROCR_VISIBLE_DEVICES: ${ROCR_VISIBLE_DEVICES:-unset}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"

echo "===== tools ====="
command -v hipcc || true
hipcc --version || true
command -v cmake || true
cmake --version || true
if [[ -r /opt/rocm/.info/version ]]; then
  echo -n "ROCm version file: "
  cat /opt/rocm/.info/version
fi

echo "===== hipconfig ====="
hipconfig --full || true

echo "===== rocminfo ====="
rocminfo

echo "===== rocm-smi ====="
rocm-smi
