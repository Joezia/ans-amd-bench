#!/usr/bin/env bash
# Environment probe on the GPU compute node. Read-only; touches nothing in the repo.
set -uo pipefail

partition="${GPU_PARTITION:-mi210_u280_u55c}"
run_time="${RUN_TIME:-00:10:00}"

if [[ "${1:-}" != "--inside-allocation" && -z "${SLURM_JOB_ID:-}" ]]; then
  exec srun \
    --job-name=hipans-env-probe \
    --partition="${partition}" \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=1 \
    --time="${run_time}" \
    "${BASH_SOURCE[0]}" --inside-allocation
fi

if [[ -z "${SLURM_JOB_ID:-}" || "$(hostname)" == hacchead* ]]; then
  echo "ERROR: refusing to probe GPUs outside a Slurm compute-node allocation" >&2
  exit 2
fi

echo "##### node / slurm #####"
echo "hostname: $(hostname)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-unset}"
echo "date: $(date -Iseconds)"
echo

echo "##### rocminfo | grep -E 'gfx|Marketing Name' #####"
rocminfo 2>&1 | grep -E "gfx|Marketing Name"
echo

echo "##### hipcc --version #####"
command -v hipcc; hipcc --version 2>&1
echo

echo "##### cat /opt/rocm/.info/version #####"
cat /opt/rocm/.info/version 2>/dev/null || echo "(missing)"
echo "/opt/rocm -> $(readlink -f /opt/rocm)"
ls -d /opt/rocm* 2>/dev/null
echo

echo "##### rocm-smi #####"
rocm-smi 2>&1
echo

echo "##### rocm-smi --showproductname --showmeminfo vram #####"
rocm-smi --showproductname --showmeminfo vram 2>&1
echo

echo "##### toolchain #####"
command -v cmake && cmake --version | head -1
ls -l /opt/rocm/llvm/bin/clang++ 2>&1
/opt/rocm/llvm/bin/clang++ --version 2>&1 | head -2
echo

echo "##### torch / libtorch availability (needed by 'hipans' lib + 'benchmark' target) #####"
command -v python python3
python3 -c "import torch,sys; print('torch', torch.__version__, 'hip', torch.version.hip); print(torch.utils.cmake_prefix_path)" 2>&1
python -c "import torch; print(torch.utils.cmake_prefix_path)" 2>&1
echo "-- searching for TorchConfig.cmake (bounded) --"
for d in /opt /usr/local /usr/lib /usr/lib64 "$HOME"; do
  [ -d "$d" ] && find "$d" -maxdepth 6 -name TorchConfig.cmake 2>/dev/null
done
ls -d /opt/conda /opt/miniconda* /opt/anaconda* "$HOME"/miniconda* "$HOME"/anaconda* 2>/dev/null
echo

echo "##### modules (if any) #####"
command -v module >/dev/null 2>&1 && module avail 2>&1 | head -40 || echo "(no 'module' command)"
echo "##### done #####"
