#!/usr/bin/env bash
#SBATCH --job-name=h3-t1
#SBATCH --partition=devel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --output=results/slurm-h3-t1-%j.out
#SBATCH --error=results/slurm-h3-t1-%j.err
set -euo pipefail

bench_dir="${ANS_BENCH_ROOT:-${SLURM_SUBMIT_DIR:?Submit from ans-amd-bench}}"
cd "${bench_dir}"
python="${bench_dir}/.venv/bin/python"
if [[ ! -x "${python}" ]]; then
  echo "Expected local venv Python at ${python}" >&2
  exit 2
fi
echo "H3 T1 generation: job=${SLURM_JOB_ID} node=$(hostname) start=$(date -Iseconds)"
"${python}" scripts/generate_h3_bandwidth_definitions.py
echo "H3 T1 generation complete: $(date -Iseconds)"
