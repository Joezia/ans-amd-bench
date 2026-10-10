#!/usr/bin/env bash
set -euo pipefail
bench_dir="${ANS_BENCH_ROOT:-${SLURM_SUBMIT_DIR:-}}"
if [[ -z "${bench_dir}" || ! -f "${bench_dir}/bench/lds_bench.cpp" ]]; then
  echo "Submit with sbatch from ans-amd-bench or set ANS_BENCH_ROOT" >&2
  exit 2
fi
cd "${bench_dir}"
manifest="${LDS_MANIFEST:-${bench_dir}/inputs/lds/dsv4-flash-base-expert10-w1/artifact.json}"
output="${bench_dir}/results/profiling/lds-${SLURM_JOB_ID}"
mkdir -p "${output}"
# Independent correctness cases exercise padding, constant frequency=4096,
# all-symbol tables, lane boundaries and multiple tiles on the actual device.
"${bench_dir}/.venv/bin/python" "${bench_dir}/tests/run_lds_gpu_cases.py" "${output}/gpu-cases"
for table in lds global; do
  "${bench_dir}/.venv/bin/python" scripts/run_lds_bench.py --manifest "${manifest}" \
    --mode decode --table "${table}" --warmup 10 --iters 50 >"${output}/decode-${table}.json"
done
for mode in raw lds-control; do
  "${bench_dir}/.venv/bin/python" scripts/run_lds_bench.py --manifest "${manifest}" \
    --mode "${mode}" --warmup 10 --iters 50 >"${output}/${mode}.json"
done
"${bench_dir}/.venv/bin/python" scripts/audit_lds_isa.py \
  "${bench_dir}/build-lds/lds_bench-hip-amdgcn-amd-amdhsa-gfx942.s" >"${output}/isa.json"
bash profiling/run_lds_baseline.sh "${manifest}" "${output}/profile"
echo "LDS baseline complete: ${output}"
