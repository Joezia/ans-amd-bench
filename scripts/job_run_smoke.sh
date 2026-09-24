#!/usr/bin/env bash
# Phase four: run the existing hipANS tests / benchmarks once on the GPU node and
# print their output verbatim. Does not modify any file in the hipANS source tree.
set -uo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
bench_dir="$(cd -- "${script_dir}/.." && pwd)"
build_dir="${BUILD_DIR:-${bench_dir}/build-hipans-full}"
in_dir="${INPUT_DIR:-${bench_dir}/inputs}"
work_dir="${WORK_DIR:-${bench_dir}/results/run-${SLURM_JOB_ID:-manual}}"
bin="${build_dir}/bin"
partition="${GPU_PARTITION:-mi210_u280_u55c}"
run_time="${RUN_TIME:-01:00:00}"
gpu_index="${GPU_INDEX:-0}"

if [[ "${1:-}" != "--inside-allocation" && -z "${SLURM_JOB_ID:-}" ]]; then
  exec srun \
    --job-name=hipans-smoke \
    --partition="${partition}" \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=2 \
    --time="${run_time}" \
    env HIP_VISIBLE_DEVICES="${gpu_index}" \
    "${BASH_SOURCE[0]}" --inside-allocation
fi

if [[ -z "${SLURM_JOB_ID:-}" || "$(hostname)" == hacchead* ]]; then
  echo "ERROR: refusing to run GPU tests outside a Slurm allocation" >&2
  exit 2
fi

mkdir -p "${work_dir}"

echo "##### run job #####"
echo "node: $(hostname)  job: ${SLURM_JOB_ID:-unset}  date: $(date -Iseconds)"
echo "build dir: ${build_dir}"
echo "work dir:  ${work_dir}"
echo "HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-unset} ROCR_VISIBLE_DEVICES=${ROCR_VISIBLE_DEVICES:-unset}"
echo "binaries:"; ls -l "${bin}"
echo

section() { echo; echo "================================================================"; echo "$*"; echo "================================================================"; }

run_timed() {
  # run_timed <label> <cmd...>   : prints the command, runs it, prints rc + wall time
  local label="$1"; shift
  echo "\$ $*"
  local t0=$(date +%s.%N)
  "$@"
  local rc=$?
  local t1=$(date +%s.%N)
  echo "[${label}] rc=${rc} wall=$(echo "${t1} - ${t0}" | bc) s"
  return ${rc}
}

roundtrip_check() {
  # roundtrip_check <input> <output>
  if cmp -s "$1" "$2"; then
    echo "[cmp] BIT-EXACT: $(basename "$1") == $(basename "$2") ($(stat -c%s "$1") bytes)"
  else
    echo "[cmp] MISMATCH: $(basename "$1") vs $(basename "$2") (sizes $(stat -c%s "$1") / $(stat -c%s "$2" 2>/dev/null))"
    cmp "$1" "$2" | head -3
  fi
}

# ---------------------------------------------------------------- gtest suites
for t in ans_test ans_statistics_test batch_prefix_sum_test float_test; do
  section "gtest: ${t}"
  if [ -x "${bin}/${t}" ]; then
    run_timed "${t}" "${bin}/${t}"
  else
    echo "(not built)"
  fi
done

# ---------------------------------------------------------------- compress (raw ANS, file in/out)
section "compress (raw byte-wise ANS, file round trip)"
if [ -x "${bin}/compress" ]; then
  for f in uniform_u8_64MiB.bin "expo20_u8_64MiB+1234.bin" normal_f32_64MiB.bin; do
    echo "--- input: ${f} ---"
    run_timed "compress:${f}" "${bin}/compress" "${in_dir}/${f}" "${work_dir}/${f}.ans" "${work_dir}/${f}.out"
    echo "compressed file size: $(stat -c%s "${work_dir}/${f}.ans" 2>/dev/null) bytes"
    roundtrip_check "${in_dir}/${f}" "${work_dir}/${f}.out"
    echo "--- second run (same input) ---"
    run_timed "compress2:${f}" "${bin}/compress" "${in_dir}/${f}" "${work_dir}/${f}.ans" "${work_dir}/${f}.out"
    roundtrip_check "${in_dir}/${f}" "${work_dir}/${f}.out"
  done
else
  echo "(not built)"
fi

# ---------------------------------------------------------------- float codec file benchmarks
section "fpcompress (float32 codec, file round trip)"
if [ -x "${bin}/fpcompress" ]; then
  f=normal_f32_64MiB.bin
  run_timed "fpcompress" "${bin}/fpcompress" "${in_dir}/${f}" "${work_dir}/${f}.fp.ans" "${work_dir}/${f}.fp.out"
  echo "compressed file size: $(stat -c%s "${work_dir}/${f}.fp.ans" 2>/dev/null) bytes"
  roundtrip_check "${in_dir}/${f}" "${work_dir}/${f}.fp.out"
else
  echo "(not built)"
fi

section "fp16compress (float16 codec, file round trip)"
if [ -x "${bin}/fp16compress" ]; then
  f=normal_f16_64MiB.bin
  run_timed "fp16compress" "${bin}/fp16compress" "${in_dir}/${f}" "${work_dir}/${f}.fp.ans" "${work_dir}/${f}.fp.out"
  echo "compressed file size: $(stat -c%s "${work_dir}/${f}.fp.ans" 2>/dev/null) bytes"
  roundtrip_check "${in_dir}/${f}" "${work_dir}/${f}.fp.out"
else
  echo "(not built)"
fi

section "fpB16compress (bfloat16 codec, file round trip)"
if [ -x "${bin}/fpB16compress" ]; then
  f=normal_bf16_64MiB.bin
  run_timed "fpB16compress" "${bin}/fpB16compress" "${in_dir}/${f}" "${work_dir}/${f}.fp.ans" "${work_dir}/${f}.fp.out"
  echo "compressed file size: $(stat -c%s "${work_dir}/${f}.fp.ans" 2>/dev/null) bytes"
  roundtrip_check "${in_dir}/${f}" "${work_dir}/${f}.fp.out"
else
  echo "(not built)"
fi

# ---------------------------------------------------------------- torch benchmark
section "benchmark (libtorch, synthetic randn, no args)"
if [ -x "${bin}/benchmark" ]; then
  run_timed "benchmark" "${bin}/benchmark"
  section "benchmark <normal_f32_64MiB.bin>"
  run_timed "benchmark:file" "${bin}/benchmark" "${in_dir}/normal_f32_64MiB.bin"
else
  echo "(not built)"
fi

# ---------------------------------------------------------------- python benchmark
section "benchmark.py (needs build/lib/libhipans.so relative to hipans/ dir)"
if [ -f "${build_dir}/lib/libhipans.so" ]; then
  echo "(libhipans.so exists at ${build_dir}/lib/libhipans.so, but benchmark.py hard-codes ../build/lib/libhipans.so; not run in this job)"
else
  echo "(libhipans.so not built)"
fi

echo
echo "##### work dir contents #####"
ls -l "${work_dir}"
echo "##### done #####"
