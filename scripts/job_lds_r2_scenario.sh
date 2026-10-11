#!/usr/bin/env bash
set -euo pipefail

bench_dir="${ANS_BENCH_ROOT:-${SLURM_SUBMIT_DIR:?Submit from ans-amd-bench}}"
cd "${bench_dir}"
source_file="${LDS_SOURCE:-${bench_dir}/inputs/lds/glm52-layer0-gate-proj/source.json}"
artifact_dir="${LDS_ARTIFACT_DIR:-${bench_dir}/inputs/lds/glm52-layer0-gate-proj/mi350x}"
output="${bench_dir}/results/profiling/lds-r2-glm-${LDS_MACHINE:?Set LDS_MACHINE}-${SLURM_JOB_ID}"
mkdir -p "${output}"

case "${LDS_MACHINE}" in
  mi350x)
    export GPU_ARCH=gfx950 BUILD_PARTITION=mi3501x
    export HIPANS_BUILD_DIR="${bench_dir}/build-hipans-mi350"
    export HIPANS_LDS_BUILD_DIR="${bench_dir}/build-hipans-lds-mi350"
    export HIPANS_LDS_BINARY="${bench_dir}/bin/hipans_lds_bench_mi350"
    ;;
  mi300x)
    export GPU_ARCH=gfx942 BUILD_PARTITION=mi3001x
    export HIPANS_BUILD_DIR="${bench_dir}/build-hipans-mi300"
    export HIPANS_LDS_BUILD_DIR="${bench_dir}/build-hipans-lds-mi300"
    export HIPANS_LDS_BINARY="${bench_dir}/bin/hipans_lds_bench"
    ;;
  *) echo "Unsupported LDS_MACHINE=${LDS_MACHINE}" >&2; exit 2 ;;
esac
export HIPANS_ROOT="${HIPANS_ROOT:-${bench_dir}/../hipANS}"
export HIPANS_LDS_ISA="${HIPANS_LDS_BUILD_DIR}/hipans_lds_bench-hip-amdgcn-amd-amdhsa-${GPU_ARCH}.s"

manifest="${artifact_dir}/artifact.json"
echo "R2 GLM scenario baseline: job=${SLURM_JOB_ID} machine=${LDS_MACHINE} node=$(hostname) start=$(date -Iseconds)"
rocminfo >"${output}/rocminfo.txt"
if [[ -f "${HIPANS_BUILD_DIR}/lib/libgpu_ans.a" ]]; then
  cmake --build "${HIPANS_BUILD_DIR}" --parallel 4 --target glog hipans_utils gpu_ans \
    >"${output}/native-build.log" 2>&1
else
  HIPANS_REPO="${HIPANS_ROOT}" BUILD_DIR="${HIPANS_BUILD_DIR}" BUILD_CPUS=4 \
    bash scripts/job_build_full.sh --inside-allocation >"${output}/native-build.log" 2>&1
fi
bash scripts/build_hipans_lds.sh >"${output}/adapter-build.log" 2>&1

python="${bench_dir}/.venv/bin/python"
"${python}" scripts/run_lds_bench.py --manifest "${manifest}" --mode decode \
  --replicas 1 --threads 128 --grid-mult 4 --streams 1 --warmup 10 --iters 50 \
  >"${output}/scenario-primary.json"
"${python}" scripts/run_lds_bench.py --manifest "${manifest}" --mode raw \
  --replicas 1 --threads 128 --grid-mult 4 --streams 1 --warmup 10 --iters 50 \
  >"${output}/raw-control.json"

for replicas in 4 16 64; do
  "${python}" scripts/run_lds_bench.py --manifest "${manifest}" --mode decode \
    --replicas "${replicas}" --threads 128 --grid-mult 4 --streams 1 --warmup 5 --iters 20 \
    >"${output}/replica-aux-${replicas}.json"
done

"${python}" - "${output}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
def load(name):
    result = json.loads((root / name).read_text())
    if result.get("verified_actual_LDS") is not True:
        raise SystemExit(f"actual LDS verification failed for {name}")
    return result

summary = {
    "schema": "lds-real-scenario-r2-v1",
    "scenario": {
        "model": "zai-org/GLM-5.2-FP8",
        "revision": "f33c6dc501ee5a2c7e35155653b1b1abbc320951",
        "tensor": "model.layers.0.mlp.gate_proj.weight",
        "shape": [12288, 6144],
        "encoding": "F8_E4M3",
        "inference_context": "batch=1, prefill prompt=4096 tokens; M=4096, K=6144, N=12288",
        "measurement_scope": "decode-only LDS proxy; no activation GEMM or fused consumer",
    },
    "primary_scenario": load("scenario-primary.json"),
    "raw_to_lds_control": load("raw-control.json"),
    "replica_auxiliary": [load(f"replica-aux-{count}.json") for count in (4, 16, 64)],
    "replica_note": "replicas are repeated copies of the same tensor for resource/cache diagnostics; they are not model parameters or batch",
}
(root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(json.dumps({"summary": str(root / "summary.json"), "verified": True}))
PY
echo "R2 scenario baseline complete: ${output}"
