#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
bench_dir="$(cd -- "${script_dir}/.." && pwd)"
model="${MODEL_FILE:-${bench_dir}/models/qwen2.5-1.5b/model.safetensors}"
input_dir="${INPUT_DIR:-${bench_dir}/inputs}"

mkdir -p "${input_dir}"

# 先看看有什么
python3 "${script_dir}/dump_weights.py" list "${model}" | head -20

# ① 单个代表性张量（分布最纯，无拼接痕迹）
python3 "${script_dir}/dump_weights.py" dump "${model}" "${input_dir}/qwen_bf16_single_mlp.bin" \
    'layers\.0\.mlp\.gate_proj\.weight' --dtype BF16

# ② 中等规模，跨多层（≈64 MiB，覆盖 S4 下界）
python3 "${script_dir}/dump_weights.py" dump "${model}" "${input_dir}/qwen_bf16_64MiB.bin" \
    'mlp\.(gate|up|down)_proj\.weight' --dtype BF16 --limit 67108864

# ③ 大规模（256 MiB，压 grid-stride 多轮循环）
python3 "${script_dir}/dump_weights.py" dump "${model}" "${input_dir}/qwen_bf16_256MiB.bin" \
    'mlp\.(gate|up|down)_proj\.weight' --dtype BF16 --limit 268435456

# 记录指纹
(cd "${input_dir}" && md5sum qwen_bf16_*.bin > MD5SUMS)
