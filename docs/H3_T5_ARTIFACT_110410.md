# H3 T5 multi-matrix artifact — Slurm 110410

Prepared and byte-exactly verified one current-hipANS archive from nine distinct native GLM-5.2-FP8 MLP matrices: layers 0–2 × gate/up/down. The original per-tensor byte streams were concatenated without conversion; every 72 MiB tensor boundary is tile-aligned. The resulting `[110592, 6144]` shape is a byte-layout for the decoder harness, not a claim that these matrices form one logical model tensor.

Unique weight bytes: 679,477,248 (648.0 MiB); scale bytes retained separately and not consumed by the decode kernel: 165,888. ANS artifact: 588,470,288 bytes; metadata: 21,897,760 bytes; code-only ratio: 1.154650.

The nine real matrices use one shared current-hipANS p10 table in this composite artifact. This is a large-working-set decoder measurement, not a separate per-matrix codec-format result. Artifact SHA-256: `2ac416ab458d9b3f69c5a69d2799dc80c2359cd8ed6f242a1637a0e6a23a3f1a`.

| Layer | Projection | Logical shape | Weight bytes | Scale bytes | Source shard |
|---:|---|---:|---:|---:|---|
| 0 | gate_proj | [12288, 6144] | 75,497,472 | 18,432 | `model-00001-of-00141.safetensors` |
| 0 | up_proj | [12288, 6144] | 75,497,472 | 18,432 | `model-00001-of-00141.safetensors` |
| 0 | down_proj | [6144, 12288] | 75,497,472 | 18,432 | `model-00001-of-00141.safetensors` |
| 1 | gate_proj | [12288, 6144] | 75,497,472 | 18,432 | `model-00001-of-00141.safetensors` |
| 1 | up_proj | [12288, 6144] | 75,497,472 | 18,432 | `model-00001-of-00141.safetensors` |
| 1 | down_proj | [6144, 12288] | 75,497,472 | 18,432 | `model-00001-of-00141.safetensors` |
| 2 | gate_proj | [12288, 6144] | 75,497,472 | 18,432 | `model-00020-of-00141.safetensors` |
| 2 | up_proj | [12288, 6144] | 75,497,472 | 18,432 | `model-00020-of-00141.safetensors` |
| 2 | down_proj | [6144, 12288] | 75,497,472 | 18,432 | `model-00020-of-00141.safetensors` |

Artifact manifest: `/work1/bingshenghe/xbr/ziang/compression/ans-amd-bench/inputs/lds/glm52-first3-dense-stack/job-110410/artifact/artifact.json`. Curated provenance: [`h3_t5_artifact_110410.json`](../results/h3_t5_artifact_110410.json).
