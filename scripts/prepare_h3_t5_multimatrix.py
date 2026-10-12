#!/usr/bin/env python3
"""Fetch and ANS-encode nine distinct GLM FP8 MLP matrices for H3 T5."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import requests

ROOT = Path(__file__).resolve().parents[1]
REPO = "zai-org/GLM-5.2-FP8"
REVISION = "f33c6dc501ee5a2c7e35155653b1b1abbc320951"
LAYERS = (0, 1, 2)
PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
REFERENCE_GATE_DIR = ROOT / "inputs/lds/glm52-layer0-gate-proj"
MAX_TENSOR_BYTES = 96 << 20


def sha_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch_index(path, token):
    url = f"https://huggingface.co/{REPO}/resolve/{REVISION}/model.safetensors.index.json"
    headers = {"Accept-Encoding": "identity"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    chunks, size = [], 0
    with requests.get(url, headers=headers, timeout=(30, 120), stream=True) as response:
        response.raise_for_status()
        for chunk in response.iter_content(1 << 20):
            size += len(chunk)
            if size > 64 << 20:
                raise RuntimeError("safetensors index exceeds 64 MiB safety limit")
            chunks.append(chunk)
    content = b"".join(chunks)
    index = json.loads(content)
    if not isinstance(index.get("weight_map"), dict):
        raise ValueError("downloaded safetensors index has no weight_map")
    path.write_bytes(content)
    return sha_file(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--job-id", required=True)
    args = parser.parse_args()
    if args.output_root.exists():
        raise FileExistsError(f"refusing to overwrite existing T5 workspace: {args.output_root}")
    source_root = args.output_root / "source"
    members_root = source_root / "members"
    artifact_dir = args.output_root / "artifact"
    source_root.mkdir(parents=True)
    members_root.mkdir()
    artifact_dir.mkdir()

    index_path = args.output_root / "model.safetensors.index.json"
    index_sha = fetch_index(index_path, os.environ.get("HF_TOKEN"))
    index = json.loads(index_path.read_text())
    member_entries = []
    for layer in LAYERS:
        for projection in PROJECTIONS:
            tensor_name = f"model.layers.{layer}.mlp.{projection}.weight"
            scale_name = f"model.layers.{layer}.mlp.{projection}.weight_scale_inv"
            directory = members_root / f"layer{layer}-{projection}"
            if layer == 0 and projection == "gate_proj":
                source_manifest_path = REFERENCE_GATE_DIR / "source.json"
                source_manifest = json.loads(source_manifest_path.read_text())
                for record in source_manifest["tensors"].values():
                    source_path = REFERENCE_GATE_DIR / record["file"]
                    if source_path.stat().st_size != record["bytes"] or sha_file(source_path) != record["sha256"]:
                        raise ValueError(f"existing GLM gate source failed integrity check: {source_path}")
            else:
                directory.mkdir()
                command = [sys.executable, str(ROOT / "scripts/fetch_lds_weights.py"),
                           "--repo", REPO, "--revision", REVISION, "--model-id", "glm52-fp8",
                           "--tensor", tensor_name, "--scale-tensor", scale_name,
                           "--index", str(index_path), "--output", str(directory),
                           "--max-tensor-bytes", str(MAX_TENSOR_BYTES)]
                subprocess.run(command, check=True)
                source_manifest_path = directory / "source.json"
                source_manifest = json.loads(source_manifest_path.read_text())
            if tensor_name not in index["weight_map"]:
                raise ValueError(f"tensor is missing from index: {tensor_name}")
            if index["weight_map"][tensor_name] != source_manifest["shard"]:
                raise ValueError(f"index shard disagrees with fetched metadata for {tensor_name}")
            if source_manifest["tensor_name"] != tensor_name or source_manifest["encoding"] != "F8_E4M3":
                raise ValueError(f"unexpected source tensor/encoding for {tensor_name}")
            weight = source_manifest["tensors"][tensor_name]
            scale = source_manifest["tensors"][scale_name]
            if weight["bytes"] != 75497472 or weight["bytes"] % 8192:
                raise ValueError(f"expected tile-aligned 72 MiB native FP8 weight: {tensor_name}")
            if scale["dtype"] != "F32":
                raise ValueError(f"expected F32 companion scale for {tensor_name}")
            member_entries.append({
                "layer": layer,
                "projection": projection,
                "tensor_name": tensor_name,
                "scale_tensor_name": scale_name,
                "logical_shape": weight["shape"],
                "dtype": weight["dtype"],
                "source_shard": source_manifest["shard"],
                "weight_bytes": weight["bytes"],
                "weight_sha256": weight["sha256"],
                "scale_shape": scale["shape"],
                "scale_bytes": scale["bytes"],
                "scale_sha256": scale["sha256"],
                "source_manifest_sha256": sha_file(source_manifest_path),
                "source_manifest_path": str(source_manifest_path),
                "source_directory": str(source_manifest_path.parent),
                "weight_file": weight["file"],
                "scale_file": scale["file"]
            })

    aggregate_weight = source_root / "weights.bin"
    aggregate_scales = source_root / "scales.bin"
    weight_hash = hashlib.sha256()
    scale_hash = hashlib.sha256()
    with aggregate_weight.open("xb") as weight_out, aggregate_scales.open("xb") as scale_out:
        for member in member_entries:
            base = Path(member["source_directory"])
            for filename, expected_size, expected_hash, destination, digest in (
                    (member["weight_file"], member["weight_bytes"], member["weight_sha256"], weight_out, weight_hash),
                    (member["scale_file"], member["scale_bytes"], member["scale_sha256"], scale_out, scale_hash)):
                source_path = base / filename
                if source_path.stat().st_size != expected_size or sha_file(source_path) != expected_hash:
                    raise ValueError(f"member integrity changed before assembly: {source_path}")
                with source_path.open("rb") as source_handle:
                    for block in iter(lambda: source_handle.read(1 << 20), b""):
                        destination.write(block)
                        digest.update(block)
    total_weight_bytes = aggregate_weight.stat().st_size
    if total_weight_bytes != 9 * 75497472 or total_weight_bytes % (6144 * 64):
        raise ValueError("aggregate must contain nine complete 72 MiB tensors and full 64x128 tiles")
    scale_bytes = aggregate_scales.stat().st_size
    aggregate_name = "glm52_layers0_2_gate_up_down_concatenated"
    aggregate_tensor = {"dtype": "F8_E4M3", "shape": [total_weight_bytes // 6144, 6144],
                        "data_offsets": [0, total_weight_bytes], "file": "weights.bin",
                        "bytes": total_weight_bytes, "sha256": weight_hash.hexdigest()}
    scale_tensor = {"dtype": "F32-concatenated", "shape": [scale_bytes // 4],
                    "data_offsets": [0, scale_bytes], "file": "scales.bin",
                    "bytes": scale_bytes, "sha256": scale_hash.hexdigest()}
    source_manifest = {
        "schema": "lds-weight-source-v1",
        "model_id": "glm52-fp8-first3-dense-mlp-composite",
        "hf_repo": REPO,
        "revision": REVISION,
        "tensor_name": aggregate_name,
        "shard": "multiple; see members",
        "source": "native",
        "conversion": "none; concatenate original FP8 tensor byte streams without transposition or requantization",
        "encoding": "F8_E4M3",
        "shape": aggregate_tensor["shape"],
        "scale_consumed_by_baseline": False,
        "checkpoint_header_sha256": hashlib.sha256("".join(m["source_manifest_sha256"] for m in member_entries).encode()).hexdigest(),
        "index_sha256": index_sha,
        "tensors": {aggregate_name: aggregate_tensor, "concatenated_scales": scale_tensor},
        "members": member_entries,
        "composition": "nine distinct layer-0..2 gate/up/down matrices concatenated in projection-major order within each layer; each 72 MiB boundary is exactly tile-aligned; viewed as bytes with pseudo-shape [110592,6144] for LDS throughput only",
        "codebook_scope": "one shared current-hipANS p10 table for the concatenated measurement workload"
    }
    source_path = source_root / "source.json"
    source_path.write_text(json.dumps(source_manifest, indent=2) + "\n")

    prepare_command = [sys.executable, str(ROOT / "scripts/prepare_lds_artifact.py"),
                       "--source", str(source_path), "--output-dir", str(artifact_dir)]
    prepared = json.loads(subprocess.check_output(prepare_command, text=True))
    if prepared.get("verified_actual_LDS") is not True:
        raise RuntimeError("T5 composite artifact failed actual-LDS byte-exact validation")
    artifact_manifest_path = artifact_dir / "artifact.json"
    artifact_manifest = json.loads(artifact_manifest_path.read_text())
    result = {
        "schema": "h3-t5-multimatrix-artifact-v1",
        "status": "prepared-and-verified",
        "slurm_job": args.job_id,
        "model": "zai-org/GLM-5.2-FP8",
        "revision": REVISION,
        "members": member_entries,
        "member_count": len(member_entries),
        "unique_weight_bytes": total_weight_bytes,
        "scale_bytes": scale_bytes,
        "composite_shape_bytes_only": aggregate_tensor["shape"],
        "composition": source_manifest["composition"],
        "encoding": "native F8_E4M3 bytes; no conversion",
        "artifact_bytes": artifact_manifest["artifact_bytes"],
        "artifact_metadata_bytes": artifact_manifest["artifact_metadata_bytes"],
        "ratio_codes": artifact_manifest["ratio_codes"],
        "ratio_total_with_scales": artifact_manifest["ratio_total_with_scales"],
        "artifact_sha256": artifact_manifest["artifact_sha256"],
        "source_manifest_sha256": artifact_manifest["source_manifest_sha256"],
        "artifact_manifest": str(artifact_manifest_path),
        "raw_weights": str(aggregate_weight),
        "raw_output_dir": str(args.output_root)
    }
    result_path = ROOT / "results" / f"h3_t5_artifact_{args.job_id}.json"
    report_path = ROOT / "docs" / f"H3_T5_ARTIFACT_{args.job_id}.md"
    if result_path.exists() or report_path.exists():
        raise FileExistsError("refusing to overwrite existing T5 report/result")
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    report = [f"# H3 T5 multi-matrix artifact — Slurm {args.job_id}", "",
              "Prepared and byte-exactly verified one current-hipANS archive from nine distinct native GLM-5.2-FP8 MLP matrices: layers 0–2 × gate/up/down. The original per-tensor byte streams were concatenated without conversion; every 72 MiB tensor boundary is tile-aligned. The resulting `[110592, 6144]` shape is a byte-layout for the decoder harness, not a claim that these matrices form one logical model tensor.", "",
              f"Unique weight bytes: {total_weight_bytes:,} ({total_weight_bytes / (1<<20):.1f} MiB); scale bytes retained separately and not consumed by the decode kernel: {scale_bytes:,}. ANS artifact: {artifact_manifest['artifact_bytes']:,} bytes; metadata: {artifact_manifest['artifact_metadata_bytes']:,} bytes; code-only ratio: {artifact_manifest['ratio_codes']:.6f}.", "",
              f"The nine real matrices use one shared current-hipANS p10 table in this composite artifact. This is a large-working-set decoder measurement, not a separate per-matrix codec-format result. Artifact SHA-256: `{artifact_manifest['artifact_sha256']}`.", "",
              "| Layer | Projection | Logical shape | Weight bytes | Scale bytes | Source shard |", "|---:|---|---:|---:|---:|---|"]
    for member in member_entries:
        report.append(f"| {member['layer']} | {member['projection']} | {member['logical_shape']} | {member['weight_bytes']:,} | {member['scale_bytes']:,} | `{member['source_shard']}` |")
    report.extend(["", f"Artifact manifest: `{artifact_manifest_path}`. Curated provenance: [`{result_path.name}`](../results/{result_path.name}).", ""])
    report_path.write_text("\n".join(report))
    print(json.dumps({"artifact_manifest": str(artifact_manifest_path),
                      "result": str(result_path), "report": str(report_path),
                      "artifact_sha256": artifact_manifest["artifact_sha256"],
                      "verified": True}))


if __name__ == "__main__":
    main()
