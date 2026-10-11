#!/usr/bin/env python3
"""Bounded safetensors range extraction; never writes an upstream HF cache."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import requests


def fetch_range(url, start, end, token=None):
    headers = {"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    # Distinct URLs prevent intermediary caches reusing a different partial response.
    with requests.get(url + f"?download=true&range={start}-{end}", headers=headers,
                      timeout=(30, 120), stream=True) as response:
        response.raise_for_status()
        expected = end - start + 1
        if response.status_code != 206 or not response.headers.get("Content-Range", "").startswith(f"bytes {start}-{end}/"):
            raise RuntimeError("server did not honor bounded byte range; refusing full-shard download")
        chunks, length = [], 0
        for chunk in response.iter_content(1 << 20):
            length += len(chunk)
            if length > expected:
                raise RuntimeError("range response exceeded requested bytes")
            chunks.append(chunk)
        if length != expected:
            raise RuntimeError("truncated range response")
        return b"".join(chunks)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", default="deepseek-ai/DeepSeek-V4-Flash-Base")
    p.add_argument("--revision", default="8855555deef230a27a21a8d6f294b7b7497759b6")
    p.add_argument("--tensor", default="layers.0.ffn.experts.10.w1.weight")
    p.add_argument("--scale-tensor", help="companion scale tensor (default: <tensor stem>.scale)")
    p.add_argument("--model-id", default="dsv4-flash-base")
    p.add_argument("--index", required=True, type=Path, help="read-only cached safetensors index")
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--token-file", type=Path, help="optional credential; contents are never logged")
    p.add_argument("--max-tensor-bytes", type=int, default=64 << 20)
    a = p.parse_args()
    token = os.environ.get("HF_TOKEN")
    if a.token_file:
        token = a.token_file.read_text().strip()
    index = json.loads(a.index.read_text())
    shard = index["weight_map"][a.tensor]
    url = f"https://huggingface.co/{a.repo}/resolve/{a.revision}/{shard}"
    length = struct.unpack("<Q", fetch_range(url, 0, 7, token))[0]
    if length > 16 << 20:
        raise ValueError("safetensors header exceeds bounded 16 MiB allowance")
    header_raw = fetch_range(url, 8, 7 + length, token)
    header = json.loads(header_raw)
    weight = header[a.tensor]
    if weight["dtype"] not in ("F8_E4M3", "F8_E5M2") or len(weight["shape"]) != 2:
        raise ValueError(f"expected native FP8 matrix, got {weight}")
    scale_name = a.scale_tensor or (a.tensor.removesuffix(".weight") + ".scale")
    names = [a.tensor, scale_name]
    if any(name not in header or name not in index["weight_map"] for name in names):
        raise ValueError(f"weight and companion scale must both be present in the selected shard: {names}")
    a.output.mkdir(parents=True, exist_ok=True)
    if any((a.output / name).exists() for name in ("source.json", "weights.bin", "scales.bin")):
        raise ValueError("output already has a source manifest; choose a new directory")
    extracted = {}
    for name, filename in zip(names, ["weights.bin", "scales.bin"]):
        entry = header[name]
        lo, hi = entry["data_offsets"]
        if not 0 < hi - lo <= a.max_tensor_bytes:
            raise ValueError("target tensor exceeds bounded download allowance")
        if index["weight_map"][name] != shard:
            raise ValueError("companion scale in another shard is not supported")
        content = fetch_range(url, 8 + length + lo, 7 + length + hi, token)
        target = a.output / filename
        with target.open("xb") as f:
            f.write(content)
        extracted[name] = {**entry, "file": filename, "bytes": len(content),
                           "sha256": hashlib.sha256(content).hexdigest()}
    manifest = {"schema": "lds-weight-source-v1", "model_id": a.model_id,
                "hf_repo": a.repo, "revision": a.revision, "tensor_name": a.tensor,
                "shard": shard, "source": "native", "conversion": "none",
                "encoding": weight["dtype"], "shape": weight["shape"],
                "scale_consumed_by_baseline": False,
                "checkpoint_header_sha256": hashlib.sha256(header_raw).hexdigest(),
                "index_sha256": hashlib.sha256(a.index.read_bytes()).hexdigest(),
                "tensors": extracted}
    (a.output / "source.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest))


if __name__ == "__main__":
    main()
