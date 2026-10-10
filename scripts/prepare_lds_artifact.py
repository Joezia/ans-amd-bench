#!/usr/bin/env python3
"""Offline artifact preparation, separate from the GPU measurement process."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True, type=Path)
    p.add_argument("--packer", type=Path, default=ROOT / "bin/lds_pack")
    a = p.parse_args()
    source = json.loads(a.source.read_text())
    base = a.source.parent
    if source["source"] != "native" or source["encoding"] not in ("F8_E4M3", "F8_E5M2"):
        raise ValueError("this delivery accepts native FP8 only, no 4-bit or simulated conversion")
    for tensor in source["tensors"].values():
        path = base / tensor["file"]
        if path.stat().st_size != tensor["bytes"] or sha(path) != tensor["sha256"]:
            raise ValueError("source tensor integrity check failed")
    artifact = base / "weights.ans"
    manifest = base / "artifact.json"
    if artifact.exists() or manifest.exists():
        raise ValueError("artifact already exists; refusing overwrite")
    weights = base / source["tensors"][source["tensor_name"]]["file"]
    stats = json.loads(subprocess.check_output(
        [str(a.packer), str(weights), *map(str, source["shape"]), str(artifact)], text=True))
    scale_bytes = sum(t["bytes"] for name, t in source["tensors"].items() if name != source["tensor_name"])
    result = {**stats, "source": source, "artifact_file": artifact.name,
              "artifact_sha256": sha(artifact), "source_manifest_sha256": sha(a.source),
              "scale_bytes": scale_bytes, "scales_compressed": False,
              "ratio_total_with_scales": (stats["raw_bytes"] + scale_bytes) / (stats["artifact_bytes"] + scale_bytes),
              "layout": "row-major 64x128 tiles, row-major tile traversal, zero edge padding",
              "stream_mapping": "64 lanes; lane l writes tile indices l + 64*i, i=0..127",
              "codec": "local byte-rANS v1; 32-bit state; Lower=2^23; 12-bit probabilities",
              "codebook_scope": "one matrix; padding included; not an entire transformer layer",
              "producer_git_commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
              "implementation_sha256": {name: sha(ROOT / "bench" / name) for name in ["lds_pack.cpp", "lds_format.h"]}}
    manifest.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
