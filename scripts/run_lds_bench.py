#!/usr/bin/env python3
"""Integrity-checked real-artifact LDS benchmark; stdout is one JSON object."""
import argparse
import json
from pathlib import Path
import subprocess
from prepare_lds_artifact import sha, ROOT


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--mode", choices=["decode", "raw", "lds-control"], default="decode")
    p.add_argument("--table", choices=["lds", "global"], default="lds")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--skip-verify", action="store_true", help="profiler only, after a verified run")
    a = p.parse_args()
    manifest = json.loads(a.manifest.read_text())
    base, source = a.manifest.parent, manifest["source"]
    artifact = base / manifest["artifact_file"]
    if sha(artifact) != manifest["artifact_sha256"]:
        raise ValueError("artifact hash mismatch")
    for tensor in source["tensors"].values():
        if sha(base / tensor["file"]) != tensor["sha256"]:
            raise ValueError("reference/scale hash mismatch")
    reference = base / source["tensors"][source["tensor_name"]]["file"]
    command = [str(ROOT / "bin/lds_bench"), "--artifact", str(artifact), "--reference", str(reference),
               "--mode", a.mode, "--table", a.table, "--warmup", str(a.warmup), "--iters", str(a.iters)]
    if a.skip_verify:
        command.append("--skip-verify")
    result = json.loads(subprocess.check_output(command, text=True))
    result["workload"] = {key: source[key] for key in ["model_id", "hf_repo", "revision", "tensor_name", "source", "encoding", "shape", "conversion"]}
    result["artifact_sha256"] = manifest["artifact_sha256"]
    result["scale_bytes"] = manifest["scale_bytes"]
    result["ratio_total_with_scales"] = manifest["ratio_total_with_scales"]
    result["scale_consumed"] = False
    result["provenance"]["benchmark_git_commit"] = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    result["provenance"]["benchmark_dirty"] = bool(subprocess.check_output(["git", "-C", str(ROOT), "status", "--porcelain"], text=True).strip())
    result["provenance"]["binary_sha256"] = sha(ROOT / "bin/lds_bench")
    result["provenance"]["implementation_sha256"] = {name: sha(ROOT / "bench" / name) for name in ["lds_bench.cpp", "lds_format.h"]}
    print(json.dumps(result))


if __name__ == "__main__":
    main()
