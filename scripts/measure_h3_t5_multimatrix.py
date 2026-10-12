#!/usr/bin/env python3
"""Measure one large, byte-exact multi-matrix GLM artifact on one GPU."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MALL_BYTES = 256 << 20


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--machine", choices=("mi300x", "mi350x"), required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--threads", required=True, type=int)
    parser.add_argument("--grid", required=True, type=int)
    parser.add_argument("--streams", required=True, type=int)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    if manifest.get("schema") != "hipans-lds-v1":
        raise ValueError("not a prepared current-hipANS artifact")
    if manifest.get("unique_weight_bytes", 0) <= MALL_BYTES or manifest.get("artifact_bytes", 0) <= MALL_BYTES:
        raise ValueError("T5 working set must exceed the 256 MiB MALL capacity in both raw and compressed bytes")
    command = [sys.executable, str(ROOT / "scripts/run_lds_bench.py"),
               "--manifest", str(args.manifest), "--mode", "decode", "--replicas", "1",
               "--threads", str(args.threads), "--grid-mult", str(args.grid),
               "--streams", str(args.streams), "--warmup", str(args.warmup), "--iters", str(args.iters)]
    proc = subprocess.run(command, text=True, capture_output=True)
    if proc.returncode:
        raise RuntimeError(f"benchmark failed rc={proc.returncode}: {proc.stderr[-8000:]}")
    measurement = json.loads(proc.stdout)
    if measurement.get("verified_actual_LDS") is not True:
        raise RuntimeError("T5 result failed actual-LDS byte-exact verification")
    result = {
        "schema": "h3-t5-multimatrix-measurement-v1",
        "status": "measured",
        "machine": args.machine,
        "slurm_job": args.job_id,
        "artifact_manifest": str(args.manifest),
        "artifact_sha256": manifest["artifact_sha256"],
        "distinct_matrix_count": 9,
        "unique_decoded_bytes": measurement["unique_weight_bytes"],
        "compressed_archive_bytes": measurement["artifact_bytes"],
        "metadata_bytes": measurement["artifact_metadata_bytes"],
        "mall_capacity_bytes": MALL_BYTES,
        "unique_decoded_bytes_over_mall": measurement["unique_weight_bytes"] / MALL_BYTES,
        "compressed_archive_bytes_over_mall": measurement["artifact_bytes"] / MALL_BYTES,
        "launch": measurement["launch"],
        "decode_ms": measurement["decode_ms"],
        "effective_LDS_GBps": measurement["effective_LDS_GBps"],
        "compressed_archive_GBps": measurement["compressed_archive_GBps"],
        "ratio_codes": measurement["ratio_codes"],
        "ratio_total_with_scales": measurement["ratio_total_with_scales"],
        "warmup": args.warmup,
        "iters": args.iters,
        "verified_actual_LDS": True,
        "throughput_definition": "unique decoded FP8 bytes / HIP-event median decode time; decimal GB/s",
        "cache_interpretation": "both the unique raw collection and its compressed archive exceed nominal 256 MiB MALL; capacity class only, not direct proof of cache misses",
        "provenance": measurement["provenance"]
    }
    result_path = ROOT / "results" / f"h3_t5_measure_{args.machine}_{args.job_id}.json"
    report_path = ROOT / "docs" / f"H3_T5_MEASURE_{args.machine.upper()}_{args.job_id}.md"
    if result_path.exists() or report_path.exists():
        raise FileExistsError("refusing to overwrite existing T5 result/report")
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    launch = measurement["launch"]
    report = [f"# H3 T5 multi-matrix decode — {args.machine.upper()}", "",
              f"Slurm job `{args.job_id}` measured the byte-exact nine-matrix composite. The launch was selected from the same GPU's T4 single-matrix sweep and uses one independent artifact address (no replicas).", "",
              f"- Decoded output: {measurement['unique_weight_bytes']:,} bytes ({measurement['unique_weight_bytes'] / (1<<20):.1f} MiB).",
              f"- Compressed ANS archive: {measurement['artifact_bytes']:,} bytes ({measurement['artifact_bytes'] / (1<<20):.1f} MiB), {measurement['artifact_bytes'] / MALL_BYTES:.2f}× nominal 256 MiB MALL capacity.",
              f"- Launch: {launch['threads']} threads/WG, grid multiplier {launch['grid_multiplier']}, {launch['streams']} stream(s); LDS {launch['LDS_bytes']:,} B/WG.",
              f"- Decode median/p10/p90: {measurement['decode_ms']['median']:.6f}/{measurement['decode_ms']['p10']:.6f}/{measurement['decode_ms']['p90']:.6f} ms.",
              f"- Effective LDS output throughput: **{measurement['effective_LDS_GBps']:.2f} decimal GB/s**; compressed archive read rate: {measurement['compressed_archive_GBps']:.2f} decimal GB/s.",
              f"- Correctness: actual LDS output byte-exact verification passed; warmup={args.warmup}, iters={args.iters}.", "",
              "This is a capacity-class 1b-style large-working-set measurement, not proof of every load missing MALL: the hardware profiler does not expose a direct MALL-hit counter here. The decoder pseudo-matrix is a byte layout of distinct source matrices; it is not one logical model tensor and does not run GEMM. The nine tensor streams were jointly re-encoded with one p10 table to form one HIP-ANS artifact.", "",
              f"Artifact SHA-256: `{manifest['artifact_sha256']}`. Curated result: [`{result_path.name}`](../results/{result_path.name}).", ""]
    report_path.write_text("\n".join(report))
    print(json.dumps({"result": str(result_path), "report": str(report_path),
                      "effective_LDS_GBps": measurement["effective_LDS_GBps"], "verified": True}))


if __name__ == "__main__":
    main()
