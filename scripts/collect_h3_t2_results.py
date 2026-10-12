#!/usr/bin/env python3
"""Validate and publish both-machine H3 T2 ablation results (must run in Slurm)."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
R2 = {
    "mi300x": ROOT / "results/profiling/lds-r2-glm-mi300x-109848/summary.json",
    "mi350x": ROOT / "results/profiling/lds-r2-glm-mi350x-109847/summary.json",
}
JSON_OUT = ROOT / "results/h3_ablation_t2.json"
MD_OUT = ROOT / "docs/H3_ABLATION_T2.md"
EXPECTED_VARIANTS = {"A1", "A2", "A2c", "A3"}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_machine(name: str, raw_path: Path, job_id: str, binary: Path) -> dict:
    raw = json.loads(raw_path.read_text())
    if raw.get("schema") != "h3-ablation-t2-v1":
        raise ValueError(f"unexpected result schema for {name}")
    if raw.get("result_validity", "").startswith("INVALID:") is not True:
        raise ValueError(f"{name} ablation result is not explicitly marked INVALID")
    if raw.get("warmup") != 10 or raw.get("iters") != 50:
        raise ValueError(f"{name} did not use 10 warmup / 50 timed iterations")
    variants = {row["id"]: row for row in raw.get("variants", [])}
    if set(variants) != EXPECTED_VARIANTS:
        raise ValueError(f"{name} variant set mismatch: {sorted(variants)}")
    for required in ("A2", "A2c"):
        if variants[required].get("verified_output") is not True:
            raise ValueError(f"{name} {required} failed byte-exact output verification")
    if any(not row.get("result_validity", "").startswith("INVALID:") for row in variants.values()):
        raise ValueError(f"{name} contains a variant not marked INVALID")
    baseline = json.loads(R2[name].read_text())["primary_scenario"]
    return {
        "machine": name,
        "slurm_job_id": job_id,
        "raw_result": str(raw_path.relative_to(ROOT)),
        "gpu_name": raw["provenance"]["gpu_name"],
        "gpu_arch": raw["provenance"]["gpu_arch"],
        "ROCm_version": raw["provenance"].get("ROCm_version", "unknown"),
        "hipANS_commit": raw["provenance"]["hipANS_commit"],
        "build_flags": raw["provenance"]["build_flags"],
        "workload": raw["workload"],
        "launch_common": raw["launch_common"],
        "baseline_A0": {
            "source": str(R2[name].relative_to(ROOT)),
            "effective_LDS_GBps": baseline["effective_LDS_GBps"],
            "median_ms": baseline["decode_ms"]["median"],
            "LDS_bytes": baseline["launch"]["LDS_bytes"],
        },
        "variants": [variants[key] for key in ("A1", "A2", "A2c", "A3")],
        "binary_sha256": sha(binary),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mi300x", required=True, type=Path)
    parser.add_argument("--mi350x", required=True, type=Path)
    parser.add_argument("--mi300x-job", required=True)
    parser.add_argument("--mi350x-job", required=True)
    args = parser.parse_args()
    if JSON_OUT.exists() or MD_OUT.exists():
        raise FileExistsError("refusing to overwrite an existing T2 result or report")
    if subprocess.check_output(["git", "-C", str(ROOT / "../hipANS"), "status", "--porcelain"], text=True).strip():
        raise RuntimeError("hipANS source tree is not clean; refusing to publish baseline comparison")

    machines = [
        load_machine("mi300x", args.mi300x, args.mi300x_job, ROOT / "bin/hipans_lds_ablation_mi300x"),
        load_machine("mi350x", args.mi350x, args.mi350x_job, ROOT / "bin/hipans_lds_ablation_mi350x"),
    ]
    payload = {
        "schema": "h3-ablation-t2-v1",
        "result_validity": "INVALID: all A1/A2/A2c/A3 measurements are bench-only ablations, not product-code performance",
        "interpretation": {
            "A1_vs_A0": "isolates removal of the per-symbol workgroup barrier; output is still byte-checked when reported",
            "A2_vs_A2c": "A2 stages compressed words in LDS; A2c has the same additional dynamic LDS allocation but continues global loads",
            "A3_vs_A0": "combines barrier removal and compressed-input LDS staging; effects are not assumed additive",
            "throughput": "decoded output bytes written to LDS / median HIP-event ms; decimal GB/s, single tensor, one stream",
        },
        "provenance": {
            "ablation_source_sha256": sha(ROOT / "bench/hipans_lds_ablation.cpp"),
            "build_script_sha256": sha(ROOT / "scripts/build_hipans_lds_ablation.sh"),
            "job_script_sha256": sha(ROOT / "scripts/job_h3_t2.sh"),
            "collector_script_sha256": sha(Path(__file__)),
            "hipANS_dirty": False,
        },
        "machines": machines,
    }
    report = [
        "# H3 T2 bench-only decode ablations",
        "",
        "**All A1/A2/A2c/A3 measurements below are INVALID as product-code performance.** The kernels live only in `bench/`; the upstream hipANS source tree was not changed. Timing excludes the separate byte-exact verification launch.",
        "",
        "## Results",
        "",
        "| GPU | Variant | Effective LDS output | Median / p10–p90 | LDS per WG | Resident WG/CU | Output verification | Comparison purpose |",
        "|---|---|---:|---:|---:|---:|---|---|",
    ]
    purposes = {
        "A1": "barrier removed vs A0",
        "A2": "compressed words staged in LDS",
        "A2c": "A2 LDS allocation control; global reads retained",
        "A3": "A1 + A2 combined",
    }
    for machine in machines:
        report.append(
            f"| {machine['gpu_name']} | A0 current hipANS | {machine['baseline_A0']['effective_LDS_GBps']:.1f} GB/s | "
            f"{machine['baseline_A0']['median_ms']:.3f} ms | {machine['baseline_A0']['LDS_bytes']:,} B | — | R2 byte-exact passed | reference |"
        )
        for row in machine["variants"]:
            timing = row["decode_ms"]
            report.append(
                f"| {machine['gpu_name']} | {row['id']} (INVALID) | {row['effective_LDS_GBps']:.1f} GB/s | "
                f"{timing['median']:.3f} / {timing['p10']:.3f}–{timing['p90']:.3f} ms | "
                f"{row['dynamic_LDS_bytes']:,} B | {row['resident_workgroups_per_CU']} | "
                f"{'passed' if row['verified_output'] else 'FAILED'} | {purposes[row['id']]} |"
            )
    report.extend(
        [
            "",
            "## Measurement and interpretation notes",
            "",
            "- Workload: GLM-5.2-FP8 layer-0 gate projection, shape `[12288, 6144]`, F8_E4M3 bytes arranged as 9,216 independent 8 KiB tiles; one matrix, one stream, 128 threads (2 waves/WG), grid multiplier 4.",
            "- Each variant uses 10 warmups and 50 timed iterations with HIP events. Throughput numerator is decoded output bytes written to LDS; denominator is median event time. Table output is decimal GB/s.",
            "- A2/A2c must pass byte-exact verification or the collection job fails. A1/A3 verification status is recorded, but all ablation data remains invalid even if correct because these are isolated experiments, not supported codec builds.",
            "- A2 stages each tile's compressed words to a per-wave LDS region once, then decodes from LDS. A2c allocates the same extra LDS footprint but keeps the original global reads, controlling for occupancy changes. A1 removes only the copied decoder's per-symbol workgroup barrier; the table-load and output-safety barriers remain.",
            "- Compare A2 with A2c for the combined benefit of replacing dependent global refills with LDS refills after paying staging cost. Compare A1/A3 against A0 for barrier and combined effects. The effects need not add linearly.",
            "- Raw per-job JSON is retained under `results/profiling/h3-t2-*`; the combined machine results and provenance are in `results/h3_ablation_t2.json`.",
            "",
        ]
    )
    with JSON_OUT.open("x") as output:
        output.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    with MD_OUT.open("x") as output:
        output.write("\n".join(report))
    print(json.dumps({"results": str(JSON_OUT), "report": str(MD_OUT), "machines": len(machines)}))


if __name__ == "__main__":
    main()
