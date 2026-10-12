#!/usr/bin/env python3
"""Generate the H3 bandwidth table from verified R2 result JSONs."""
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MALL_CAPACITY_BYTES = 256 * 1024 * 1024
SOURCES = {
    "MI300X": ROOT / "results/profiling/lds-r2-glm-mi300x-109848/summary.json",
    "MI350X": ROOT / "results/profiling/lds-r2-glm-mi350x-109847/summary.json",
}
DOC_PATH = ROOT / "docs/H3_BANDWIDTH_DEFINITIONS.md"
JSON_PATH = ROOT / "results/h3_bandwidth_definitions.json"


def result_rows(path: Path, gpu: str) -> list[dict]:
    summary = json.loads(path.read_text())
    if summary.get("scenario", {}).get("measurement_scope") != (
        "decode-only LDS proxy; no activation GEMM or fused consumer"
    ):
        raise ValueError(f"unexpected measurement scope in {path}")
    primary = summary["primary_scenario"]
    cases = [primary, *summary["replica_auxiliary"]]
    rows = []
    for result in cases:
        if result.get("verified_actual_LDS") is not True:
            raise ValueError(f"unverified LDS output in {path}")
        replicas = int(result["replicas"])
        decoded_bytes = int(result["effective_decoded_bytes"])
        median_ms = float(result["decode_ms"]["median"])
        throughput = decoded_bytes / (median_ms * 1_000_000.0)
        recorded = float(result["effective_LDS_GBps"])
        if abs(throughput - recorded) / recorded > 0.001:
            raise ValueError(
                f"throughput formula mismatch in {path}: {throughput} vs {recorded}"
            )
        artifact_bytes = int(result["artifact_bytes"])
        allocated_each = int(result["compressed_allocation_bytes"]) // replicas
        allocated_total = int(result["compressed_allocation_bytes"])
        ratio = allocated_total / MALL_CAPACITY_BYTES
        if ratio < 0.90:
            relation = "below 256 MiB"
        elif ratio <= 1.10:
            relation = "near 256 MiB (slightly under)"
        else:
            relation = "above 256 MiB"
        rows.append(
            {
                "gpu": gpu,
                "replicas": replicas,
                "category": "single-matrix" if replicas == 1 else f"{replicas}-replica auxiliary",
                "decoded_bytes_numerator": decoded_bytes,
                "median_hip_event_ms_denominator": median_ms,
                "effective_lds_gbps": recorded,
                "archive_bytes_per_replica_including_metadata": artifact_bytes,
                "device_input_allocation_bytes_per_replica": allocated_each,
                "compressed_input_working_set_allocated_bytes": allocated_total,
                "compressed_input_working_set_mib": allocated_total / (1024 * 1024),
                "fraction_of_256_mib_mall_capacity": ratio,
                "mall_size_relation": relation,
                "actual_lds_verified": True,
                "source_summary": str(path.relative_to(ROOT)),
                "paper_class": "single real matrix" if replicas == 1 else (
                    "1a cache-resident approximation" if replicas == 4 else
                    "1b HBM-sourced approximation" if replicas == 64 else
                    "auxiliary cache/HBM sweep"
                ),
            }
        )
    return rows


def main() -> None:
    rows = [row for gpu, path in SOURCES.items() for row in result_rows(path, gpu)]
    payload = {
        "schema": "h3-bandwidth-definitions-v1",
        "workload": {
            "model": "zai-org/GLM-5.2-FP8",
            "tensor": "model.layers.0.mlp.gate_proj.weight",
            "shape": [12288, 6144],
            "dtype": "F8_E4M3",
            "scope": "ANS decode to LDS only; no activation GEMM or fused consumer",
        },
        "definitions": {
            "effective_lds_throughput": "effective_decoded_bytes / (decode_ms.median * 1,000,000), decimal GB/s",
            "numerator": "effective_decoded_bytes: output bytes written to LDS across all replicas",
            "denominator": "median HIP-event duration for the complete decode launch; R2 used one stream and common start/stop events",
            "mall_working_set": "replica count * per-replica compressed device input allocation, including archive metadata and allocation padding; excludes lookup/LDS/output/other resident data",
            "mall_capacity_reference_bytes": MALL_CAPACITY_BYTES,
            "cache_residency_claim": "working-set size is only a capacity comparison; it does not prove MALL residency or hit rate",
            "replica_semantics": "same tensor content copied to independent device addresses; diagnostic scale, not extra model parameters or batch",
        },
        "rows": rows,
    }
    markdown = [
        "# H3 bandwidth definitions and R2 comparison",
        "",
        "This table is generated from the verified R2 summaries listed in the source column. It separates the decoder's effective output rate from the compressed-input footprint.",
        "",
        "## Definitions",
        "",
        "- Effective LDS throughput = `effective_decoded_bytes / (median HIP-event ms × 1,000,000)`, in decimal GB/s. The numerator is decoded output bytes written to LDS across all replicas; the denominator is the median end-to-end decode event interval. It is not HBM bandwidth or measured LDS physical traffic.",
        "- Compressed input footprint = replica count × per-replica compressed device allocation. The allocation includes ANS archive metadata and alignment padding; it excludes lookup data, output buffers, and unrelated cache occupants.",
        "- The 256 MiB comparison is a capacity-only estimate. Being below it does not establish MALL residency; actual residency requires counters or a controlled cache experiment.",
        "- Replicas are identical tensor contents at independent GPU addresses. They are auxiliary cache/scale diagnostics, not distinct model weights, batch items, or real inference throughput.",
        "",
        "## Measurements",
        "",
        "| GPU | Case | Effective LDS output | Numerator: decoded bytes | Denominator: median event | Compressed archive / replica | Allocated compressed working set | vs 256 MiB MALL | Verification |",
        "|---|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in rows:
        replicas = row["replicas"]
        case = "single matrix" if replicas == 1 else f"{replicas} replicas"
        markdown.append(
            f"| {row['gpu']} | {case} ({row['paper_class']}) | "
            f"{row['effective_lds_gbps']:.1f} GB/s | {row['decoded_bytes_numerator']:,} B | "
            f"{row['median_hip_event_ms_denominator']:.3f} ms | "
            f"{row['archive_bytes_per_replica_including_metadata'] / (1024 * 1024):.2f} MiB | "
            f"{row['compressed_input_working_set_mib']:.2f} MiB "
            f"({row['fraction_of_256_mib_mall_capacity'] * 100:.1f}%) | "
            f"{row['mall_size_relation']} | passed |"
        )
    markdown.extend(
        [
            "",
            "The 4-replica case is a cache-resident *approximation* (about 249.5 MiB allocated, just under 256 MiB); do not describe it as confirmed cache-resident. The 64-replica case is about 3.90 GiB and exceeds that capacity. The single matrix is about 62.4 MiB. MI300X and MI350X figures are separate hardware measurements; the MI350X case has no measured BabelStream denominator in this table.",
            "",
            "## Sources and scope",
            "",
            "- `results/profiling/lds-r2-glm-mi300x-109848/summary.json`",
            "- `results/profiling/lds-r2-glm-mi350x-109847/summary.json`",
            "- Both summaries mark `verified_actual_LDS: true`. The single-matrix case uses 10 warmups and 50 timed iterations; replica sweeps use 5 warmups and 20 timed iterations.",
            "- Rates are the existing R2 values. The generator recomputes output bytes divided by median event time and rejects discrepancies over 0.1%.",
            "",
        ]
    )
    with DOC_PATH.open("x") as output:
        output.write("\n".join(markdown))
    with JSON_PATH.open("x") as output:
        output.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"doc": str(DOC_PATH), "json": str(JSON_PATH), "rows": len(rows)}))


if __name__ == "__main__":
    main()
