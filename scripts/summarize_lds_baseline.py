#!/usr/bin/env python3
"""Curate measured LDS evidence without binary weights or machine-local paths."""
import argparse
import json
from pathlib import Path
import re


def metrics(text):
    result, headings = {}, []
    for line in text.splitlines():
        if not line.startswith("│"):
            continue
        fields = [v.strip() for v in line.split("│")[1:-1]]
        if fields and fields[0] == "Metric_ID":
            headings = fields
        elif fields and re.fullmatch(r"\d+\.\d+\.\d+", fields[0]):
            result[fields[0]] = dict(zip(headings, fields))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True, type=Path)
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    runs = {name: json.loads((a.run / f"{name}.json").read_text()) for name in ["decode-lds", "decode-global", "raw", "lds-control"]}
    artifact = json.loads(a.manifest.read_text())
    assert all(r["artifact_sha256"] == artifact["artifact_sha256"] for r in runs.values())
    assert all(runs[name]["verified_actual_LDS"] for name in ["decode-lds", "decode-global", "raw"])
    validation = json.loads((a.run / "gpu-cases/validation.json").read_text())
    assert len(validation) == 15 and all(r["verified_actual_LDS"] for r in validation)
    isa = json.loads((a.run / "isa.json").read_text())
    assert isa["discard_store_retention_pass"]
    result = {"schema": "lds-baseline-summary-v1", "job_id": a.run.name.removeprefix("lds-"),
              "artifact": artifact, "measurements": runs, "gpu_validation": validation,
              "isa": isa, "tool_availability": (a.run / "profile/availability.tsv").read_text(),
              "official_counter_metrics": metrics((a.run / "analysis.txt").read_text()),
              "analysis_dispatch": int((a.run / "analysis-dispatch.txt").read_text()),
              "limitations": ["one 8 MiB matrix, not cold >256 MiB streaming", "local byte-rANS reference, not optimized hipANS archive decoder",
                              "no GEMM/dequantization; scale stored but not consumed", "LDS staging included; global LUT alternative not resident-LDS",
                              "MALL hit rate and separate barrier time unavailable", "roofline microbenchmark not collected; BabelStream is a separate HBM reference"]}
    a.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(a.output), "job_id": result["job_id"], "GPU_validations": len(validation), "metrics": len(result["official_counter_metrics"])}))


if __name__ == "__main__":
    main()
