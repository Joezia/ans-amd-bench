#!/usr/bin/env python3
"""Convert BabelStream v5 CSV output to the committed peak-bandwidth schema."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--command", required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--gpu-arch", required=True)
    parser.add_argument("--rocm", required=True)
    parser.add_argument("--hostname", required=True)
    args = parser.parse_args()

    lines = args.csv.read_text().splitlines()
    header = next((index for index, line in enumerate(lines)
                   if line.startswith("function,")), None)
    if header is None:
        raise SystemExit("BabelStream CSV header not found")
    rows = list(csv.DictReader(lines[header:]))
    kernels: dict[str, float] = {}
    array_elements = element_bytes = iterations = None
    for row in rows:
        name = row.get("function", "")
        bandwidth = row.get("max_mbytes_per_sec")
        if not name or bandwidth is None:
            continue
        kernels[name] = float(bandwidth) / 1000.0
        array_elements = int(row["n_elements"])
        element_bytes = int(row["sizeof"])
        iterations = int(row["num_times"])
    if "Triad" not in kernels:
        raise SystemExit("BabelStream Triad row not found")

    result = {
        "schema_version": 1,
        "benchmark": "BabelStream HIP",
        "babelstream_git_commit": args.commit,
        "command": args.command,
        "hostname": args.hostname,
        "gpu_name": args.gpu,
        "gpu_arch": args.gpu_arch,
        "rocm_version": args.rocm,
        "array_elements": array_elements,
        "element_bytes": element_bytes,
        "bytes_per_array": array_elements * element_bytes,
        "total_allocated_array_bytes": 3 * array_elements * element_bytes,
        "iterations": iterations,
        "kernel_GBps": kernels,
        "selected_kernel": "Triad",
        "selected_peak_GBps": kernels["Triad"],
        "selection_reason": (
            "BabelStream Triad is the conventional achievable STREAM peak; "
            "three arrays are much larger than the 256 MiB MI300X MALL."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
