#!/usr/bin/env python3
"""Run standardized raw-decode cases for the six Figure 9 model inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess


def fmt(value: object, digits: int = 3) -> str:
    return "—" if value is None else f"{float(value):.{digits}f}"


def fmt_percent(value: object) -> str:
    return "—" if value is None else f"{100 * float(value):.1f}%"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--peak-gbps", type=float)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--tile-size", type=int, default=65536)
    args = parser.parse_args()

    cases = json.loads(args.config.read_text())
    rows: list[dict[str, object]] = []
    for case in cases:
        binary = args.input_dir / f"{case['slug']}.bin"
        meta = args.input_dir / f"{case['slug']}.meta.json"
        row = dict(case)
        row["input"] = str(binary)
        if not binary.is_file():
            row["status"] = "MISSING_INPUT"
            rows.append(row)
            continue
        if not meta.exists():
            # hipANS treats a bf16 file as a stream of byte symbols.  num_elems
            # is therefore the exact decoded byte count, not bf16 element count.
            meta.write_text(json.dumps({
                "num_elems": binary.stat().st_size,
                "bits_per_symbol": 8,
                "tile_size": args.tile_size,
            }, sort_keys=True) + "\n")
        command = [str(args.bench), "--input", str(binary), "--meta", str(meta),
                   "--warmup", str(args.warmup), "--iters", str(args.iters)]
        if args.peak_gbps:
            command += ["--peak-GBps", str(args.peak_gbps)]
        try:
            process = subprocess.run(command, check=True, capture_output=True,
                                     text=True, timeout=7200)
            row["status"] = "OK"
            row["amd"] = json.loads(process.stdout)
            row["stderr"] = process.stderr
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired,
                json.JSONDecodeError) as error:
            row["status"] = "FAILED"
            row["error"] = str(error)
        rows.append(row)

    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")

    table = [
        "# Paper microbenchmark comparison",
        "",
        "Tan et al. Figure 9 reports speedup of fused decode+GEMM over a naive",
        "DietGPU-decode-plus-CUTLASS baseline. The AMD columns below are the",
        "standalone global-output hipANS decoder and are intentionally not divided",
        "into that speedup. This side-by-side table prevents unlike metrics from",
        "being silently conflated.",
        "",
        "| Model (paper matrix K×N) | Paper tile-aligned speedup | Paper final speedup | AMD ratio | AMD decoded GB/s | AMD utilization | Status |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        amd = row.get("amd", {})
        util = amd.get("utilization_vs_peak") if isinstance(amd, dict) else None
        table.append(
            f"| {row['model']} ({row['matrix'][0]}×{row['matrix'][1]}) "
            f"| {fmt(row['paper_tile_aligned_speedup'], 2)}× "
            f"| {fmt(row['paper_final_speedup'], 2)}× "
            f"| {fmt(amd.get('ratio') if isinstance(amd, dict) else None)} "
            f"| {fmt(amd.get('decode_GBps_over_decoded_written') if isinstance(amd, dict) else None, 1)} "
            f"| {fmt_percent(util)} "
            f"| {row['status']} |"
        )
    table += [
        "",
        "Paper values were transcribed from Figure 9 of arXiv:2606.15789. A row",
        "is measured only when `<slug>.bin` exists in the input directory; missing",
        "licensed model weights remain explicit rather than being replaced by",
        "synthetic data.",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(table) + "\n")


if __name__ == "__main__":
    main()
