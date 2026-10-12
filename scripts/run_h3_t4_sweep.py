#!/usr/bin/env python3
"""Run the H3 A4 single-matrix deployment sweep inside a Slurm allocation."""
import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
THREADS = (64, 128, 192, 256, 320, 384, 448)
GRIDS = (1, 2, 4, 8)
STREAMS = (1, 2, 4, 8)


def run_one(manifest, output_dir, replicas, threads, grid, streams, warmup, iters, label):
    command = [sys.executable, str(ROOT / "scripts/run_lds_bench.py"),
               "--manifest", str(manifest), "--mode", "decode",
               "--replicas", str(replicas), "--threads", str(threads),
               "--grid-mult", str(grid), "--streams", str(streams),
               "--warmup", str(warmup), "--iters", str(iters)]
    proc = subprocess.run(command, text=True, capture_output=True)
    record = {"label": label, "status": "failed", "command": command,
              "returncode": proc.returncode, "stderr": proc.stderr[-8000:]}
    if proc.returncode == 0:
        try:
            record = json.loads(proc.stdout)
            record.update(label=label, status="ok")
            if record.get("verified_actual_LDS") is not True:
                record.update(status="failed", error="actual-LDS byte-exact verification did not pass")
        except json.JSONDecodeError as error:
            record.update(error=f"benchmark stdout was not one JSON object: {error}",
                          stdout=proc.stdout[-8000:])
    filename = f"{label}-t{threads}-g{grid}-s{streams}-r{replicas}.json"
    (output_dir / filename).write_text(json.dumps(record, indent=2) + "\n")
    return record


def write_svg(records, path, machine):
    points = [r for r in records if r.get("status") == "ok"]
    width, height = 920, 540
    left, right, top, bottom = 78, 22, 48, 76
    xs = [r["launch"]["max_resident_workgroups_per_CU"] * r["launch"]["waves_per_workgroup"] for r in points]
    ys = [r["effective_LDS_GBps"] for r in points]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    if ymin == ymax:
        ymin, ymax = max(0, ymin - 1), ymax + 1
    xpad = max(1, (xmax - xmin) * 0.05)
    ypad = (ymax - ymin) * 0.08
    xmin -= xpad
    xmax += xpad
    ymin = max(0, ymin - ypad)
    ymax += ypad

    def xmap(value):
        return left + (value - xmin) * (width - left - right) / (xmax - xmin)

    def ymap(value):
        return top + (ymax - value) * (height - top - bottom) / (ymax - ymin)

    colors = {1: "#2563eb", 2: "#16a34a", 4: "#d97706", 8: "#dc2626"}
    chunks = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
              '<rect width="100%" height="100%" fill="white"/>',
              f'<text x="{width/2}" y="26" text-anchor="middle" font-family="sans-serif" font-size="18">{machine}: single-matrix A4 launch sweep</text>']
    for tick in range(5):
        xv = xmin + (xmax - xmin) * tick / 4
        yv = ymin + (ymax - ymin) * tick / 4
        xp, yp = xmap(xv), ymap(yv)
        chunks.append(f'<line x1="{xp:.1f}" y1="{top}" x2="{xp:.1f}" y2="{height-bottom}" stroke="#e5e7eb"/>')
        chunks.append(f'<text x="{xp:.1f}" y="{height-bottom+22}" text-anchor="middle" font-family="sans-serif" font-size="12">{xv:.1f}</text>')
        chunks.append(f'<line x1="{left}" y1="{yp:.1f}" x2="{width-right}" y2="{yp:.1f}" stroke="#e5e7eb"/>')
        chunks.append(f'<text x="{left-10}" y="{yp+4:.1f}" text-anchor="end" font-family="sans-serif" font-size="12">{yv:.0f}</text>')
    chunks.extend([f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#111827"/>',
                   f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#111827"/>',
                   f'<text x="{(left+width-right)/2}" y="{height-18}" text-anchor="middle" font-family="sans-serif" font-size="14">Theoretical resident waves per CU (occupancy API)</text>',
                   f'<text x="20" y="{height/2}" transform="rotate(-90 20 {height/2})" text-anchor="middle" font-family="sans-serif" font-size="14">Effective LDS output (decimal GB/s)</text>'])
    for r in points:
        launch = r["launch"]
        resident_waves = launch["max_resident_workgroups_per_CU"] * launch["waves_per_workgroup"]
        cx, cy = xmap(resident_waves), ymap(r["effective_LDS_GBps"])
        color = colors[launch["streams"]]
        chunks.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="4" fill="{color}" fill-opacity="0.65"><title>threads={launch["threads"]}, grid={launch["grid_multiplier"]}, streams={launch["streams"]}, {r["effective_LDS_GBps"]:.2f} GB/s</title></circle>')
    legend_x = left + 10
    for i, stream_count in enumerate(STREAMS):
        lx = legend_x + i * 105
        chunks.append(f'<circle cx="{lx}" cy="{top-12}" r="5" fill="{colors[stream_count]}"/><text x="{lx+10}" y="{top-8}" font-family="sans-serif" font-size="12">{stream_count} stream(s)</text>')
    chunks.append('</svg>')
    path.write_text("\n".join(chunks) + "\n")


def write_report(result, path):
    records = result["sweep"]
    good = [r for r in records if r.get("status") == "ok"]
    top = sorted(good, key=lambda r: r["effective_LDS_GBps"], reverse=True)[:10]
    grouped = {}
    for record in good:
        launch = record["launch"]
        waves = launch["max_resident_workgroups_per_CU"] * launch["waves_per_workgroup"]
        grouped.setdefault(waves, []).append(record["effective_LDS_GBps"])
    lines = [f"# H3 T4 single-matrix launch sweep — {result['machine']}", "",
             f"Slurm job `{result['slurm_job']}` completed with {len(good)}/{len(records)} valid sweep points.", "",
             "Workload is the one real GLM-5.2-FP8 layer-0 `gate_proj` matrix (12,288 × 6,144, 72 MiB); one matrix, one replica. The MI350X-created ANS artifact is consumed unchanged on the selected GPU; the archive is device-independent and its source/reference hashes are checked. Each invocation verifies actual LDS output byte-for-byte.", "",
             "The 112-point sweep crosses threads 64–448, grid multiplier 1/2/4/8, and 1/2/4/8 HIP streams (5 warmups, 30 timed HIP-event iterations). The top three configurations receive a fresh 10-warmup/50-iteration confirmation. Throughput is decoded output bytes divided by median time, decimal GB/s.", "",
             "Occupancy x-values use the HIP occupancy API's maximum resident workgroups per CU × waves per workgroup. They are theoretical resource limits, not measured time-average wave occupancy. Repeated single-matrix reads are cache-resident-sized; this is not the HBM-streaming 1b case.", "",
             f"Best confirmed: **{result['best_confirmation']['effective_LDS_GBps']:.2f} GB/s** at threads={result['best_confirmation']['launch']['threads']}, grid={result['best_confirmation']['launch']['grid_multiplier']}, streams={result['best_confirmation']['launch']['streams']}; median={result['best_confirmation']['decode_ms']['median']:.6f} ms.", "",
             f"![Resident waves and throughput]({result['plot_file'].split('/')[-1]})", "",
             "## Top sweep points", "",
             "| Rank | Threads | Grid × | Streams | Theoretical waves/CU | Median ms | Effective LDS GB/s |", "|---:|---:|---:|---:|---:|---:|---:|"]
    for rank, r in enumerate(top, 1):
        launch = r["launch"]
        waves = launch["max_resident_workgroups_per_CU"] * launch["waves_per_workgroup"]
        lines.append(f"| {rank} | {launch['threads']} | {launch['grid_multiplier']} | {launch['streams']} | {waves} | {r['decode_ms']['median']:.6f} | {r['effective_LDS_GBps']:.2f} |")
    lines.extend(["", "## Grouped by theoretical resident waves/CU", "", "| Waves/CU | Configurations | Median throughput range GB/s |", "|---:|---:|---:|"])
    for waves in sorted(grouped):
        values = grouped[waves]
        lines.append(f"| {waves} | {len(values)} | {statistics.median(values):.2f} ({min(values):.2f}–{max(values):.2f}) |")
    lines.extend(["", "Raw point-level measurements and provenance are in the sibling JSON file. The sweep changes launch/deployment parameters only; it does not modify the codec or decoder algorithm.", ""])
    path.write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--machine", choices=("mi300x", "mi350x"), required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    if not args.manifest.is_file():
        raise FileNotFoundError(args.manifest)
    journal = args.output_dir / "journal.jsonl"
    records = []
    for threads in THREADS:
        for grid in GRIDS:
            for streams in STREAMS:
                label = f"sweep-{len(records):03d}"
                record = run_one(args.manifest, args.output_dir, 1, threads, grid, streams,
                                 5, 30, label)
                records.append(record)
                with journal.open("a") as handle:
                    handle.write(json.dumps(record) + "\n")
                print(f"{label}: t={threads} g={grid} s={streams} status={record['status']} "
                      f"GBps={record.get('effective_LDS_GBps', 'n/a')}", flush=True)
    failures = [r for r in records if r.get("status") != "ok"]
    if failures:
        raise RuntimeError(f"{len(failures)} of {len(records)} sweep points failed; inspect {journal}")
    ranked = sorted(records, key=lambda r: r["effective_LDS_GBps"], reverse=True)
    confirmations = []
    for rank, candidate in enumerate(ranked[:3]):
        launch = candidate["launch"]
        confirmation = run_one(args.manifest, args.output_dir, 1, launch["threads"],
                               launch["grid_multiplier"], launch["streams"], 10, 50,
                               f"confirmation-{rank+1}")
        confirmations.append(confirmation)
    if any(r.get("status") != "ok" for r in confirmations):
        raise RuntimeError("a top-three confirmation failed")
    best = max(confirmations, key=lambda r: r["effective_LDS_GBps"])
    result = {"schema": "h3-t4-single-matrix-sweep-v1", "status": "measured",
              "machine": args.machine, "slurm_job": args.job_id,
              "actual_hipANS": True, "artifact_manifest": str(args.manifest),
              "artifact_scope": "single real GLM-5.2-FP8 layer-0 gate_proj matrix",
              "verified_all_sweep_outputs": True, "sweep": records,
              "confirmations": confirmations, "best_confirmation": best,
              "plot_file": str(args.report.with_suffix(".svg")),
              "claim": "best confirmed within tested one-matrix deployment envelope; not a mathematical ceiling"}
    args.result.write_text(json.dumps(result, indent=2) + "\n")
    write_svg(records, args.report.with_suffix(".svg"), args.machine.upper())
    write_report(result, args.report)
    print(json.dumps({"result": str(args.result), "report": str(args.report),
                      "best_GBps": best["effective_LDS_GBps"], "verified": True}))


if __name__ == "__main__":
    main()
