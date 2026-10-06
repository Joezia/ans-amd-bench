#!/usr/bin/env python3
"""Generate byte-exact repeated inputs and metadata for cache/tile sweeps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sizes-mib", default="64,192,384")
    parser.add_argument("--tile-sizes", default="8192,65536,1048576")
    args = parser.parse_args()

    source = args.input.read_bytes()
    if not source:
        raise SystemExit("input must be nonempty")
    sizes = [int(value) * 1024 * 1024 for value in args.sizes_mib.split(",")]
    tiles = [int(value) for value in args.tile_sizes.split(",")]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = ["case\tinput\tmeta\tbytes\ttile_size"]
    for size in sizes:
        target = args.output_dir / f"input-{size // 1024 // 1024}MiB.bin"
        with target.open("wb") as handle:
            whole, tail = divmod(size, len(source))
            for _ in range(whole):
                handle.write(source)
            handle.write(source[:tail])
        for tile in tiles:
            name = f"size-{size // 1024 // 1024}MiB_tile-{tile}"
            meta = args.output_dir / f"{name}.json"
            meta.write_text(json.dumps({
                "num_elems": size,
                "bits_per_symbol": 8,
                "tile_size": tile,
            }, sort_keys=True) + "\n")
            manifest.append(f"{name}\t{target}\t{meta}\t{size}\t{tile}")
    (args.output_dir / "cases.tsv").write_text("\n".join(manifest) + "\n")


if __name__ == "__main__":
    main()
