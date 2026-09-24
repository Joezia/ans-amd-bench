#!/usr/bin/env python3
"""Dump raw tensor bytes out of a .safetensors file.

No third-party dependencies - standard library only. Works on Python 3.6+.

Usage:
    python3 dump_weights.py list  model.safetensors
    python3 dump_weights.py dump  model.safetensors OUT.bin PATTERN [--dtype BF16] [--limit BYTES]

Examples:
    # see what is in the file
    python3 dump_weights.py list model.safetensors

    # one representative MLP weight from layer 0
    python3 dump_weights.py dump model.safetensors w_single.bin \
        'layers\\.0\\.mlp\\.gate_proj\\.weight' --dtype BF16

    # concatenate MLP weights until we have 256 MiB
    python3 dump_weights.py dump model.safetensors w_mlp_256MiB.bin \
        'mlp\\.(gate|up|down)_proj\\.weight' --dtype BF16 --limit 268435456
"""

import hashlib
import json
import re
import struct
import sys


def load_header(path):
    """Return (header_dict, offset_of_tensor_data_region)."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n).decode("utf-8"))
    return header, 8 + n


def entries(header):
    for name, meta in header.items():
        if name != "__metadata__":
            yield name, meta


def cmd_list(path):
    header, _ = load_header(path)
    rows = []
    for name, meta in entries(header):
        begin, end = meta["data_offsets"]
        rows.append((end - begin, meta["dtype"], tuple(meta["shape"]), name))
    rows.sort(reverse=True)
    total = 0
    for size, dtype, shape, name in rows:
        total += size
        print("%12d  %-6s %-20s %s" % (size, dtype, shape, name))
    print("\n%d tensors, %d total bytes" % (len(rows), total))


def cmd_dump(path, out, pattern, dtype_filter=None, limit=0):
    header, base = load_header(path)
    rx = re.compile(pattern)

    selected = [(n, m) for n, m in entries(header) if rx.search(n)]
    if dtype_filter:
        selected = [(n, m) for n, m in selected if m["dtype"] == dtype_filter]
    selected.sort(key=lambda t: t[0])

    if not selected:
        sys.exit("no tensor matched pattern %r (dtype filter %r)" % (pattern, dtype_filter))

    digest = hashlib.md5()
    written = 0
    with open(path, "rb") as src, open(out, "wb") as dst:
        for name, meta in selected:
            if limit and written >= limit:
                break
            begin, end = meta["data_offsets"]
            src.seek(base + begin)
            remaining = end - begin
            while remaining > 0:
                if limit and written >= limit:
                    break
                chunk = src.read(min(1 << 20, remaining))
                if not chunk:
                    break
                if limit and written + len(chunk) > limit:
                    chunk = chunk[: limit - written]
                dst.write(chunk)
                digest.update(chunk)
                written += len(chunk)
                remaining -= len(chunk)
            print("  + %-56s %-6s %12d B   (running total %d)"
                  % (name, meta["dtype"], end - begin, written))

    print("\nwrote %s" % out)
    print("  bytes : %d" % written)
    print("  md5   : %s" % digest.hexdigest())


def main(argv):
    if len(argv) < 3:
        sys.exit(__doc__)
    cmd = argv[1]
    if cmd == "list":
        cmd_list(argv[2])
        return
    if cmd == "dump":
        if len(argv) < 5:
            sys.exit(__doc__)
        model, out, pattern = argv[2], argv[3], argv[4]
        dtype_filter, limit = None, 0
        rest = argv[5:]
        i = 0
        while i < len(rest):
            if rest[i] == "--dtype":
                dtype_filter = rest[i + 1]
                i += 2
            elif rest[i] == "--limit":
                limit = int(rest[i + 1])
                i += 2
            else:
                sys.exit("unknown option %r" % rest[i])
        cmd_dump(model, out, pattern, dtype_filter, limit)
        return
    sys.exit(__doc__)


if __name__ == "__main__":
    main(sys.argv)
