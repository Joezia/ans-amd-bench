#!/usr/bin/env python3
"""Generate deterministic benchmark input files (seed fixed) for the hipANS smoke run."""
import os
import sys
import numpy as np

out_dir = sys.argv[1] if len(sys.argv) > 1 else "."
os.makedirs(out_dir, exist_ok=True)
rng = np.random.default_rng(12345)

MiB = 1024 * 1024

def write(name, arr):
    p = os.path.join(out_dir, name)
    arr.tofile(p)
    print(f"{name}: {os.path.getsize(p)} bytes")

# 1) uniform random bytes: incompressible, worst-case entropy
write("uniform_u8_64MiB.bin", rng.integers(0, 256, size=64 * MiB, dtype=np.uint8))

# 2) skewed bytes, same generator family as ANSTest::generateSymbols (exponential, lambda=20)
s = np.minimum(rng.exponential(1.0 / 20.0, size=64 * MiB + 1234), 1.0)
write("expo20_u8_64MiB+1234.bin", (s * 256.0).astype(np.uint8))

# 3) float32 N(0,1), 16M elements = 64 MiB (what benchmark.cpp / fpcompress consume)
write("normal_f32_64MiB.bin", rng.standard_normal(16 * MiB, dtype=np.float32))

# 4) float16 N(0,1), 32M elements = 64 MiB
write("normal_f16_64MiB.bin", rng.standard_normal(32 * MiB, dtype=np.float32).astype(np.float16))

# 5) bfloat16 N(0,1): upper 16 bits of float32, 32M elements = 64 MiB
f32 = rng.standard_normal(32 * MiB, dtype=np.float32)
write("normal_bf16_64MiB.bin", (f32.view(np.uint32) >> 16).astype(np.uint16))
