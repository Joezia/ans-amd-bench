"""Actual LDS validation of scalar-codec corner cases (inside GPU allocation)."""
import json
from pathlib import Path
import random
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
base = Path(sys.argv[1])
base.mkdir(parents=True, exist_ok=True)
rng = random.Random(0)
cases = [("constant", 64, 128, bytes([147]) * 8192),
         ("alphabet", 128, 256, bytes(range(256)) * 128),
         ("tail", 67, 139, bytes(rng.randrange(256) for _ in range(67 * 139))),
         ("one", 1, 1, b"\xff"),
         ("skew", 64, 128, bytes(rng.choices(range(256), weights=[3000] + [1] * 255, k=8192)))]
results = []
for name, rows, cols, data in cases:
    raw, artifact = base / f"{name}.bin", base / f"{name}.ans"
    raw.write_bytes(data)
    subprocess.run([str(ROOT / "bin/lds_pack"), str(raw), str(rows), str(cols), str(artifact)], check=True, stdout=subprocess.DEVNULL)
    for mode, table in [("decode", "lds"), ("decode", "global"), ("raw", "lds")]:
        result = json.loads(subprocess.check_output([str(ROOT / "bin/lds_bench"), "--artifact", str(artifact), "--reference", str(raw),
                            "--mode", mode, "--table", table, "--warmup", "0", "--iters", "1"], text=True))
        assert result["verified_actual_LDS"]
        results.append({"case": name, "mode": mode, "table": table, "verified_actual_LDS": True})
(base / "validation.json").write_text(json.dumps(results, indent=2) + "\n")
print(f"GPU LDS validation passed: {len(results)} cases")
