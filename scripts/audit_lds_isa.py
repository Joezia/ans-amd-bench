#!/usr/bin/env python3
"""Check timed discard functions themselves, not only verification kernels."""
import hashlib
import json
from pathlib import Path
import re
import sys

path = Path(sys.argv[1])
text = path.read_text()
results = []
for name in re.findall(r"\.type\s+([^,\s]+),@function", text):
    if "ldsBaselineILb0" not in name:
        continue
    match = re.search(re.escape(name) + r":.*?\.size\s+" + re.escape(name) + r"\b", text, re.S)
    if match:
        body = match.group()
        results.append({"function": name, "ds_write_instruction_sites": len(re.findall(r"\bds_write\w*\b", body)),
                        "decoded_ds_write_b8_sites": len(re.findall(r"\bds_write_b8\b", body)),
                        "flat_store_sites": len(re.findall(r"\bflat_store\w*\b", body)),
                        "ds_read_instruction_sites": len(re.findall(r"\bds_read\w*\b", body))})
if len(results) != 4 or any(r["decoded_ds_write_b8_sites"] == 0 or r["flat_store_sites"] for r in results):
    raise RuntimeError(f"discard-kernel LDS stores missing or unexpected function inventory: {results}")
print(json.dumps({"isa_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                  "discard_store_retention_pass": True, "kernels": results}))
