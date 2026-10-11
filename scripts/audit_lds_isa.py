#!/usr/bin/env python3
"""Check all timed discard kernels, not only verification kernels."""
import hashlib
import json
from pathlib import Path
import re
import sys

path = Path(sys.argv[1])
text = path.read_text()
results = []
for name in re.findall(r'\.type\s+([^,\s]+),@function', text):
    if not re.search(r'hipansLDSILi\d+ELb0ELb[01]EE', name):
        continue
    match = re.search(re.escape(name) + r':.*?\.size\s+' + re.escape(name) + r'\b', text, re.S)
    if match:
        body = match.group()
        results.append({'function': name,
                        'decoded_ds_write_b8_sites': len(re.findall(r'\bds_write_b8\b', body)),
                        'flat_store_sites': len(re.findall(r'\b(?:flat|global|buffer)_store\w*\b', body)),
                        'barrier_sites': len(re.findall(r'\bs_barrier\b', body)),
                        'ds_read_sites': len(re.findall(r'\bds_read\w*\b', body))})
if len(results) != 14 or any(r['decoded_ds_write_b8_sites'] == 0 or r['flat_store_sites'] for r in results):
    raise RuntimeError(f'discard LDS stores missing or unexpected global stores: {results}')
print(json.dumps({'isa_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                  'discard_store_retention_pass': True, 'kernels': results}))
