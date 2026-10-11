#!/usr/bin/env python3
"""Prepare a native hipANS archive; verify actual LDS output before publication."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', required=True, type=Path)
    p.add_argument('--replace', action='store_true')
    p.add_argument('--output-dir', type=Path)
    a = p.parse_args()
    source = json.loads(a.source.read_text())
    base = a.source.parent
    if source['source'] != 'native' or source['encoding'] not in ('F8_E4M3', 'F8_E5M2'):
        raise ValueError('native FP8 required; no 4-bit or simulated conversion')
    for tensor in source['tensors'].values():
        path = base / tensor['file']
        if path.stat().st_size != tensor['bytes'] or sha(path) != tensor['sha256']:
            raise ValueError('source integrity check failed')
    destination = a.output_dir or base
    destination.mkdir(parents=True, exist_ok=True)
    artifact, manifest = destination / 'weights.ans', destination / 'artifact.json'
    if (artifact.exists() or manifest.exists()) and not a.replace:
        raise ValueError('existing artifact requires --replace')
    reference = base / source['tensors'][source['tensor_name']]['file']
    command = [os.environ.get('HIPANS_LDS_BINARY', str(ROOT / 'bin/hipans_lds_bench')), '--prepare', '--artifact', str(artifact),
               '--reference', str(reference), '--rows', str(source['shape'][0]),
               '--cols', str(source['shape'][1]), '--warmup', '1', '--iters', '3']
    if a.replace:
        command.append('--replace')
    stats = json.loads(subprocess.check_output(command, text=True))
    if not stats['verified_actual_LDS']:
        raise ValueError('unverified native archive')
    scale_bytes = sum(t['bytes'] for n, t in source['tensors'].items() if n != source['tensor_name'])
    result = {**stats, 'source': source, 'source_directory': str(base.resolve()), 'artifact_file': artifact.name,
              'artifact_sha256': sha(artifact), 'source_manifest_sha256': sha(a.source),
              'scale_bytes': scale_bytes, 'scales_compressed': False,
              'ratio_total_with_scales': (stats['unique_weight_bytes'] + scale_bytes) / (stats['artifact_bytes'] + scale_bytes),
              'layout': 'row-major 64x128 tiles, zero edge padding',
              'codebook_scope': 'one matrix, not whole transformer layer'}
    temporary = manifest.with_suffix('.json.pending')
    temporary.write_text(json.dumps(result, indent=2) + '\n')
    temporary.replace(manifest)
    print(json.dumps(result))

if __name__ == '__main__':
    main()
