#!/usr/bin/env python3
"""Integrity-checked current-hipANS LDS benchmark; stdout is one JSON object."""
import argparse
import json
import os
from pathlib import Path
import subprocess
from prepare_lds_artifact import ROOT, sha

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', required=True, type=Path)
    p.add_argument('--mode', choices=['decode', 'raw'], default='decode')
    for name, default in [('warmup', 10), ('iters', 50), ('replicas', 1), ('threads', 128), ('grid-mult', 4), ('streams', 1)]:
        p.add_argument('--' + name, type=int, default=default)
    p.add_argument('--skip-verify', action='store_true', help='profiler only after verified measurement')
    a = p.parse_args()
    manifest = json.loads(a.manifest.read_text())
    if manifest.get('schema') != 'hipans-lds-v1':
        raise ValueError('not a native hipANS artifact; prepare on GPU first')
    base, source = a.manifest.parent, manifest['source']
    artifact = base / manifest['artifact_file']
    if sha(artifact) != manifest['artifact_sha256']:
        raise ValueError('archive hash mismatch')
    for tensor in source['tensors'].values():
        if sha(base / tensor['file']) != tensor['sha256']:
            raise ValueError('source/scale hash mismatch')
    hipans = Path(os.environ.get('HIPANS_ROOT', str(ROOT / '../hipANS'))).resolve()
    commit = subprocess.check_output(['git', '-C', str(hipans), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = bool(subprocess.check_output(['git', '-C', str(hipans), 'status', '--porcelain'], text=True).strip())
    if dirty or commit != manifest['provenance']['hipANS_commit_built']:
        raise ValueError('baseline requires unchanged hipANS matching artifact')
    reference = base / source['tensors'][source['tensor_name']]['file']
    command = [str(ROOT / 'bin/hipans_lds_bench'), '--artifact', str(artifact), '--reference', str(reference),
               '--rows', str(source['shape'][0]), '--cols', str(source['shape'][1]), '--mode', a.mode]
    for name in ['warmup', 'iters', 'replicas', 'threads', 'grid_mult', 'streams']:
        command.extend(['--' + name.replace('_', '-'), str(getattr(a, name))])
    if a.skip_verify:
        command.append('--skip-verify')
    result = json.loads(subprocess.check_output(command, text=True))
    if result['provenance']['hipANS_commit_built'] != commit:
        raise ValueError('binary built from different hipANS revision')
    result.update(workload=source, artifact_sha256=manifest['artifact_sha256'], scale_bytes=manifest['scale_bytes'],
                  scale_consumed=False, ratio_total_with_scales=manifest['ratio_total_with_scales'])
    result['provenance'].update(hipANS_dirty=dirty, binary_sha256=sha(ROOT / 'bin/hipans_lds_bench'),
        implementation_sha256=sha(ROOT / 'bench/hipans_lds_bench.cpp'),
        hipANS_decode_header_sha256=sha(hipans / 'hipans/ans/GpuANSDecode.h'),
        benchmark_git_commit=subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip())
    build = Path(os.environ.get('HIPANS_BUILD_DIR', str(ROOT / 'build-hipans-mi300')))
    result['provenance']['hipANS_encoder_source_sha256'] = sha(hipans / 'hipans/ans/GpuANSEncode.cpp')
    result['provenance']['native_library_sha256'] = sha(build / 'lib/libgpu_ans.a')
    result['provenance']['native_CMakeCache_sha256'] = sha(build / 'CMakeCache.txt')
    flags = build / 'hipans/ans/CMakeFiles/gpu_ans.dir/flags.make'
    result['provenance']['native_build_flags'] = flags.read_text() if flags.exists() else 'see native_CMakeCache_sha256'
    version = Path('/opt/rocm/.info/version')
    result['provenance']['ROCm_version'] = version.read_text().strip() if version.exists() else subprocess.check_output(['hipconfig', '--version'], text=True).strip()
    print(json.dumps(result))

if __name__ == '__main__':
    main()
