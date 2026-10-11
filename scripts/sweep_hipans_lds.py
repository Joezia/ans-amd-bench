#!/usr/bin/env python3
"""Measure current hipANS deployment envelope and confirm workload saturation."""
import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
from prepare_lds_artifact import ROOT

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, str(ROOT / 'tests/run_lds_gpu_cases.py'),
                    str(a.output / 'gpu-cases')], check=True)
    isa = subprocess.check_output([sys.executable, str(ROOT / 'scripts/audit_lds_isa.py'),
        str(ROOT / 'build-hipans-lds/hipans_lds_bench-hip-amdgcn-amd-amdhsa-gfx942.s')], text=True)
    (a.output / 'isa.json').write_text(isa)
    records = []
    def run(stage, replicas=256, threads=128, grid=4, streams=1, mode='decode', iters=30):
        cmd = [sys.executable, str(ROOT / 'scripts/run_lds_bench.py'), '--manifest', str(a.manifest),
               '--replicas', str(replicas), '--threads', str(threads), '--grid-mult', str(grid),
               '--streams', str(streams), '--mode', mode, '--warmup', '5', '--iters', str(iters)]
        proc = subprocess.run(cmd, text=True, capture_output=True)
        result = ({'stage': stage, 'status': 'failed', 'command': cmd, 'stderr': proc.stderr}
                  if proc.returncode else {**json.loads(proc.stdout), 'stage': stage, 'status': 'ok'})
        (a.output / f'{len(records):03d}-{stage}.json').write_text(json.dumps(result, indent=2) + '\n')
        records.append(result)
        print(f"{len(records)} {stage} R={replicas} T={threads} G={grid} S={streams}: "
              f"{result.get('effective_LDS_GBps', result.get('status'))}", flush=True)
        return result
    workload = [run('workload', replicas=r) for r in [1, 4, 16, 64, 128, 256]]
    if any(r['status'] != 'ok' for r in workload):
        raise RuntimeError('workload/correctness sweep failed')
    deployment = [run('deployment', threads=t, grid=g) for t in [64, 128, 192, 256, 320, 384, 448] for g in [1, 2, 4, 8]]
    good = [r for r in deployment if r['status'] == 'ok']
    if not good:
        raise RuntimeError('all deployments failed')
    top = sorted(good, key=lambda r: r['effective_LDS_GBps'], reverse=True)[:3]
    concurrent = [run('concurrency', threads=r['launch']['threads'], grid=r['launch']['grid_multiplier'], streams=s)
                  for r in top for s in [2, 4, 8, 16]]
    best = max(good + [r for r in concurrent if r['status'] == 'ok'], key=lambda r: r['effective_LDS_GBps'])
    params = dict(threads=best['launch']['threads'], grid=best['launch']['grid_multiplier'], streams=best['launch']['streams'])
    boundary = [run('selected-workload', replicas=r, **params) for r in [64, 128, 256]]
    confirmations = [run('confirmation', **params, iters=50) for _ in range(3)]
    raw = run('raw-matched', mode='raw', **params, iters=50)
    if any(r['status'] != 'ok' for r in boundary + confirmations + [raw]):
        raise RuntimeError('selected confirmation failed')
    values = [r['effective_LDS_GBps'] for r in confirmations]
    delta = abs(boundary[-1]['effective_LDS_GBps'] / boundary[-2]['effective_LDS_GBps'] - 1)
    summary = {'schema': 'hipans-s1-baseline-v1', 'status': 'measured', 'actual_hipANS': True,
        'selected_launch': best['launch'], 'effective_LDS_GBps': statistics.median(values),
        'confirmation_GBps': values, 'confirmation_spread_fraction': (max(values)-min(values))/statistics.median(values),
        'workload_boundary_delta_fraction': delta, 'workload_plateau_within_5_percent': delta <= .05,
        'claim': 'confirmed best in tested envelope; not mathematical ceiling',
        'raw_matched': raw, 'confirmation': confirmations, 'records': records}
    (a.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    target = ROOT / 'results/lds_baseline.json'
    temporary = target.with_suffix('.json.pending')
    temporary.write_text(json.dumps(summary, indent=2) + '\n')
    temporary.replace(target)
    for label, selected in [('default', {}), ('selected', params)]:
        env = dict(os.environ)
        env.update(PROFILE_LDS_THREADS=str(selected.get('threads', 128)),
                   PROFILE_LDS_GRID_MULT=str(selected.get('grid', 4)),
                   PROFILE_LDS_STREAMS=str(selected.get('streams', 1)))
        profile = a.output.parent / ('profile-' + label)
        proc = subprocess.run(['bash', str(ROOT / 'profiling/run_lds_baseline.sh'),
                               str(a.manifest), str(profile)], env=env, capture_output=True, text=True)
        (a.output / ('profile-' + label + '.log')).write_text(proc.stdout + proc.stderr)
        if label == 'default':
            analysis = subprocess.run(['bash', str(ROOT / 'scripts/analyze_lds_profile.sh'),
                                       str(profile)], capture_output=True, text=True)
            (a.output / 'analysis.log').write_text(analysis.stdout + analysis.stderr)
    subprocess.run([sys.executable, str(ROOT / 'scripts/summarize_lds_baseline.py'),
                    '--run', str(a.output.parent)], check=True)

if __name__ == '__main__':
    main()
