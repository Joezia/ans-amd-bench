#!/usr/bin/env python3
"""Publish only the verified actual-hipANS result and regenerate the baseline report."""
import argparse
import json
import os
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def metrics(text):
    result, headings = {}, []
    for line in text.splitlines():
        if not line.startswith('│'):
            continue
        fields = [field.strip() for field in line.split('│')[1:-1]]
        if fields and fields[0] == 'Metric_ID':
            headings = fields
        elif fields and re.fullmatch(r'\d+\.\d+\.\d+', fields[0]):
            result[fields[0]] = dict(zip(headings, fields))
    return result

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', required=True, type=Path, help='new hipans-lds job directory')
    a = p.parse_args()
    summary = json.loads((a.run / 'sweep/summary.json').read_text())
    rocm_root = Path(subprocess.check_output(['hipconfig', '--rocmpath'], text=True).strip())
    rocm_version_file = rocm_root / '.info/version'
    hip_version = subprocess.check_output(['hipconfig', '--version'], text=True).strip()
    # Early records used hipconfig --version in the ROCm field. Preserve that
    # observed HIP value and explicitly correct only the metadata, not timings.
    if rocm_version_file.exists():
        for record in summary['records'] + summary['confirmation'] + [summary['raw_matched']]:
            provenance = record.get('provenance', {})
            if provenance.get('ROCm_version') == hip_version:
                provenance['HIP_version'] = hip_version
                provenance['ROCm_version'] = rocm_version_file.read_text().strip()
                provenance['ROCm_version_source'] = str(rocm_version_file)
                provenance['metadata_correction'] = 'original runner labelled HIP version as ROCm; timings unchanged'
    validation = json.loads((a.run / 'sweep/gpu-cases/validation.json').read_text())
    isa = json.loads((a.run / 'sweep/isa.json').read_text())
    assert summary['actual_hipANS'] and len(validation) == 70
    assert all(r['verified_actual_LDS'] for r in validation) and isa['discard_store_retention_pass']
    summary['gpu_validation'] = validation
    summary['isa'] = isa
    summary['job_id'] = a.run.name.rsplit('-', 1)[-1]
    availability = {}
    for label in ['default', 'selected']:
        path = a.run / ('profile-' + label) / 'availability.tsv'
        availability[label] = path.read_text() if path.exists() else 'not collected'
    summary['tool_availability'] = availability
    summary['profiling_analysis_status'] = 'raw counters/timeline collected where available; attribution pending'
    analysis_path = a.run / 'profile-default/analysis.txt'
    summary['official_counter_metrics'] = metrics(analysis_path.read_text()) if analysis_path.exists() else {}
    dispatch_path = a.run / 'profile-default/analysis-dispatch.txt'
    summary['counter_scope'] = {'configuration': 'default 128 threads, grid multiplier 4, single queue, replicas 256',
                              'dispatch': int(dispatch_path.read_text()) if dispatch_path.exists() else None,
                              'not_selected_multi_queue_configuration': True}
    target = Path(os.environ.get('LDS_RESULT_FILE', str(ROOT / 'results/lds_baseline.json')))
    temporary = target.with_suffix('.json.pending')
    temporary.write_text(json.dumps(summary, indent=2) + '\n')
    temporary.replace(target)
    launch = summary['selected_launch']
    baseline = summary['confirmation'][0]
    rate = summary['effective_LDS_GBps']
    rows = []
    for record in summary['records']:
        if record['stage'] in ['workload', 'selected-workload', 'confirmation', 'raw-matched'] and record['status'] == 'ok':
            rows.append(f"| {record['stage']} | {record['replicas']} | {record['effective_LDS_GBps']:.3f} | {record['decode_ms']['median']:.6f} |")
    report = f"""# {summary.get('machine', 'mi300x')} 当前 hipANS → LDS：S1 baseline

Slurm job {summary['job_id']}，真实 hipANS 原生 archive 与实际 LDS 验证通过。
错误 scalar baseline 已覆盖，不作为参考。

## 结果

测试范围内重复确认的最佳有效吞吐：**{rate:.3f} GB/s**。
threads={launch['threads']}，grid multiplier={launch['grid_multiplier']}，
streams={launch['streams']}，replicas=256（2 GiB 有效解码产出）。
三次独立确认：{summary['confirmation_GBps']} GB/s；
spread={summary['confirmation_spread_fraction']:.2%}。
最大两档工作量差异={summary['workload_boundary_delta_fraction']:.2%}；
5% 平台验收={summary['workload_plateau_within_5_percent']}。
平台或重复性不达标时，不能宣称最大资源供给已充分饱和。
匹配 raw→LDS：{summary['raw_matched']['effective_LDS_GBps']:.3f} GB/s，
不是解码吞吐，也不是 LDS 理论峰值。

| 测量 | replicas | 有效 GB/s | median ms |
|---|---|---|---|
""" + '\n'.join(rows) + f"""

## 实现与范围

- hipANS revision={baseline['provenance']['hipANS_commit_built']}，dirty=false。
- native ansEncodeBatchPointer；原有 ANSDecodeWarpFullBlock/decodeOneWarp，
  state recurrence、16-bit refill、ballot/prefix 与同步没有改写。
- 必要适配仅为 LDS Writer、验证与部署循环。默认 128 threads；
  部署 sweep 包含单 wave，编译器可消除单 wave barrier，未修改原有代码。
- 真实 DeepSeek-V4-Flash-Base 原生 FP8 专家矩阵 2048×4096，8 MiB；
  scale 保存但不消费，无 GEMM、无 4-bit。
- 独立地址副本检查资源饱和，不代表新增真实模型参数或冷模型 streaming。
- 分子为真实未填充 bytes × replicas；分母为多队列共同完成的 HIP-event 时间。
  table 准备、H2D、验证在计时外，kernel 内 LUT staging 与 LDS completion 在计时内。
- 70 项 GPU 边界验证通过，14 个 timed kernel 保留 ds_write_b8，无全局输出。
- launch 理论资源不等于实测 occupancy；测试范围的最大值不等于数学极限。
- Profiler 插桩/重放会改变调度，正式吞吐只使用未插桩记录。
  counters/timeline 可用项见 {target.name}；硬件归因尚待分析。

证据目录：{a.run}。所有部署案例、失败 stderr、hash 与 provenance 存于正式 JSON。
"""
    report_path = Path(os.environ.get('LDS_REPORT_FILE', str(ROOT / 'docs/LDS_BASELINE_REPORT.md')))
    counters = summary['official_counter_metrics']
    if counters:
        report += '\n## 单队列资源证据（不是上述最佳多队列配置）\n\n'
        report += '仅分析默认 128 threads、grid multiplier=4、streams=1 的末次 decoder dispatch；\n'
        report += '不将重放 counters 当作未插桩吞吐，也不把这些值外推到最佳 384-thread/4-queue 配置。\n\n'
        report += '| Metric ID | 指标 | Avg | 单位 | Percent of Peak |\n|---|---|---|---|---|\n'
        for identifier in ['2.1.8', '2.1.10', '2.1.16', '2.1.21', '6.2.8', '7.2.4', '17.6.6', '17.6.12']:
            row = counters.get(identifier, {})
            if row:
                report += f"| {identifier} | {row.get('Metric', '')} | {row.get('Avg', '')} | {row.get('Unit', '')} | {row.get('Percent of Peak', '—')} |\n"
        report += '\n工具报告的 L2–HBM interface bandwidth 保留原单位标签，不等同于独立实测 HBM pin 峰值。\n'
        report += 'Active CUs 是工具标记 deprecated 的指标；occupancy 及调度拒绝比例不是算法上限的证明。\n'
    report_path.write_text(report)
    # Concurrent machine jobs publish their own results/reports, never rewrite
    # the shared strategic/tactical status document with a read/modify/write race.
    print(json.dumps({'effective_LDS_GBps': rate, 'job_id': summary['job_id']}))

if __name__ == '__main__':
    main()
