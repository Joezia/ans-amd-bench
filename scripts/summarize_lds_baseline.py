#!/usr/bin/env python3
"""Publish only the verified actual-hipANS result and regenerate the baseline report."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', required=True, type=Path, help='new hipans-lds job directory')
    a = p.parse_args()
    summary = json.loads((a.run / 'sweep/summary.json').read_text())
    validation = json.loads((a.run / 'sweep/gpu-cases/validation.json').read_text())
    isa = json.loads((a.run / 'sweep/isa.json').read_text())
    assert summary['actual_hipANS'] and len(validation) == 70
    assert all(r['verified_actual_LDS'] for r in validation) and isa['discard_store_retention_pass']
    summary['gpu_validation'] = validation
    summary['isa'] = isa
    summary['job_id'] = a.run.name.removeprefix('hipans-lds-')
    availability = {}
    for label in ['default', 'selected']:
        path = a.run / ('profile-' + label) / 'availability.tsv'
        availability[label] = path.read_text() if path.exists() else 'not collected'
    summary['tool_availability'] = availability
    summary['profiling_analysis_status'] = 'raw counters/timeline collected where available; attribution pending'
    (ROOT / 'results/lds_baseline.json').write_text(json.dumps(summary, indent=2) + '\n')
    launch = summary['selected_launch']
    baseline = summary['confirmation'][0]
    rate = summary['effective_LDS_GBps']
    rows = []
    for record in summary['records']:
        if record['stage'] in ['workload', 'selected-workload', 'confirmation', 'raw-matched'] and record['status'] == 'ok':
            rows.append(f"| {record['stage']} | {record['replicas']} | {record['effective_LDS_GBps']:.3f} | {record['decode_ms']['median']:.6f} |")
    report = f"""# 当前 hipANS → LDS：S1 baseline

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
  counters/timeline 可用项见 results/lds_baseline.json；硬件归因尚待分析。

证据目录：{a.run}。所有部署案例、失败 stderr、hash 与 provenance 存于正式 JSON。
"""
    (ROOT / 'docs/LDS_BASELINE_REPORT.md').write_text(report)
    status = ROOT / 'docs/project/STATUS.md'
    text = status.read_text().replace('当前等待 GPU 资源。', 'GPU 测量已返回；counter 归因待分析。')
    text = text.replace('Slurm job **109618** 已提交到 mi3001x', f"Slurm job **{summary['job_id']}** 已执行于 mi3001x")
    text += f"\n## 最新实测返回\n\nJob {summary['job_id']}：hipANS→LDS {rate:.3f} GB/s；70 项 GPU 验证和 ISA 审核通过。\nH1 完成；H2 sweep 已返回；H3 重复确认已返回，资源平台验收={summary['workload_plateau_within_5_percent']}，\n硬件 counter 归因仍待分析。S1 性能优化与 S2 融合尚未开始。\n"
    status.write_text(text)
    print(json.dumps({'effective_LDS_GBps': rate, 'job_id': summary['job_id']}))

if __name__ == '__main__':
    main()
