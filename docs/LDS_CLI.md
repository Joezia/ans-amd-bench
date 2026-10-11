# 当前 hipANS LDS CLI

该路径独立于历史全局输出 `ans_bench`，使用真实 hipANS 原生 archive。
没有替代 scalar codec、global-table 模式或纯 LDS control 模式。

## 构建与执行

```bash
HIPANS_ROOT=../hipANS bash scripts/build_hipans_lds.sh
sbatch -p mi3001x -N1 -n1 --gpus=1 -c4 -t 00:30:00 \
  -o results/slurm-hipans-lds-%j.out -e results/slurm-hipans-lds-%j.err \
  scripts/job_lds_baseline.sh
```

路径为共享目录，无需复制。作业内调用：

```bash
.venv/bin/python scripts/prepare_lds_artifact.py \
  --source inputs/lds/dsv4-flash-base-expert10-w1/source.json --replace
.venv/bin/python scripts/run_lds_bench.py \
  --manifest inputs/lds/dsv4-flash-base-expert10-w1/artifact.json \
  --replicas 256 --threads 128 --grid-mult 4 --streams 1 --warmup 10 --iters 50
```

prepare 用当前 hipANS GPU encoder 离线压缩，实际 LDS 恢复验证通过后才保存。
`--replace` 显式覆盖旧包与 manifest。运行入口检查权重、scale、archive hash
及 hipANS clean revision。输出一个 JSON object，schema=`hipans-lds-v1`。

## 参数与指标

- `--mode decode|raw`；raw 将预先排好的原始 tiles 直接写 LDS，不进行 ANS。
- `--threads 64|128|192|256|320|384|448`，每 wave 处理一个原有 8192-byte ANS block。
- `--grid-mult 1..16`，实际 grid 上限=resident workgroups/CU × CU 数 × multiplier。
- `--replicas 1..256`，每个副本独立设备地址，同一真实矩阵内容。
- `--streams 1..32`，互不重叠地分割总任务；共同 start/所有队列 join 的 HIP event 时间。
- `effective_LDS_GBps`：未填充源 bytes × replicas / median ms / 1e6。
- `padded_LDS_GBps` 单列填充后数值；`compressed_archive_GBps` 单列 archive bytes/time。
- `decode_ms` 包含 median/p10/p90。不包含离线 encode、H2D、table 构建及验证读回；
  包含每次 kernel 内 LDS table staging、原有 decode 同步和 LDS output completion。
- `launch` 记录 grid、wave/workgroup、LDS、VGPR、scratch、理论 resident workgroups。
  理论 occupancy 不是硬件实测利用率。
- `--skip-verify` 仅供已验证配置的 profiler 运行，不作为验收结果。

`scripts/sweep_hipans_lds.py` 输出每个案例（含失败 stderr）和重复确认。
只有成功验证的测量才覆盖 `results/lds_baseline.json`。该最大值限于测试范围；
最大两档工作量未形成平台时，不声称已充分饱和。

## 查看 Slurm 返回

```bash
squeue -j JOBID
sacct -j JOBID --format=JobID,State,ExitCode,Elapsed
tail -n 30 results/slurm-hipans-lds-JOBID.out
tail -n 30 results/slurm-hipans-lds-JOBID.err
```

排队时通常还没有 stdout；作业开始后持续写日志，结束后检查 ExitCode。
profiling 入口 `profiling/run_lds_baseline.sh artifact.json output-dir`，
可用 PROFILE_LDS_REPLICAS/THREADS/GRID_MULT/STREAMS 控制部署。
完整 counters 失败会记录原因并尝试精简 counters；插桩时间不作正式吞吐。

## MI350X 独立任务

```bash
sbatch -p mi3501x -N1 -n1 --gpus=1 -c4 -t 00:40:00 \
  -o results/slurm-hipans-lds-mi350x-%j.out \
  -e results/slurm-hipans-lds-mi350x-%j.err scripts/job_lds_mi350.sh
```

该入口在 compute node 构建 gfx950 原生库，使用独立 binary、build 与 ISA；
artifact 保存于源数据目录的 mi350x 子目录，源权重与 scale 共享只读，不复制。
结果为 `results/lds_baseline_mi350x.json`，报告为 `docs/LDS_BASELINE_REPORT_MI350X.md`。
MI300X 默认入口和结果文件保持不变。

通用运行器接受 HIPANS_LDS_BINARY、HIPANS_LDS_ISA、HIPANS_BUILD_DIR、
LDS_RESULT_FILE、LDS_REPORT_FILE 与 LDS_MACHINE 环境配置；上述 launcher 会设置。
硬件实测名称/架构写入每条 provenance，编译架构必须匹配当前 GPU。
不同架构分别选择部署参数、解释 counters；不能使用 MI300X 峰值归一化 MI350X。
