# H3 T3 MI300X follow-up

The MI300X probe is now complete. Slurm job 110400 ran on `tn1-003-001`
(`gfx942`) and exited 0. The earlier job 110302 failed before probing because
the launcher required a nonexistent MI300X-specific copy of the GLM R2
manifest. The launcher was changed so tool discovery is independent of the
optional counter replay; the fix is in commit `11b6ab6`.

`rocprofv3 --list-avail`, `rocprof-compute --list-metrics gfx942`, and
`--list-blocks gfx942` all succeeded. ROCm reported rocprofiler SDK 1.3.5 and
rocprofiler-compute 3.8.0, with no permission failures. The MI300X PC-sampling
catalog lists `host_trap` (time interval 1–1,048,576) and `stochastic` (cycle
interval 256–1,048,576, power-of-two); availability was checked, but no PC
sampling capture was run.

The `gfx942` catalog includes L2 `TCC_HIT`/`TCC_MISS`, HBM-directed
`TCC_EA0_RDREQ_DRAM`, L2-Fabric read bandwidth, and HBM/remote traffic metrics.
No direct MALL/Infinity Cache/L3 hit counter was identified. This agrees with
the existing MI300X R3 counter capture (job 109852), which remains the source
of measured T3 counters. Job 110400 skipped a second counter replay because
the MI300X-local manifest is absent; this does not affect the tool-availability
probe.

The captured raw output is git-ignored under
`results/profiling/h3-t3-mi300x-110400/`; the curated fields are recorded in
[`h3_t3_mi300x_followup.json`](../results/h3_t3_mi300x_followup.json). This
follow-up supersedes the pending-job status for MI300X in the earlier T3
report without replacing that report.
