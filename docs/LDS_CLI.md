# Real-weight LDS baseline v1

This independent path corresponds to strategic S0/S1. The historical
`ans_bench --input --meta` JSON contract is unchanged. No GEMM, dequantization,
FP4, kernel tuning, or end-to-end inference is claimed here.

## Workload and offline preparation

The selected upstream workload is `dsv4-flash-base`, immutable revision
`8855555deef230a27a21a8d6f294b7b7497759b6`, tensor
`layers.0.ffn.experts.10.w1.weight`: native safetensors `F8_E4M3`, shape
`[2048,4096]`, 8,388,608 bytes. Its `F32` `.scale` is `[16,32]`, 2,048 bytes.
This exact tensor appears in entropyflow-lab M2. See the
[delivery plan](project/DELIVERY_LDS_BASELINE.md) for the selection rationale.

Python uses this repository's system-site-packages `.venv`; `requests` is the
only additional extraction dependency. Build and CPU processing run inside
Slurm; the shared workspace requires no node-local copy. Both sibling source
trees and their existing caches remain read-only.

```bash
bash scripts/build_lds_baseline.sh  # devel allocation, gfx942 by default

# INDEX is an already downloaded index JSON for the pinned revision.
# TOKEN_FILE is optional; provision it securely outside git, never print it.
srun -p devel -N1 -n1 -c2 -t 00:05:00 .venv/bin/python \
  scripts/fetch_lds_weights.py --index "$INDEX" --token-file "$TOKEN_FILE" \
  --output inputs/lds/dsv4-flash-base-expert10-w1

srun -p devel -N1 -n1 -c2 -t 00:05:00 .venv/bin/python \
  scripts/prepare_lds_artifact.py \
  --source inputs/lds/dsv4-flash-base-expert10-w1/source.json
```

Extraction refuses a server that ignores HTTP Range, bounds header/tensor
responses, and downloads only the selected weight and companion scale.
Preparation verifies source hashes, performs an independent CPU roundtrip of
every stream, and refuses to replace an existing artifact. New extractions
should use a new output directory.

Local-only files: `weights.bin`, `scales.bin`, `source.json`, `weights.ans`,
`artifact.json`. Manifests identify source revision, shapes/dtypes, source and
artifact hashes, code hashes, stream/layout definitions, and exact storage
accounting. Weights and binary artifacts are intentionally ignored by git.

## Artifact v1

This is a **new local scalar byte-rANS reference layout**, not a hipANS archive
or an optimized decoder. Its normalization rule and state recurrence reuse
the reference design inspected in entropyflow-lab `codec/ref_ans.py`; tests
independently decode persisted bytes. hipANS provenance identifies the frozen
reference checkout, not the implementation performing this benchmark.

All integer fields are little-endian. The C++ packer/loader currently require
a little-endian host (the tested x86-64 cluster).

| Segment | Definition |
|---|---|
| Header, 64 bytes | `8s8I3Q`: magic `LDSANS1\0`, version=1, precision=12, rows, cols, tile_rows=64, tile_cols=128, lanes=64, tiles, raw_bytes, payload_bytes, reserved=0 |
| Frequencies, 1,024 bytes | 256 uint32 normalized frequencies, sum=4096; one table per source matrix, padding included |
| Stream descriptors | tile-major/lane-major `{uint64 offset; uint32 length; uint32 state;}`, 64 descriptors per tile |
| Payload | concatenated lane streams; byte-renormalized 32-bit rANS, lower state bound `2^23` |

Tiles traverse the original matrix in row-major order; each tile stores its
64×128 contents row-major, padding edges with zero. Lane `l` produces indices
`l + 64*i` for `i=0..127`. Tile-to-matrix correspondence is preserved, unlike
simply cutting a flat matrix into consecutive 8 KiB regions. This layout is
GEMM-tile-addressable, not yet an MFMA-swizzled consumer layout.

The decode LUT is expanded **once on the host before timing** to 4,096 uint32
entries: `symbol | (cdf<<8) | ((frequency-1)<<20)`. The minus-one encoding
also handles a single-symbol frequency of 4096. In `--table lds` each workgroup
stages this 16 KiB LUT; this staging **is included** in measured latency. The
tile adds 8 KiB LDS. `--table global` reads the prebuilt cached global LUT and
uses 8 KiB LDS; it is not a persistent/resident-LDS table measurement. LDS
does not persist between separate kernel launches, so the current interface
does not claim a resident-LDS steady-state result.

## Measurement

Inside an MI300X allocation:

```bash
.venv/bin/python scripts/run_lds_bench.py \
  --manifest inputs/lds/dsv4-flash-base-expert10-w1/artifact.json \
  --mode decode --table lds --warmup 10 --iters 50
```

Modes: `decode` (compressed→LDS), `raw` (same pre-tiled raw weights→LDS),
`lds-control` (constant byte pattern→LDS, no weight reads). The wrapper checks
weight/scale/artifact SHA-256, then emits one JSON object with workload and
binary/source provenance. The native executable additionally accepts
`--artifact PATH --reference PATH`; it checks structural validity but not
SHA-256. Prefer the integrity-checking wrapper for real measurements.

Validation is outside timing: decoder writes LDS, waits and synchronizes,
then reads **actual LDS** back into a global verification buffer. Every byte,
including padding, and each lane's final state/payload-consumption count are
checked. Raw mode uses the same store/layout and its own LDS readback. Control
mode is not real-weight correctness validation. `--skip-verify` is for profiler
runs only after an ordinary run passes.

Timed kernels use volatile inline-assembly `ds_write_b8`, a final LDS wait
and workgroup barrier; no global decoded output or checksum is produced.
`scripts/audit_lds_isa.py` checks **all four timed discard functions**, requiring
decoded `ds_write_b8` sites and prohibiting FLAT stores. This catches compiler
elimination and distinguishes table staging from decoded-output writes.

Key JSON metrics (decimal GB/s):

- `decode_ms.{median,p10,p90}`: HIP-event per-launch latency, linear percentiles.
- `decoded_LDS_GBps`: padded tile bytes / median seconds / `1e9`. For control
  mode this is a store-only rate, **not real-weight decoder throughput**.
- `valid_weight_GBps`, `weights_per_second`: unpadded real-weight production
  for decode/raw modes; no float conversion is included.
- `compressed_payload_GBps`: payload bytes / latency, logical only; excludes
  descriptors, table staging, scale traffic and physical transaction expansion.
- `artifact_bytes`: header + normalized frequencies + descriptors + payload;
  `ratio_total_with_scales` additionally accounts for uncompressed scales.
- `launch`: grid/block, streams, tile shape, declared LDS allocation.

Weights are produced once per launch; no GEMM reuse is modeled. The current
8 MiB workload repeatedly executes without cache flush and is explicitly
cache-resident/warm, not a >256 MiB inference-streaming test. Its GB/s must not
be labeled HBM or LDS peak utilization. Scale bytes are saved/accounted for but
are not fetched or interpreted by the timed kernel. FP8 codes remain native
checkpoint bits; MI300X FP8 GEMM FNUZ compatibility is a later-stage question.

## GPU campaign and returned results

```bash
sbatch -J lds-baseline -p mi3001x --gpus=1 -N1 -n1 -c4 -t 00:20:00 \
  --chdir="$PWD" --output=results/slurm-lds-baseline-%j.out \
  --error=results/slurm-lds-baseline-%j.err scripts/job_lds_baseline.sh
```

Each job writes `results/profiling/lds-JOBID/`: 15 GPU edge-case checks, four
unprofiled timing JSON files, ISA audit, rocprofv3 timeline, rocprof-compute
counters, availability TSV and failure logs. Counter permission failures are
recorded with a reduced-counter fallback, not hidden. Profiler-instrumented
latencies are **not** used as baseline timing. Use `squeue -j JOBID` while
pending/running, `sacct -j JOBID --format=JobID,State,ExitCode,Elapsed,NodeList`
afterwards, and read the explicit stdout/stderr plus the result directory.

CPU regression:

```bash
srun -p devel -N1 -n1 -c2 -t 00:05:00 .venv/bin/python \
  -m unittest discover -s tests -p 'test_lds_*.py' -v
```
