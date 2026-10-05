# Repository inventory

This inventory describes the repository at the start of the MI300X
standardization work.  It distinguishes measurements that already exist from
the interfaces and measurements added by the rest of this project.

## Repositories and build layout

- `ans-amd-bench` is the writable benchmark repository.  Its initial history
  contains the MI210 baseline (`67189ae`) and portable MI300 workflow
  (`6ffe245`).
- hipANS is a separate, read-only checkout selected by `HIPANS_ROOT` (legacy
  scripts call the same setting `HIPANS_REPO`) and defaults to `../hipANS`.
  At inventory time it is clean at commit
  `8cc8bc263bc943557682a5cc675a045cb9a86412`.
- hipANS is already built out of tree in `build-hipans-mi300/`.  The required
  static archives are `libgpu_ans.a`, `libhipans_utils.a`, and `libglog.a`.
  `scripts/job_build_full.sh` can reproduce those archives without writing to
  the hipANS checkout.  `harness/build.sh` links them into the standalone
  benchmark with `-O3 -DNDEBUG --offload-arch=gfx942` on MI300X.

## Existing benchmark paths

### Raw ANS decode (`harness/`)

`harness/hipans_raw_decode_bench.cpp` is the most complete reusable path.  It:

1. generates uniform or exponential bytes, or repeats/truncates a file;
2. copies the input to the GPU and calls `ansEncodeBatchPointer`;
3. calls `ansDecodeBatchPointer`, copies the result back, and performs a
   byte-exact comparison outside the timed region;
4. runs at least 10 warmups and 50 measurements; and
5. times one complete `ansDecodeBatchPointer` call with HIP events on the same
   non-blocking stream.

The measured call includes hipANS's per-call decode-table construction and the
decode kernel.  It excludes input generation, H2D, encode, D2H verification,
and the post-run output checksum.  The caller-owned compressed and decoded
buffers are disjoint.  The harness also rejects stack-memory overflow warnings
and marks a run contended when an idle-GPU check cannot establish 0% use.

`harness/run_all.sh` covers 108 combinations: three data distributions, six
sizes, probability precisions 9/10/11, and batch counts 1/8.
`harness/run_mi300_peak.sh` focuses on large MI300X cases.  Both are Slurm
launchers.

### Decode plus GEMM (`sepbase/`)

`sepbase/sepbase_bench.cpp` measures raw vendor GEMM, hipANS decode, and serial
decode-then-GEMM intervals for three bf16 shapes.  It tries hipBLASLt and falls
back to `rocblas_gemm_ex`, uses fp32 accumulation, and verifies both decoded
weights and GEMM results.  This is deliberately a separated baseline: decoded
weights are materialized in HBM and decode is neither fused with nor overlapped
with GEMM.

There is no aiter dependency or aiter call in this repository.  Existing GEMM
coverage is hipBLASLt/rocBLAS only.

## Existing results

- `results/published/raw_decode_qwen_mi210.csv` is a curated 108-case MI210
  matrix at hipANS `f3de741...`.  For 64 MiB per member and eight members it
  reports 131--190 GB/s of decoded output, depending on probability precision.
- `results/published/sepbase_mi210.csv` contains the separated GEMM baseline.
  Its 2 MiB decode takes about 0.113 ms and makes serial decode+GEMM about
  2.1--5.2 times the raw GEMM interval for the tested shapes.
- `harness/results/mi300-search-98541.csv` contains MI300X (`gfx942`) large-case
  exploratory results at hipANS `ec807b...`.  The valid 8 x 64 MiB and
  8 x 128 MiB cases reach about 667 and 686 GB/s of decoded output.  The 8 x
  256 MiB row is marked `INVALID` because the fixed 4 GiB
  `StackDeviceMemory` overflow warning was observed and must not be used as a
  performance result.
- Raw job logs and build logs under `harness/results/` retain environment and
  diagnostic details, but the published CSVs are the curated MI210 records.

No BabelStream result, hardware-counter capture, roofline analysis, or
rocprof timeline existed at inventory time.

## Existing byte and utilization definitions

The raw harness records:

- `uncompressed_bytes`: sum of decoded byte counts over batch members;
- `compressed_bytes`: sum of the archive sizes returned by hipANS;
- `compression_ratio`: `compressed_bytes / uncompressed_bytes`;
- `throughput_GBps_uncompressed`: `uncompressed_bytes / median_decode_seconds`;
- `throughput_GBps_compressed`: `compressed_bytes / median_decode_seconds`;
- `throughput_GBps_total_traffic`: the sum of those two logical rates.

All GB/s values use decimal GB.  The last metric is a useful logical minimum
traffic proxy (one archive read plus one decoded-output write), but it is not a
measured HBM rate: it omits lookup-table traffic, archive-header/table reads,
cache-line amplification, write allocation, temporary data, and replays.

The existing code does **not** compute bandwidth utilization against either
MI300X's specification or a measured STREAM/BabelStream peak.  Its
`rocm-smi` "GPU use" sample is only an idle/contended-run check, not memory
bandwidth utilization.  Therefore the previously quoted 8--12% is an external
comparison, not a field currently emitted by this harness.

## hipANS encode/decode entry points and launch behavior

The reusable byte-codec API is declared in
`hipans/ans/GpuANSCodec.h`.  `ansEncodeBatchPointer` and
`ansDecodeBatchPointer` accept host arrays of device pointers and execute on a
caller-provided HIP stream.  The archive is self-describing and contains a
32-byte fixed header, a 256-entry uint16 probability table, per-rANS-block
wave states, aligned per-block descriptors, and the encoded words.

The current codec has these fixed internal parameters:

- one rANS block represents 8,192 decoded bytes;
- the decode-table kernel launches 512 threads for each independent archive;
- the main decode kernel launches 128 threads per workgroup;
- the x-grid is four times the occupancy-derived resident-block count, divided
  across batch members; and
- one HIP API call uses one caller stream.  There is no exposed "streams per
  workgroup" control in this fork.

Consequently, the existing public API permits sweeps over input size, number of
independent archives, and concurrent host/HIP streams.  Grid size, workgroup
size, and streams per workgroup are compile-time/internal choices and cannot be
changed through the current API without building an explicitly marked launch
variant.  Profiling must record this limitation rather than silently claiming
those controls were swept.

## Reuse plan

The standardized CLI reuses the validated pointer-batch encode/decode sequence,
HIP-event timing, byte-exact check, provenance collection, and out-of-tree link
model from `harness/`.  The legacy matrix and separated-GEMM programs remain
available for regression comparison; they are not rewritten.
