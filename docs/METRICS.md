# Metric definitions

The existing `ans_bench` fields below describe the historical global-output
byte decoder. Current LDS-stage metrics are defined separately at the end of
this document; the existing JSON field meanings are preserved.

All byte rates use decimal GB/s (`1 GB = 10^9 bytes`).  All primary latency
statistics use HIP events recorded on the same non-blocking stream as the
codec.  Percentiles are linearly interpolated after sorting measured samples.

## Compression

Let `R` be the exact byte length of `symbols.bin`, `C_i` the archive size
returned by hipANS for tile `i`, and `C = sum(C_i)`.

```text
raw_bytes        = R
compressed_bytes = C
ratio            = R / C
```

Thus a ratio greater than one means compression saves storage.  Allocation
capacity, stride padding between archives, the caller's output buffer, and
hipANS scratch memory are excluded from `C`.

For a tile with `B_i = ceil(tile_bytes / 8192)` rANS blocks, the archive's
non-payload bytes are:

```text
32                                  fixed archive header
+ 256 * 2                           probability-frequency table
+ B_i * 64 * 4                      initial rANS states (AMD wave64)
+ round_up(B_i, 2) * 8              aligned block descriptors
```

`table_metadata_bytes` is the sum of this quantity over tiles.  This is the
actual hipANS on-disk overhead, not the paper's different design of one shared
layer codebook plus one 4-byte offset per tile.

## Decode latency and logical rates

The timed region is one `ansDecodeBatchPointer` call.  It includes device
pointer/capacity staging done by the API, per-tile decode-table construction,
and the decode dispatch.  It excludes H2D input copy, encode, D2H output copy,
and byte-exact comparison.

For median decode time `t` seconds:

```text
decode_GBps_over_compressed_read = C / t / 1e9
decode_GBps_over_decoded_written = R / t / 1e9
```

These are useful effective-throughput numerators, not claims that exactly `C`
and `R` bytes reached HBM.  Cache hits, cache-line amplification, table and
scratch traffic, replays, and write behavior make actual HBM read/write bytes
different.  rocprof-compute counters are the authority for measured HBM/L2/MALL
traffic.

The legacy harness also reports `(R + C) / t`.  That logical minimum traffic
proxy is retained in legacy CSV files but is deliberately not called bandwidth
utilization by the standardized interface.

## Achievable peak and utilization

`P` is the best valid BabelStream HIP Triad rate measured on the same physical
GPU, ROCm stack, and clock policy as the decoder.  The benchmark uses arrays
larger than the 256 MiB MALL/Infinity Cache so that the result represents HBM,
not cache.  `results/peak_bw.json` records the command, sizes, individual
kernel rates, selected `P`, and complete provenance.

The standardized utilization is:

```text
utilization_vs_peak = decode_GBps_over_decoded_written / P
```

It is emitted as a fraction; multiply by 100 for percent. The numerator is
logical decoded output bytes. This output-normalized HBM reference is a
harness convention, not a verified definition of a 30% result in the paper.
Using compressed read plus decoded write answers a different question.

## Relationship to Tan et al. (ISCA 2026)

Tan et al., *Approaching Shannon Bound with Lossless LLM Weight Compression*,
evaluate A100 (2 TB/s stated peak) and H200 and describe decoder throughput in
the TB/s range at the register/shared-memory level.  Their final design decodes
directly into shared memory inside a tiled GEMM and overlaps a producer decode
wave with tensor-core consumers.  The paper's principal microbenchmark figures
report fused GEMM TFLOP/s and speedups, not a standalone global-output decoder
counter table.

The supplied local PDF does not give an explicit 30% numerator/denominator.
The old output-normalized comparison is therefore not a verified paper metric.
It is not directly equivalent to either:

- hardware HBM bytes, because the NVIDIA fused path does not materialize all
  decoded weights in global memory; or
- the paper's end-to-end fused-GEMM throughput, which benefits from shared
  memory reuse and overlap absent from this standalone hipANS decoder.

The standardized number is therefore a controlled diagnostic comparison, not
a claim of reproducing the fused paper kernel.  Profiling must use measured HBM
counters and timeline evidence to explain the AMD gap.

Paper: <https://arxiv.org/abs/2606.15789>

## LDS-stage metrics

Offline compression, artifact loading/H2D and host correctness copies are
outside the runtime decode measurement. The implemented v1 LDS-table path
includes per-workgroup codebook staging in the complete kernel. The alternative
global-LUT path is not a resident-LDS steady-state measurement; persistent-table
experiments remain future work. See [LDS_CLI.md](LDS_CLI.md) for exact JSON fields.

Let `D` be actual consumer-format bytes produced in LDS, `N` the actual
weights produced (including repeated tile decoding), `C` the logical compressed
bytes consumed, and `t` the stated GPU interval in seconds:

```text
decoded_to_LDS_GBps       = D / t / 1e9
decoded_weights_per_sec  = N / t
compressed_input_GBps    = C / t / 1e9
```

Record unique weights separately from actual tile work. Record storage dtype,
ANS symbol dtype, LDS consumer dtype, scale metadata, and unpack/dequantization
work. For example, 4-bit weights expanded to BF16 produce 2 consumer bytes per
weight; their rate cannot be compared to packed-nibble output by byte rates alone.

The above logical rates remain distinct from profiler HBM reads/writes and LDS
transactions. `decoded_to_LDS_GBps / BabelStream_Triad_GBps` may be reported as
an explicitly named output-normalized HBM reference, but is not LDS hardware
utilization and may exceed 100% when decompression expands input. Actual LDS
utilization requires LDS counters and a stated, relevant LDS roofline/control
benchmark. The primary practical control is raw-weight-to-LDS at the same
tile geometry, consumer format and store pattern.

Discarded output must still execute real LDS stores, established through ISA
and LDS counters. Complete requirements are in [LDS_STAGE1.md](LDS_STAGE1.md).

The v1 `decoded_LDS_GBps` includes padded tile bytes; `valid_weight_GBps` and
`weights_per_second` exclude padding. `compressed_payload_GBps` counts only
payload, not descriptors, table staging or scales. All three are logical rates,
not physical memory traffic. The store-only control produces no real weights
and is not decoder throughput; its LDS byte-store rate is labeled as a control.
