# Metric definitions

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

It is emitted as a fraction; multiply by 100 for percent.  The numerator is
decoded output bytes because that is the convention behind the stated
"8--12% vs 30%+" comparison: for example, 650 GB/s decoded output against a
5,300 GB/s measured peak is 12.3%.  Using compressed read plus decoded write
would answer a different question and inflate this comparison.

## Relationship to Tan et al. (ISCA 2026)

Tan et al., *Approaching Shannon Bound with Lossless LLM Weight Compression*,
evaluate A100 (2 TB/s stated peak) and H200 and describe decoder throughput in
the TB/s range at the register/shared-memory level.  Their final design decodes
directly into shared memory inside a tiled GEMM and overlaps a producer decode
wave with tensor-core consumers.  The paper's principal microbenchmark figures
report fused GEMM TFLOP/s and speedups, not a standalone global-output decoder
counter table.

Accordingly, the 30%+ reference is mapped here only through the same effective
decoded-byte/peak convention.  It is not directly equivalent to either:

- hardware HBM bytes, because the NVIDIA fused path does not materialize all
  decoded weights in global memory; or
- the paper's end-to-end fused-GEMM throughput, which benefits from shared
  memory reuse and overlap absent from this standalone hipANS decoder.

The standardized number is therefore a controlled diagnostic comparison, not
a claim of reproducing the fused paper kernel.  Profiling must use measured HBM
counters and timeline evidence to explain the AMD gap.

Paper: <https://arxiv.org/abs/2606.15789>
