# Standardized `ans_bench` CLI

`ans_bench` is the stable single-case interface consumed by
`entropyflow-lab`.  It runs the hipANS byte codec on one GPU, verifies a
byte-exact round trip, and writes exactly one JSON object to stdout.  Progress,
warnings, and errors go to stderr.

## Invocation

```bash
./ans_bench \
  --input symbols.bin \
  --meta meta.json \
  [--warmup 10] [--iters 50] [--peak-GBps 5300]
```

`--symbols` is an alias for `--input`, and the no-op `--json` flag is accepted
for compatibility with the current `entropyflow-lab` bridge.  `--prob-bits`
may be 9, 10, or 11 and defaults to 10.  The canonical interface should not
depend on that advanced option.

Build the executable on a ROCm compute node:

```bash
HIPANS_ROOT=/path/to/hipANS \
HIPANS_BUILD_DIR="$PWD/build-hipans-mi300" \
GPU_ARCH=gfx942 \
./scripts/build_ans_bench.sh
```

This build command does not modify the source checkout; subsequent codec
optimization edits are authorized as of 2026-10-11. The default `HIPANS_ROOT` is
`../hipANS`; the legacy `HIPANS_REPO` spelling is confined to older scripts.

## Input contract

`meta.json` contains three required positive integers:

```json
{
  "num_elems": 67108864,
  "bits_per_symbol": 8,
  "tile_size": 65536
}
```

The canonical binary representation is packed, so the input file contains
`ceil(num_elems * bits_per_symbol / 8)` bytes.  Nonfinal packed tile boundaries
must be byte aligned.  As a compatibility accommodation for
`entropyflow-lab`, 1--8-bit symbols may instead be stored as one `uint8` per
symbol; this is detected when the file contains exactly `num_elems` bytes and
the packed size differs.  The chosen layout is recorded in provenance.

Each tile becomes one independent hipANS archive in a pointer batch.  hipANS
operates on bytes, so a packed low-bit input is round-tripped as the exact
packed byte sequence; the codec does not reinterpret or alter symbols.

## Output schema

Example (formatted here only for readability):

```json
{
  "raw_bytes": 67108864,
  "compressed_bytes": 54241424,
  "table_metadata_bytes": 1053184,
  "ratio": 1.23722,
  "encode_ms": 2.31,
  "decode_ms": {"median": 0.41, "p10": 0.40, "p90": 0.43},
  "decode_GBps_over_compressed_read": 132.30,
  "decode_GBps_over_decoded_written": 163.68,
  "utilization_vs_peak": 0.03088,
  "provenance": {
    "hipANS_git_commit": "...",
    "hipANS_dirty": false,
    "build_flags": "...",
    "ROCm_version": "...",
    "GPU_name": "AMD Instinct MI300X OAM",
    "GPU_arch": "gfx942:...",
    "launch_config": {
      "grid": {"x": "ceil(occupancy_blocks/N)*4", "y": 1024},
      "block": {"decode": 128, "table": 512},
      "streams_per_workgroup": 128,
      "hip_streams": 1,
      "rans_block_bytes": 8192,
      "max_rans_blocks_per_tile": 8
    },
    "probability_bits": 10,
    "num_tiles": 1024,
    "input_layout": "packed_bits"
  }
}
```

Field meanings:

- `raw_bytes` is the exact number of bytes read from `symbols.bin` and written
  by the byte decoder.
- `compressed_bytes` is the sum of archive sizes returned by hipANS, including
  each archive's headers and tables but excluding unused allocation stride.
- `table_metadata_bytes` is the exact non-payload archive overhead: fixed
  header, 256 uint16 probabilities, 64-lane rANS states per 8,192-byte block,
  and aligned per-block descriptors.  It is a subset of `compressed_bytes`.
- `ratio` is `raw_bytes / compressed_bytes`; values above one save space.
- `encode_ms` is one HIP-event interval around GPU encode.  Encode is not part
  of decode throughput.
- `decode_ms` contains linearly interpolated median, p10, and p90 HIP-event
  times after discarded warmups.  One sample is one complete
  `ansDecodeBatchPointer` call, including table construction and decode.
- The two decode rates divide their named logical byte count by median decode
  time.  They are not hardware-counter HBM traffic.
- `utilization_vs_peak` is decoded-write GB/s divided by `--peak-GBps`, or
  `null` when no peak is supplied.  It is a fraction, not a percentage.
- `streams_per_workgroup` means independent rANS state streams, not HIP command
  queues: two 64-lane AMD waves in the fixed 128-thread decode workgroup.

`grid.x` is a formula because the current hipANS library computes the exact
value internally from runtime occupancy and does not expose it.  Counter/timeline
captures provide the realized dispatch dimensions.  The fixed algorithm is
`ceil(max resident blocks / num_tiles) * 4` in x and `num_tiles` in y.

The process exits nonzero and emits no result JSON if metadata validation,
encode, decode, or byte-exact verification fails.
