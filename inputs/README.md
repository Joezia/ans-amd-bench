# Local benchmark inputs

This directory is intentionally excluded from Git because it contains generated
binary data and model-derived weight bytes.

The primary real-weight input used by both benchmarks is:

```text
qwen_bf16_256MiB.bin
```

`harness/run_all.sh` accepts any binary through `FILE_INPUT`; `sepbase/run_all.sh`
accepts a bf16 weight byte stream through `WEIGHT_FILE`. Keep these files on the
cluster's shared `/data` storage and pass an explicit path when necessary.

Synthetic inputs can be regenerated with `scripts/gen_inputs.py`. Model-derived
files can be generated with `scripts/dump_weights.py`, subject to the source
model's license. Neither the source model nor extracted weights should be
committed without a separate license review.
