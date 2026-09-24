# Curated results

This directory contains selected, sanitized benchmark CSVs. Machine-local
absolute paths, private-reference metadata, and identifying hostnames are not
published. Raw logs and job-numbered result files remain on the HACC shared
filesystem and are excluded from Git.

- `raw_decode_qwen_mi210.csv`: full 108-case raw-decode matrix using real Qwen
  bf16 weight bytes as the DF input.
- `sepbase_mi210.csv`: final cold/hot separated decode plus vendor GEMM run for
  the three reference shapes.

Use `scripts/sanitize_results.py` to produce equivalent public copies from a
new raw CSV.
