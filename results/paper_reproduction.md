# Paper microbenchmark comparison

Tan et al. Figure 9 reports speedup of fused decode+GEMM over a naive
DietGPU-decode-plus-CUTLASS baseline. The AMD columns below are the
standalone global-output hipANS decoder and are intentionally not divided
into that speedup. This side-by-side table prevents unlike metrics from
being silently conflated.

| Model (paper matrix K×N) | Paper tile-aligned speedup | Paper final speedup | AMD ratio | AMD decoded GB/s | AMD utilization | Status |
|---|---:|---:|---:|---:|---:|---|
| Qwen-1.5B (8960×1536) | 3.66× | 4.41× | — | — | — | MISSING_INPUT |
| Mistral-7B (14336×4096) | 3.30× | 4.02× | — | — | — | MISSING_INPUT |
| Qwen-14B (13824×5120) | 4.19× | 5.08× | — | — | — | MISSING_INPUT |
| DeepSeek-67B (8192×8192) | 5.57× | 6.71× | — | — | — | MISSING_INPUT |
| Mixtral-176B (6144×6144) | 6.03× | 7.57× | — | — | — | MISSING_INPUT |
| Llama-405B (16384×16384) | 8.20× | 10.06× | — | — | — | MISSING_INPUT |

Paper values were transcribed from Figure 9 of arXiv:2606.15789. A row
is measured only when `<slug>.bin` exists in the input directory; missing
licensed model weights remain explicit rather than being replaced by
synthetic data.
