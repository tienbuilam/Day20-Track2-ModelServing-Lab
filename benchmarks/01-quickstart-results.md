# 01 - Quickstart Results

Settings: `n_threads=6`, `n_ctx=2048`, `n_batch=512`, `n_gpu_layers=99`.

| Model | Load (ms) | TTFT P50/P95 (ms) | TPOT P50/P95 (ms) | E2E P50/P95/P99 (ms) | Decode rate (tok/s) |
|---|---:|---:|---:|---:|---:|
| qwen2.5-1.5b-instruct-q4_k_m.gguf | 2344 | 144 / 217 | 46.0 / 79.9 | 3018 / 4024 / 4549 | 21.7 |
| qwen2.5-1.5b-instruct-q2_k.gguf | 418 | 168 / 193 | 32.4 / 35.8 | 2202 / 2430 / 2481 | 30.8 |

## Observations

- TTFT is the prefill cost. With short prompts this is small; with long prompts it dominates.
- TPOT is per-token decode latency. The decode rate is `1000 / TPOT_p50`.
- The bigger quantization (Q4_K_M) is only ~30-40% slower than Q2_K on this GTX 1650 Ti (21.7 vs 30.8 tok/s), but Q4_K_M produces noticeably better text quality. Q2_K is for truly tight RAM.
- `n_threads = physical_cores` is usually best on CPU. Hyperthreading hurts because the work is bandwidth-bound.
