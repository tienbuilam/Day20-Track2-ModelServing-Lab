# 02 - llama-server Load Test Results

Server: llama-server.exe (WinGet build b9026), Qwen2.5-1.5B-Instruct Q4_K_M
Hardware: Intel i7-10750H, NVIDIA GTX 1650 Ti 4GB (CUDA), 15.8 GB RAM
Flags: --n-gpu-layers 99 --threads 6 --ctx-size 2048 --metrics --n-parallel 4

## Summary Table

| Concurrency | Total Reqs | RPS  | E2E P50 (ms) | E2E P95 (ms) | E2E P99 (ms) | Failures |
|--:|--:|--:|--:|--:|--:|--:|
| 10 | 54 | ~0.9 | 8900 | 13000 | 14000 | 0 (0%) |
| 50 | 55 | ~0.9 | 16000 | 29000 | 31000 | 0 (0%) |

## Breakdown by request type

### Concurrency 10
| Type | Reqs | P50 (ms) | P95 (ms) | P99 (ms) |
|---|--:|--:|--:|--:|
| short | 42 | 8900 | 10000 | 11000 |
| long-rag | 12 | 11000 | 14000 | 14000 |

### Concurrency 50
| Type | Reqs | P50 (ms) | P95 (ms) | P99 (ms) |
|---|--:|--:|--:|--:|
| short | 45 | 14000 | 27000 | 29000 |
| long-rag | 10 | 21000 | 31000 | 31000 |

## KV-cache metrics (from record-metrics.py during u=10 run)

Peak `llamacpp:requests_deferred` = 45 (queue depth at saturation with 4 parallel slots).
`llamacpp:kv_cache_usage_ratio` not exposed by this server build; KV cache pressure inferred
from `requests_deferred` spiking to 45 concurrent deferred requests at peak load.
`llamacpp:n_busy_slots_per_decode` peaked at ~3.79 (out of 4 parallel slots) -- near-full utilization.

## Observations

- n_parallel=4 allows true concurrent decoding; compared to Python server (serial),
  throughput improved ~5x (54 reqs vs 10 reqs in same 60s window).
- P50 latency increased from 8.9s (u=10) to 16s (u=50): queuing at the 4-slot limit.
- 0 failures across both runs -- server stable under load.
- At concurrency 50, requests_deferred peaked at 45 meaning most users were queued waiting
  for one of the 4 decode slots, confirming llama.cpp is compute-bound not I/O-bound.
