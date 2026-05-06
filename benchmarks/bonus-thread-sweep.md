# Bonus - Thread Sweep (CPU) vs GPU

Model: `qwen2.5-1.5b-instruct-q4_k_m.gguf`  |  llama-bench WinGet build

## CPU-only thread sweep (ngl=0)

| threads | tg64 (tok/s) |
|--:|--:|
| 1 | 10.6 |
| 2 | 17.9 |
| 3 | 22.7 |
| 6 | 25.9 |
| 12 | 17.2 |

**Best CPU**: `-t 6` at 25.9 tok/s.

## GPU reference (Vulkan, ngl=99)

| backend | tg64 (tok/s) |
|---|--:|
| Vulkan ngl=99 | 82.5 |

GPU (Vulkan) is 3.2x faster than best CPU config. CPU decode is memory-bandwidth-bound: throughput peaks around physical core count then drops as hyperthreads fight over the same memory channels.
