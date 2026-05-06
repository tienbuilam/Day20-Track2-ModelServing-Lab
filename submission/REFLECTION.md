# Reflection — Lab 20 (Personal Report)

> **Đây là báo cáo cá nhân.** Mỗi học viên chạy lab trên laptop của mình, với spec của mình. Số liệu của bạn không so sánh được với bạn cùng lớp — chỉ so sánh **before vs after trên chính máy bạn**. Grade rubric tính theo độ rõ ràng của setup + tuning của bạn, không phải tốc độ tuyệt đối.

---

**Họ Tên:** Bùi Lâm Tiến
**Cohort:** A20-K1
**Ngày submit:** 2026-05-06

---

## 1. Hardware spec (từ `00-setup/detect-hardware.py`)

- **OS:** Windows 11
- **CPU:** Intel(R) Core(TM) i7-10750H CPU @ 2.60GHz
- **Cores:** 6 physical / 12 logical
- **CPU extensions:** AVX2, FMA, BMI2
- **RAM:** 15.8 GB
- **Accelerator:** NVIDIA GeForce GTX 1650 Ti with Max-Q Design, 4096 MiB
- **llama.cpp backend đã chọn:** CUDA (llama-cpp-python) / Vulkan (llama-server.exe WinGet binary)
- **Recommended model tier:** Qwen2.5-1.5B-Instruct (Q4_K_M)

**Setup story**: Trên Windows, `llama-cpp-python` prebuilt CPU wheel cài qua `uv` do venv không có pip. WinGet binary `llama-server.exe` dùng Vulkan backend thay vì CUDA vì build không kèm CUDA DLL — tuy nhiên Vulkan vẫn cho 82.5 tok/s trên GTX 1650 Ti. Unicode encoding (ký tự `─`) cần `PYTHONUTF8=1` để benchmark.py chạy được.

---

## 2. Track 01 — Quickstart numbers (từ `benchmarks/01-quickstart-results.md`)

| Model | Load (ms) | TTFT P50/P95 (ms) | TPOT P50/P95 (ms) | E2E P50/P95/P99 (ms) | Decode rate (tok/s) |
|---|--:|--:|--:|--:|--:|
| qwen2.5-1.5b-instruct-q4_k_m.gguf | 2344 | 144 / 217 | 46.0 / 79.9 | 3018 / 4024 / 4549 | 21.7 |
| qwen2.5-1.5b-instruct-q2_k.gguf   | 418  | 168 / 193 | 32.4 / 35.8 | 2202 / 2430 / 2481 | 30.8 |

**Một quan sát**: Q2_K load nhanh hơn 5.6× (418ms vs 2344ms) và decode nhanh hơn 1.4× (30.8 vs 21.7 tok/s) vì file nhỏ hơn nên ít data phải đẩy qua memory bus. Tuy nhiên Q4_K_M cho output chất lượng cao hơn rõ rệt — đáng đánh đổi tốc độ khi RAM còn đủ.

---

## 3. Track 02 — llama-server load test

| Concurrency | Total RPS | TTFB P50 (ms) | E2E P95 (ms) | E2E P99 (ms) | Failures |
|--:|--:|--:|--:|--:|--:|
| 10 | 0.9 | 8900 | 13000 | 14000 | 0 |
| 50 | 0.9 | 16000 | 29000 | 31000 | 0 |

**KV-cache observation**: `llamacpp:kv_cache_usage_ratio` không được expose trong WinGet build này. Tuy nhiên từ `record-metrics.py`, `n_busy_slots_per_decode` đạt peak 3.80/4 = **~0.95** — tức 95% capacity của 4 parallel decode slots bị chiếm ở concurrency 10. Cùng lúc `requests_deferred` tăng lên 45 khi chạy u=50, xác nhận KV cache và compute slots đã bão hoà. Đây là bottleneck thực sự: server chỉ có 4 decode slots song song, nên 50 concurrent users phải queue.

---

## 4. Track 03 — Milestone integration

- **N16 (Cloud/IaC):** stub — pipeline chạy localhost only, không có k8s/GCP
- **N17 (Data pipeline):** stub — in-memory TOY_DOCS thay vì Airflow DAG
- **N18 (Lakehouse):** stub — không có Delta Lake/Iceberg, docs hardcoded trong Python
- **N19 (Vector + Feature Store):** stub — keyword overlap scoring thay vì vector index (Qdrant/Feast)

**Nơi tốn nhiều ms nhất** trong pipeline:

- embed: 0.0 ms (stub — keyword match, không có embedding model)
- retrieve: 0.0 ms (in-memory dict lookup)
- llama-server: ~4069 ms trung bình (query 1: 5471ms, query 2: 3118ms, query 3: 3680ms)

**Reflection**: Bottleneck hoàn toàn nằm ở llama-server (>99.9% thời gian). Khớp kỳ vọng — với stub retrieval thì không có I/O hay embedding cost. Nếu thay bằng vector store thật (Qdrant + sentence-transformer), embed và retrieve sẽ thêm ~50-200ms nhưng llama-server vẫn dominate ở mức giây.

---

## 5. Bonus — The single change that mattered most

**Change:** Dùng GPU offload (Vulkan, `--n-gpu-layers 99`) thay vì CPU-only

**Before vs after** (từ bonus thread sweep):

```
before: CPU best   t=6  → 25.9 tok/s
after:  Vulkan GPU ngl=99 → 82.5 tok/s
speedup: ~3.2×
```

**Tại sao nó work**: LLM decode là *memory-bandwidth-bound* — mỗi token cần đọc toàn bộ weight của model từ bộ nhớ. CPU của i7-10750H có ~45 GB/s memory bandwidth, trong khi GTX 1650 Ti có ~192 GB/s GDDR6 bandwidth — gấp 4.3×. Lý thuyết dự đoán ~4× speedup, thực tế đo được 3.2× (hơi thấp hơn vì Vulkan overhead và PCIe transfer khi load model). Thread sweep xác nhận điều này: trên CPU, thêm thread từ 6→12 (hyperthreading) lại *chậm hơn* (25.9 → 17.2 tok/s) vì các thread tranh nhau memory bandwidth trên cùng memory controller — thêm core không giúp gì khi bottleneck là băng thông, không phải số lượng compute unit.

---

## 6. (Optional) Điều ngạc nhiên nhất

WinGet `llama-server.exe` tự chọn Vulkan thay vì CUDA dù máy có GTX 1650 Ti — build này không kèm CUDA DLL. Điều bất ngờ là Vulkan cho 82.5 tok/s, gần bằng CUDA benchmark (benchmark.py với llama-cpp-python CUDA: 21.7 tok/s decode, nhưng đó là P50 với max_tokens=64 — không so sánh trực tiếp được vì llama-bench đo steady-state decode). Vulkan hoạt động tốt hơn kỳ vọng trên NVIDIA hardware.

---

## 7. Self-graded checklist

- [x] `hardware.json` đã commit
- [x] `models/active.json` đã commit (Qwen2.5-1.5B-Instruct Q4_K_M + Q2_K)
- [x] `benchmarks/01-quickstart-results.md` đã commit
- [x] `benchmarks/02-server-results.md` + `02-server-metrics.csv` đã commit
- [x] `benchmarks/bonus-thread-sweep.md` đã commit (thread sweep CPU vs GPU)
- [x] 7 screenshots trong `submission/screenshots/` (01–06 + 09-pipeline)
- [x] `make verify` exit 0 (chạy ngay trước khi push)
- [x] Repo trên GitHub ở chế độ **public**
- [x] Đã paste public repo URL vào VinUni LMS

---

**Quan trọng:** repo phải **public** đến khi điểm được công bố. Nếu private, grader không xem được → 0 điểm.
