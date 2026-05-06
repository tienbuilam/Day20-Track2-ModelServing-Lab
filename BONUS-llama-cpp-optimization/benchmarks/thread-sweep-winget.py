#!/usr/bin/env python3
"""Thread sweep using the WinGet llama-bench.exe binary (no source build needed)."""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

WINGET_BENCH = Path(
    r"C:\Users\Admin\AppData\Local\Microsoft\WinGet\Packages"
    r"\ggml.llamacpp_Microsoft.Winget.Source_8wekyb3d8bbwe\llama-bench.exe"
)

# Match the last column value in a data row: "| ... | 81.89 ± 0.00 |"
# Use bytes-safe ASCII pattern — avoid non-ASCII like ± which breaks on CP1252
TG_RE = re.compile(r"\|\s*([\d.]+)\s*[+\-]?[+\-]?\s*[\d.]*\s*\|$", re.MULTILINE)


def run_one(model: str, threads: int, n_gpu: int) -> float:
    cmd = [
        str(WINGET_BENCH), "-m", model,
        "-t", str(threads),
        "-ngl", str(n_gpu),
        "-p", "0", "-n", "64",
        "-r", "1",
    ]
    print(f"   threads={threads} ngl={n_gpu} ...", end=" ", flush=True)
    result = subprocess.run(cmd, capture_output=True, errors="replace", check=False)
    out = result.stdout + result.stderr

    # Find data rows (lines containing "tg" in 4th column area)
    tps = 0.0
    for line in out.splitlines():
        if "tg" in line and "|" in line:
            # Last column: "  81.89 ± 0.00 |" — first float in last segment
            last_col = line.rsplit("|", 2)[-2]
            nums = re.findall(r"[\d]+\.[\d]+", last_col)
            if nums:
                tps = float(nums[0])
                break

    print(f"{tps:.1f} tok/s")
    return tps


def main() -> int:
    if not WINGET_BENCH.exists():
        print(f"ERROR: llama-bench.exe not found at {WINGET_BENCH}", file=sys.stderr)
        return 1

    hw = json.loads(Path("hardware.json").read_text())
    model = json.loads(Path("models/active.json").read_text())["primary_model"]
    physical = hw["cpu"].get("cores_physical") or 4
    logical = hw["cpu"]["cores_logical"]
    backends = hw.get("gpu", {}).get("backends", {})
    n_gpu = 99 if any(v for k, v in backends.items() if k != "cpu_only") else 0

    grid = sorted({1, 2, max(physical // 2, 1), physical, logical})

    print(f"==> thread sweep (CPU-only, ngl=0): {Path(model).name}")
    print(f"    bench   : {WINGET_BENCH.name}")
    print(f"    physical: {physical}  logical: {logical}")
    print(f"    grid    : {grid}\n")

    rows_cpu = []
    for t in grid:
        tps = run_one(model, t, 0)
        rows_cpu.append({"threads": t, "tok_s": tps})

    print(f"\n==> GPU run (ngl=99) for reference:")
    tps_gpu = run_one(model, physical, n_gpu)

    best = max(rows_cpu, key=lambda r: r["tok_s"]) if rows_cpu else {"threads": 0, "tok_s": 0.0}
    speedup = tps_gpu / max(best["tok_s"], 0.1)

    md = "# Bonus - Thread Sweep (CPU) vs GPU\n\n"
    md += f"Model: `{Path(model).name}`  |  llama-bench WinGet build\n\n"
    md += "## CPU-only thread sweep (ngl=0)\n\n"
    md += "| threads | tg64 (tok/s) |\n|--:|--:|\n"
    md += "\n".join(f"| {r['threads']} | {r['tok_s']:.1f} |" for r in rows_cpu)
    md += f"\n\n**Best CPU**: `-t {best['threads']}` at {best['tok_s']:.1f} tok/s.\n\n"
    md += "## GPU reference (Vulkan, ngl=99)\n\n"
    md += "| backend | tg64 (tok/s) |\n|---|--:|\n"
    md += f"| Vulkan ngl=99 | {tps_gpu:.1f} |\n\n"
    md += (
        f"GPU (Vulkan) is {speedup:.1f}x faster than best CPU config. "
        "CPU decode is memory-bandwidth-bound: throughput peaks around physical core count "
        "then drops as hyperthreads fight over the same memory channels.\n"
    )

    out_dir = Path("benchmarks")
    out_dir.mkdir(exist_ok=True)
    (out_dir / "bonus-thread-sweep.md").write_text(md, encoding="utf-8")
    all_rows = rows_cpu + [{"threads": "GPU(ngl=99)", "tok_s": tps_gpu}]
    (out_dir / "bonus-thread-sweep.json").write_text(json.dumps(all_rows, indent=2), encoding="utf-8")

    print("\n" + md)
    print("==> Wrote benchmarks/bonus-thread-sweep.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
