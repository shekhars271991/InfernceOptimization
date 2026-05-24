#!/usr/bin/env python3
"""
Run latency + throughput benchmarks against two OpenAI-compatible base URLs
(e.g. FP16 baseline vs AWQ) and write a side-by-side JSON summary.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def run_py(script: Path, extra: list[str]) -> None:
    cmd = [sys.executable, str(script), *extra]
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--fp16-url",
        default=os.environ.get("BENCH_FP16_URL", "http://127.0.0.1:8000"),
    )
    ap.add_argument(
        "--awq-url",
        default=os.environ.get("BENCH_AWQ_URL", "http://127.0.0.1:8001"),
    )
    ap.add_argument("--model", default=os.environ.get("BENCH_MODEL", "google/gemma-7b-it"))
    ap.add_argument(
        "--out",
        default=str(
            Path(__file__).resolve().parent.parent / "results" / "bench_quant_compare.json"
        ),
    )
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    bench_lat = root / "benchmarks" / "bench_latency.py"
    bench_thr = root / "benchmarks" / "bench_throughput.py"
    results_dir = root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    fp16_lat = results_dir / "bench_latency_fp16.json"
    awq_lat = results_dir / "bench_latency_awq.json"
    fp16_thr = results_dir / "bench_throughput_fp16.json"
    awq_thr = results_dir / "bench_throughput_awq.json"

    common = ["--model", args.model]
    run_py(
        bench_lat,
        ["--base-url", args.fp16_url, *common, "--out", str(fp16_lat)],
    )
    run_py(
        bench_lat,
        ["--base-url", args.awq_url, *common, "--out", str(awq_lat)],
    )
    run_py(
        bench_thr,
        ["--base-url", args.fp16_url, *common, "--out", str(fp16_thr)],
    )
    run_py(
        bench_thr,
        ["--base-url", args.awq_url, *common, "--out", str(awq_thr)],
    )

    summary = {
        "meta": {
            "fp16_url": args.fp16_url,
            "awq_url": args.awq_url,
            "model": args.model,
            "ts_utc": datetime.now(timezone.utc).isoformat(),
        },
        "fp16": {
            "latency": json.loads(fp16_lat.read_text(encoding="utf-8")),
            "throughput": json.loads(fp16_thr.read_text(encoding="utf-8")),
        },
        "awq": {
            "latency": json.loads(awq_lat.read_text(encoding="utf-8")),
            "throughput": json.loads(awq_thr.read_text(encoding="utf-8")),
        },
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote combined summary to {out_path}")


if __name__ == "__main__":
    main()
