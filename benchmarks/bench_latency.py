#!/usr/bin/env python3
"""Measure TTFT, per-token ITL (from streaming deltas), and latency percentiles."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import statistics
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import aiohttp


def _pct(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return float("nan")
    k = (len(sorted_vals) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    if f == c:
        return sorted_vals[f]
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)


@dataclass
class StreamStats:
    ttft_ms: float
    itl_ms: list[float]
    total_ms: float
    num_tokens: int


def _parse_sse_line(line: str) -> dict[str, Any] | None:
    line = line.strip()
    if not line.startswith("data:"):
        return None
    payload = line[5:].strip()
    if payload == "[DONE]":
        return None
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        return None


async def one_streaming_chat(
    session: aiohttp.ClientSession,
    url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    timeout: aiohttp.ClientTimeout,
) -> StreamStats:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
    }
    headers = {}
    tok = os.environ.get("OPENAI_API_KEY")
    if tok:
        headers["Authorization"] = f"Bearer {tok}"

    t_req = time.perf_counter()
    ttft: float | None = None
    last_tok_t = t_req
    itl: list[float] = []
    n_tokens = 0

    buf = b""
    async with session.post(url, json=body, headers=headers, timeout=timeout) as resp:
        resp.raise_for_status()
        async for chunk in resp.content.iter_any():
            if not chunk:
                continue
            buf += chunk
            while b"\n" in buf:
                raw_line, buf = buf.split(b"\n", 1)
                try:
                    line = raw_line.decode("utf-8")
                except UnicodeDecodeError:
                    continue
                data = _parse_sse_line(line)
                if not data:
                    continue
                for ch in data.get("choices") or []:
                    delta = ch.get("delta") or {}
                    piece = delta.get("content") or ""
                    if piece:
                        now = time.perf_counter()
                        if ttft is None:
                            ttft = (now - t_req) * 1000.0
                        else:
                            itl.append((now - last_tok_t) * 1000.0)
                        last_tok_t = now
                        n_tokens += 1

    total_ms = (time.perf_counter() - t_req) * 1000.0
    if ttft is None:
        ttft = total_ms
    return StreamStats(ttft_ms=ttft, itl_ms=itl, total_ms=total_ms, num_tokens=n_tokens)


def _summ_percentiles(name: str, xs: list[float]) -> dict[str, float]:
    if not xs:
        return {
            f"{name}_p50_ms": float("nan"),
            f"{name}_p95_ms": float("nan"),
            f"{name}_p99_ms": float("nan"),
        }
    xs = sorted(xs)
    return {
        f"{name}_p50_ms": _pct(xs, 50),
        f"{name}_p95_ms": _pct(xs, 95),
        f"{name}_p99_ms": _pct(xs, 99),
    }


async def run_all(args: argparse.Namespace) -> dict[str, Any]:
    base = args.base_url.rstrip("/")
    url = f"{base}/v1/chat/completions"
    timeout = aiohttp.ClientTimeout(total=args.timeout_s)
    prompts = [args.prompt_template.format(i=i) for i in range(args.num_requests)]

    detail: list[dict[str, Any]] = []
    all_itl: list[float] = []
    async with aiohttp.ClientSession() as session:
        for p in prompts:
            st = await one_streaming_chat(
                session, url, args.model, p, args.max_tokens, timeout
            )
            all_itl.extend(st.itl_ms)
            detail.append(
                {
                    "ttft_ms": st.ttft_ms,
                    "itl_ms": st.itl_ms,
                    "mean_itl_ms": statistics.mean(st.itl_ms) if st.itl_ms else float("nan"),
                    "total_ms": st.total_ms,
                    "num_tokens": st.num_tokens,
                }
            )

    ttfts = sorted(d["ttft_ms"] for d in detail)
    out = {
        "meta": {
            "base_url": args.base_url,
            "model": args.model,
            "num_requests": args.num_requests,
            "max_tokens": args.max_tokens,
            "ts_utc": datetime.now(timezone.utc).isoformat(),
        },
        "ttft_ms_percentiles": _summ_percentiles("ttft", ttfts),
        "itl_ms_percentiles": _summ_percentiles("itl", all_itl),
        "per_request": detail,
    }
    per_req_means = [
        d["mean_itl_ms"] for d in detail if not math.isnan(d["mean_itl_ms"])
    ]
    if per_req_means:
        out["mean_itl_ms_across_requests"] = statistics.mean(per_req_means)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default=os.environ.get("BENCH_BASE_URL", "http://127.0.0.1:8000"))
    ap.add_argument("--model", default=os.environ.get("BENCH_MODEL", "google/gemma-7b-it"))
    ap.add_argument("--num-requests", type=int, default=20)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument(
        "--prompt-template",
        default="Write one sentence about inference optimization case {i}.",
    )
    ap.add_argument("--timeout-s", type=float, default=600.0)
    ap.add_argument(
        "--out",
        default=str(
            Path(__file__).resolve().parent.parent
            / "results"
            / "bench_latency.json"
        ),
    )
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    result = asyncio.run(run_all(args))
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["meta"], indent=2))
    print(json.dumps(result["ttft_ms_percentiles"], indent=2))
    print(json.dumps(result["itl_ms_percentiles"], indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
