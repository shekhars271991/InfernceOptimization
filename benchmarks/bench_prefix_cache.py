#!/usr/bin/env python3
"""
Approximate prefix-cache benefit by comparing TTFT on a long shared system prefix.

Run with --enable-prefix-caching on the server. First request pays full prefill;
subsequent identical-prefix requests should show lower TTFT if prefix caching hits.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import aiohttp


async def one_streaming_ttft(
    session: aiohttp.ClientSession,
    url: str,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    timeout: aiohttp.ClientTimeout,
) -> tuple[float, int]:
    body = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
    }
    headers = {}
    tok = os.environ.get("OPENAI_API_KEY")
    if tok:
        headers["Authorization"] = f"Bearer {tok}"

    def parse_line(line: str) -> dict[str, Any] | None:
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

    t_req = time.perf_counter()
    ttft: float | None = None
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
                data = parse_line(line)
                if not data:
                    continue
                for ch in data.get("choices") or []:
                    delta = ch.get("delta") or {}
                    piece = delta.get("content") or ""
                    if piece:
                        if ttft is None:
                            ttft = (time.perf_counter() - t_req) * 1000.0
                        n_tokens += 1
    total_ms = (time.perf_counter() - t_req) * 1000.0
    if ttft is None:
        ttft = total_ms
    return ttft, n_tokens


def build_messages(prefix: str, user_suffix: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": prefix},
        {"role": "user", "content": user_suffix},
    ]


async def main_async(args: argparse.Namespace) -> dict[str, Any]:
    base = args.base_url.rstrip("/")
    url = f"{base}/v1/chat/completions"
    timeout = aiohttp.ClientTimeout(total=args.timeout_s)
    prefix = (args.prefix_seed + "\n") * max(1, args.prefix_repeat)
    ttfts: list[float] = []

    async with aiohttp.ClientSession() as session:
        for i in range(args.num_rounds):
            msgs = build_messages(prefix, f"Question {i}: reply with one short sentence.")
            ttft, _n = await one_streaming_ttft(
                session, url, args.model, msgs, args.max_tokens, timeout
            )
            ttfts.append(ttft)

    first = ttfts[0]
    rest = ttfts[1:]
    speedup = first / statistics.mean(rest) if rest and statistics.mean(rest) > 0 else float("nan")
    return {
        "meta": {
            "base_url": args.base_url,
            "model": args.model,
            "prefix_chars": len(prefix),
            "num_rounds": args.num_rounds,
            "max_tokens": args.max_tokens,
            "ts_utc": datetime.now(timezone.utc).isoformat(),
            "note": "Higher first_vs_rest_ttft_speedup suggests prefix cache hits after round 0.",
        },
        "ttft_ms_per_round": ttfts,
        "ttft_first_ms": first,
        "ttft_mean_after_first_ms": statistics.mean(rest) if rest else float("nan"),
        "first_vs_rest_ttft_speedup": speedup,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default=os.environ.get("BENCH_BASE_URL", "http://127.0.0.1:8000"))
    ap.add_argument("--model", default=os.environ.get("BENCH_MODEL", "google/gemma-7b-it"))
    ap.add_argument("--prefix-seed", default="You are a helpful assistant. Context: " + ("x" * 256))
    ap.add_argument("--prefix-repeat", type=int, default=8, help="Repeat prefix block to lengthen cache key")
    ap.add_argument("--num-rounds", type=int, default=12)
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--timeout-s", type=float, default=600.0)
    ap.add_argument(
        "--out",
        default=str(
            Path(__file__).resolve().parent.parent / "results" / "bench_prefix_cache.json"
        ),
    )
    args = ap.parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result = asyncio.run(main_async(args))
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["meta"], indent=2))
    print(json.dumps({k: result[k] for k in result if k != "meta"}, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
