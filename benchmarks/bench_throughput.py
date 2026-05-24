#!/usr/bin/env python3
"""Concurrency sweep: aggregate tokens/sec and req/s for chat completions (non-stream ok)."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import aiohttp


async def one_chat(
    session: aiohttp.ClientSession,
    url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    timeout: aiohttp.ClientTimeout,
) -> int:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": False,
    }
    headers = {}
    tok = os.environ.get("OPENAI_API_KEY")
    if tok:
        headers["Authorization"] = f"Bearer {tok}"
    async with session.post(url, json=body, headers=headers, timeout=timeout) as resp:
        resp.raise_for_status()
        data = await resp.json()
    usage = data.get("usage") or {}
    return int(usage.get("completion_tokens") or 0)


async def run_level(
    concurrency: int,
    num_requests: int,
    url: str,
    model: str,
    max_tokens: int,
    timeout: aiohttp.ClientTimeout,
) -> dict[str, Any]:
    sem = asyncio.Semaphore(concurrency)

    async def wrapped(
        session: aiohttp.ClientSession, i: int
    ) -> int:
        async with sem:
            prompt = f"Briefly describe GPU inference batch {i} in one sentence."
            return await one_chat(session, url, model, prompt, max_tokens, timeout)

    t0 = time.perf_counter()
    async with aiohttp.ClientSession() as session:
        tasks = [wrapped(session, i) for i in range(num_requests)]
        tokens = await asyncio.gather(*tasks)
    wall = time.perf_counter() - t0
    total_tokens = sum(tokens)
    return {
        "concurrency": concurrency,
        "num_requests": num_requests,
        "wall_s": wall,
        "total_completion_tokens": total_tokens,
        "tokens_per_s": total_tokens / wall if wall > 0 else float("nan"),
        "requests_per_s": num_requests / wall if wall > 0 else float("nan"),
    }


async def main_async(args: argparse.Namespace) -> dict[str, Any]:
    base = args.base_url.rstrip("/")
    url = f"{base}/v1/chat/completions"
    timeout = aiohttp.ClientTimeout(total=args.timeout_s)
    levels = [int(x) for x in args.concurrency_levels.split(",")]
    rows: list[dict[str, Any]] = []
    for c in levels:
        rows.append(
            await run_level(
                c, args.num_requests_per_level, url, args.model, args.max_tokens, timeout
            )
        )
    return {
        "meta": {
            "base_url": args.base_url,
            "model": args.model,
            "max_tokens": args.max_tokens,
            "ts_utc": datetime.now(timezone.utc).isoformat(),
        },
        "levels": rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default=os.environ.get("BENCH_BASE_URL", "http://127.0.0.1:8000"))
    ap.add_argument("--model", default=os.environ.get("BENCH_MODEL", "google/gemma-7b-it"))
    ap.add_argument("--concurrency-levels", default="1,4,16,32")
    ap.add_argument("--num-requests-per-level", type=int, default=64)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--timeout-s", type=float, default=3600.0)
    ap.add_argument(
        "--out",
        default=str(
            Path(__file__).resolve().parent.parent / "results" / "bench_throughput.json"
        ),
    )
    args = ap.parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result = asyncio.run(main_async(args))
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["levels"], indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
