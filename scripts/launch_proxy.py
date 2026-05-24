#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Derived from vLLM: benchmarks/disagg_benchmarks/disagg_prefill_proxy_server.py
# Routes: /v1/completions and /v1/chat/completions → prefill (max_tokens=1) then decode stream.

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import time
import uuid
from urllib.parse import urlparse

import aiohttp
from quart import Quart, Response, make_response, request

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="vLLM P/D disaggregation proxy (Quart)")
    p.add_argument(
        "--timeout",
        type=float,
        default=6 * 60 * 60,
        help="Timeout for backend requests (seconds)",
    )
    p.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("DISAGG_PROXY_PORT", "8000")),
        help="HTTP port for this proxy (env DISAGG_PROXY_PORT overrides default 8000)",
    )
    p.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Bind address",
    )
    p.add_argument(
        "--prefill-url",
        type=str,
        default=os.environ.get("PREFILL_URL", "http://localhost:8100"),
        help="Prefill vLLM base URL",
    )
    p.add_argument(
        "--decode-url",
        type=str,
        default=os.environ.get("DECODE_URL", "http://localhost:8200"),
        help="Decode vLLM base URL",
    )
    p.add_argument(
        "--kv-host",
        type=str,
        default=os.environ.get("KV_HOST", "localhost"),
        help="Host/IP string for KV transfer addresses in X-Request-Id",
    )
    p.add_argument(
        "--prefill-kv-port",
        type=int,
        default=int(os.environ.get("PREFILL_KV_PORT", "14579")),
        help="Prefill KV port (must match launch_prefill.sh)",
    )
    p.add_argument(
        "--decode-kv-port",
        type=int,
        default=int(os.environ.get("DECODE_KV_PORT", "14580")),
        help="Decode KV port (must match launch_decode.sh)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    aiohttp_timeout = aiohttp.ClientTimeout(total=args.timeout)
    prefill_service_url = args.prefill_url
    decode_service_url = args.decode_url
    port = args.port

    prefill_kv_addr = f"{args.kv_host}:{args.prefill_kv_port}"
    decode_kv_addr = f"{args.kv_host}:{args.decode_kv_port}"
    logger.info(
        "Proxy KV addresses -> prefill: %s, decode: %s | backends %s %s",
        prefill_kv_addr,
        decode_kv_addr,
        prefill_service_url,
        decode_service_url,
    )

    app = Quart(__name__)
    app.config.update(
        AIOHTTP_TIMEOUT=aiohttp_timeout,
        PREFILL_SERVICE_URL=prefill_service_url,
        DECODE_SERVICE_URL=decode_service_url,
        PREFILL_KV_ADDR=prefill_kv_addr,
        DECODE_KV_ADDR=decode_kv_addr,
    )

    def _normalize_base_url(url: str) -> str:
        return url.rstrip("/")

    def _get_host_port(url: str) -> str:
        parsed = urlparse(url)
        host = parsed.hostname or "localhost"
        prt = parsed.port
        if prt is None:
            prt = 80 if parsed.scheme == "http" else 443
        return f"{host}:{prt}"

    prefill_base = _normalize_base_url(prefill_service_url)
    decode_base = _normalize_base_url(decode_service_url)
    kv_target = _get_host_port(decode_service_url)

    def _build_headers(request_id: str) -> dict[str, str]:
        headers: dict[str, str] = {
            "X-Request-Id": request_id,
            "X-KV-Target": kv_target,
        }
        api_key = os.environ.get("OPENAI_API_KEY")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    async def _run_prefill(
        request_path: str,
        payload: dict,
        headers: dict[str, str],
        request_id: str,
    ) -> None:
        url = f"{prefill_base}{request_path}"
        start_ts = time.perf_counter()
        logger.info("[prefill] start request_id=%s url=%s", request_id, url)
        try:
            async with aiohttp.ClientSession(timeout=aiohttp_timeout) as session:
                async with session.post(url=url, json=payload, headers=headers) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        raise RuntimeError(f"Prefill backend error {resp.status}: {error_text}")
                    await resp.read()
        except asyncio.TimeoutError as exc:
            raise RuntimeError(f"Prefill service timeout at {url}") from exc
        except aiohttp.ClientError as exc:
            raise RuntimeError(f"Prefill service unavailable at {url}") from exc
        logger.info(
            "[prefill] done request_id=%s elapsed=%.2fs",
            request_id,
            time.perf_counter() - start_ts,
        )

    async def _stream_decode(
        request_path: str,
        payload: dict,
        headers: dict[str, str],
        request_id: str,
    ):
        url = f"{decode_base}{request_path}"
        logger.info("[decode] start request_id=%s url=%s", request_id, url)
        try:
            async with aiohttp.ClientSession(timeout=aiohttp_timeout) as session:
                async with session.post(url=url, json=payload, headers=headers) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        logger.error("Decode backend error %s - %s", resp.status, error_text)
                        yield (
                            '{"error": "Decode backend error '
                            + str(resp.status)
                            + '"}'
                        ).encode()
                        return
                    async for chunk_bytes in resp.content.iter_chunked(1024):
                        yield chunk_bytes
        except asyncio.TimeoutError:
            logger.error("Decode service timeout at %s", url)
            yield b'{"error": "Decode service timeout"}'
        except aiohttp.ClientError as exc:
            logger.error("Decode service error at %s: %s", url, exc)
            yield b'{"error": "Decode service unavailable"}'
        logger.info("[decode] finished streaming request_id=%s", request_id)

    async def process_request() -> Response:
        try:
            original = await request.get_json()
            prefill_req = original.copy()
            prefill_req["max_tokens"] = 1
            if "max_completion_tokens" in prefill_req:
                prefill_req["max_completion_tokens"] = 1

            request_id = (
                f"___prefill_addr_{prefill_kv_addr}___decode_addr_"
                f"{decode_kv_addr}_{uuid.uuid4().hex}"
            )
            headers = _build_headers(request_id)
            await _run_prefill(request.path, prefill_req, headers, request_id)
            gen = _stream_decode(request.path, original, headers, request_id)
            response = await make_response(gen)
            response.timeout = None
            return response
        except Exception:
            logger.exception("Error processing request")
            return Response(
                response=b'{"error": "Internal server error"}',
                status=500,
                content_type="application/json",
            )

    @app.route("/v1/completions", methods=["POST"])
    async def handle_completions() -> Response:
        try:
            return await process_request()
        except asyncio.CancelledError:
            logger.warning("Request cancelled")
            return Response(
                response=b'{"error": "Request cancelled"}',
                status=503,
                content_type="application/json",
            )

    @app.route("/v1/chat/completions", methods=["POST"])
    async def handle_chat_completions() -> Response:
        return await handle_completions()

    app.run(host=args.host, port=port)


if __name__ == "__main__":
    main()
