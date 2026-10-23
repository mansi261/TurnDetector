"""Measure cold and warm API behavior under controlled concurrency."""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
from importlib.metadata import version
from io import BytesIO
import json
import os
from pathlib import Path
import platform
import statistics
import time
import wave

import httpx
import numpy as np
import psutil
import soundfile as sf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--requests", type=int, default=50)
    parser.add_argument("--concurrency", default="1,4,8")
    parser.add_argument("--workers", type=int, default=1)
    return parser.parse_args()


async def send(client: httpx.AsyncClient, url: str, payload: bytes):
    started = time.perf_counter()
    try:
        response = await client.post(
            f"{url}/predict",
            files={"audio": ("sample.wav", payload, "audio/wav")},
        )
        elapsed_ms = (time.perf_counter() - started) * 1000
        try:
            body = response.json()
        except json.JSONDecodeError:
            body = response.text
        return elapsed_ms, response.status_code, body
    except httpx.HTTPError as error:
        return (time.perf_counter() - started) * 1000, 0, str(error)


def distribution(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    return {
        "min": min(values),
        "p50": statistics.median(values),
        "p95": float(np.percentile(values, 95)),
        "max": max(values),
    }


async def scenario(client, url, payload, request_count, concurrency):
    semaphore = asyncio.Semaphore(concurrency)

    async def limited():
        async with semaphore:
            return await send(client, url, payload)

    started = time.perf_counter()
    results = await asyncio.gather(*(limited() for _ in range(request_count)))
    elapsed = time.perf_counter() - started
    successful = [item for item in results if item[1] == 200]
    client_latencies = [item[0] for item in successful]
    model_latencies = [
        float(item[2]["timing_ms"]["total"])
        for item in successful
        if isinstance(item[2], dict) and "timing_ms" in item[2]
    ]
    errors = request_count - len(successful)
    return {
        "concurrency": concurrency,
        "requests": request_count,
        "successful_requests": len(successful),
        "errors": errors,
        "error_rate": errors / request_count,
        "elapsed_seconds": elapsed,
        "throughput_completed_per_second": request_count / elapsed,
        "throughput_successful_per_second": len(successful) / elapsed,
        "client_latency_ms": distribution(client_latencies),
        "server_model_latency_ms": distribution(model_latencies),
    }


def invalid_rate_wav() -> bytes:
    output = BytesIO()
    with wave.open(output, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(8000)
        audio.writeframes(b"\x00\x00" * 800)
    return output.getvalue()


async def run(args: argparse.Namespace) -> dict:
    payload = args.audio.read_bytes()
    audio_info = sf.info(BytesIO(payload))
    concurrency_values = [int(value) for value in args.concurrency.split(",")]
    limits = httpx.Limits(max_connections=max(concurrency_values))
    async with httpx.AsyncClient(timeout=120, limits=limits) as client:
        health_response = await client.get(f"{args.url}/health")
        ready_response = await client.get(f"{args.url}/ready")
        health_response.raise_for_status()
        ready_response.raise_for_status()
        health = health_response.json()
        readiness = ready_response.json()
        cold_latency, cold_status, cold_body = await send(client, args.url, payload)
        for _ in range(5):
            await send(client, args.url, payload)
        scenarios = [
            await scenario(client, args.url, payload, args.requests, concurrency)
            for concurrency in concurrency_values
        ]
        invalid = {
            "empty_audio": await send(client, args.url, b""),
            "wrong_sample_rate": await send(client, args.url, invalid_rate_wav()),
            "invalid_encoding": await send(client, args.url, b"not an audio file"),
            "over_size_limit": await send(client, args.url, b"0" * (25 * 1024 * 1024 + 1)),
        }
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "method": "local closed-loop HTTP benchmark; sequential cold request, five warmups, then fixed-count concurrency scenarios",
        "url": args.url,
        "request": {
            "audio": args.audio.resolve().as_posix(),
            "bytes": len(payload),
            "format": audio_info.format,
            "sample_rate": audio_info.samplerate,
            "channels": audio_info.channels,
            "duration_seconds": audio_info.duration,
        },
        "load": {
            "requests_per_scenario": args.requests,
            "concurrency": concurrency_values,
            "workers": args.workers,
        },
        "hardware": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "logical_cpu_count": os.cpu_count(),
            "memory_bytes": psutil.virtual_memory().total,
        },
        "runtime": {
            package: version(package)
            for package in ("fastapi", "uvicorn", "onnxruntime", "numpy", "transformers")
        },
        "service": {"health": health, "ready": readiness},
        "cold_inference": {
            "client_latency_ms": cold_latency,
            "status": cold_status,
            "response": cold_body,
        },
        "warm": scenarios,
        "invalid_requests": {
            name: {"client_latency_ms": result[0], "status": result[1], "response": result[2]}
            for name, result in invalid.items()
        },
    }


def main() -> None:
    args = parse_args()
    report = asyncio.run(run(args))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
