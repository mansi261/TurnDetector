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
    parser.add_argument("--streaming-replays", type=int, default=0)
    parser.add_argument("--streaming-chunk-ms", type=int, default=350)
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


async def send_stream(
    client: httpx.AsyncClient,
    url: str,
    session_id: str,
    payload: bytes,
):
    started = time.perf_counter()
    try:
        response = await client.post(
            f"{url}/stream/{session_id}/predict",
            content=payload,
            headers={"Content-Type": "application/octet-stream"},
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
        "p99": float(np.percentile(values, 99)),
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


def pcm_wav(samples: np.ndarray, sample_rate: int) -> bytes:
    output = BytesIO()
    with wave.open(output, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(sample_rate)
        audio.writeframes(samples.astype("<i2", copy=False).tobytes())
    return output.getvalue()


def replay_summary(records: list[tuple[float, int, dict]]) -> dict:
    return {
        "requests": len(records),
        "request_bytes_total": sum(item[1] for item in records),
        "request_bytes": distribution([float(item[1]) for item in records]),
        "client_latency_ms": distribution([item[0] for item in records]),
        "preprocess_ms": distribution(
            [float(item[2]["timing_ms"]["preprocess"]) for item in records]
        ),
        "inference_ms": distribution(
            [float(item[2]["timing_ms"]["inference"]) for item in records]
        ),
        "model_total_ms": distribution(
            [float(item[2]["timing_ms"]["total"]) for item in records]
        ),
    }


async def streaming_comparison(
    client: httpx.AsyncClient,
    url: str,
    audio_bytes: bytes,
    *,
    sample_rate: int,
    window_seconds: float,
    chunk_ms: int,
    replays: int,
) -> dict:
    waveform, decoded_rate = sf.read(
        BytesIO(audio_bytes), dtype="float32", always_2d=True
    )
    if decoded_rate != sample_rate:
        raise ValueError(f"expected {sample_rate} Hz, got {decoded_rate} Hz")
    mono = np.mean(waveform, axis=1, dtype=np.float32)
    pcm = np.clip(np.rint(mono * 32768.0), -32768, 32767).astype("<i2")
    hop_samples = round(sample_rate * 0.01)
    chunk_samples = round(sample_rate * chunk_ms / 1000 / hop_samples) * hop_samples
    window_samples = round(sample_rate * window_seconds)
    chunks = [pcm[start : start + chunk_samples] for start in range(0, len(pcm), chunk_samples)]
    chunks = [chunk[: len(chunk) // hop_samples * hop_samples] for chunk in chunks]
    chunks = [chunk for chunk in chunks if len(chunk)]
    endings = np.cumsum([len(chunk) for chunk in chunks])
    windows = [pcm_wav(pcm[max(0, end - window_samples) : end], sample_rate) for end in endings]

    async def stateless_replay() -> list[tuple[float, int, dict]]:
        records = []
        for payload in windows:
            elapsed, status, body = await send(client, url, payload)
            if status != 200 or not isinstance(body, dict):
                raise RuntimeError(f"stateless replay failed with status {status}: {body}")
            records.append((elapsed, len(payload), body))
        return records

    async def stateful_replay() -> list[tuple[float, int, dict]]:
        response = await client.post(f"{url}/stream")
        response.raise_for_status()
        session_id = response.json()["session_id"]
        records = []
        try:
            for chunk in chunks:
                payload = chunk.tobytes()
                elapsed, status, body = await send_stream(
                    client, url, session_id, payload
                )
                if status != 200 or not isinstance(body, dict):
                    raise RuntimeError(
                        f"streaming replay failed with status {status}: {body}"
                    )
                records.append((elapsed, len(payload), body))
        finally:
            await client.delete(f"{url}/stream/{session_id}")
        return records

    stateless_records = []
    streaming_records = []
    probability_differences = []
    decision_mismatches = 0
    for replay in range(replays):
        if replay % 2:
            right = await stateful_replay()
            left = await stateless_replay()
        else:
            left = await stateless_replay()
            right = await stateful_replay()
        stateless_records.extend(left)
        streaming_records.extend(right)
        for stateless, streaming in zip(left, right, strict=True):
            probability_differences.append(
                abs(float(stateless[2]["p_end"]) - float(streaming[2]["p_end"]))
            )
            decision_mismatches += int(
                stateless[2]["decision"] != streaming[2]["decision"]
            )

    return {
        "method": "warm sequential chronological replays in alternating order",
        "replays": replays,
        "chunk_milliseconds": chunk_samples / sample_rate * 1000,
        "chunk_samples": chunk_samples,
        "stateless": replay_summary(stateless_records),
        "streaming": replay_summary(streaming_records),
        "parity": {
            "maximum_p_end_absolute_difference": max(probability_differences),
            "decision_mismatches": decision_mismatches,
        },
        "streaming_feature_frames": {
            "recomputed_p50": statistics.median(
                [item[2]["stream"]["feature_frames_recomputed"] for item in streaming_records]
            ),
            "reused_p50": statistics.median(
                [item[2]["stream"]["feature_frames_reused"] for item in streaming_records]
            ),
        },
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
        streaming_result = (
            await streaming_comparison(
                client,
                args.url,
                payload,
                sample_rate=audio_info.samplerate,
                window_seconds=float(readiness["preprocessing"]["window_seconds"]),
                chunk_ms=args.streaming_chunk_ms,
                replays=args.streaming_replays,
            )
            if args.streaming_replays
            else None
        )
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
            "streaming_replays": args.streaming_replays,
            "streaming_chunk_ms": args.streaming_chunk_ms,
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
        "streaming_comparison": streaming_result,
        "invalid_requests": {
            name: {"client_latency_ms": result[0], "status": result[1], "response": result[2]}
            for name, result in invalid.items()
        },
    }


def main() -> None:
    args = parse_args()
    if args.streaming_replays < 0:
        raise SystemExit("--streaming-replays must be zero or greater")
    if args.streaming_chunk_ms <= 0:
        raise SystemExit("--streaming-chunk-ms must be greater than zero")
    report = asyncio.run(run(args))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
