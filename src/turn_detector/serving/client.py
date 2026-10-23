"""Run one reproducible audio inference through the turn detector API."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
from pathlib import Path
import time
from typing import Any

import httpx
import soundfile as sf


CONTENT_TYPES = {".wav": "audio/wav", ".flac": "audio/flac"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--end-seconds", type=float)
    parser.add_argument("--expect-model-version")
    parser.add_argument("--expect-threshold", type=float)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def bytes_sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def response_json(response: httpx.Response, operation: str) -> dict[str, Any]:
    try:
        body = response.json()
    except ValueError:
        body = {"detail": response.text.strip() or response.reason_phrase}
    if response.is_error:
        raise RuntimeError(
            f"{operation} returned HTTP {response.status_code}: {body.get('detail', body)}"
        )
    if not isinstance(body, dict):
        raise RuntimeError(f"{operation} returned an unexpected response")
    return body


def prepare_audio(
    path: Path,
    *,
    expected_sample_rate: int,
    end_seconds: float | None,
) -> tuple[bytes, str, dict[str, Any]]:
    if not path.is_file():
        raise ValueError(f"audio file does not exist: {path}")
    suffix = path.suffix.lower()
    if suffix not in CONTENT_TYPES:
        raise ValueError("audio must be a WAV or FLAC file")
    if end_seconds is not None and end_seconds <= 0:
        raise ValueError("--end-seconds must be greater than zero")

    source_bytes = path.read_bytes()
    try:
        info = sf.info(BytesIO(source_bytes))
    except RuntimeError as error:
        raise ValueError(f"audio could not be decoded: {error}") from error
    if info.frames <= 0:
        raise ValueError("audio contains no samples")
    if info.samplerate != expected_sample_rate:
        raise ValueError(
            f"expected {expected_sample_rate} Hz audio, got {info.samplerate} Hz"
        )
    source_duration = info.frames / info.samplerate
    observation_end = source_duration if end_seconds is None else min(end_seconds, source_duration)

    if end_seconds is None:
        payload = source_bytes
        content_type = CONTENT_TYPES[suffix]
        payload_encoding = suffix.removeprefix(".")
    else:
        frames = min(info.frames, round(observation_end * info.samplerate))
        audio, decoded_rate = sf.read(
            BytesIO(source_bytes), frames=frames, dtype="float32", always_2d=True
        )
        if not len(audio):
            raise ValueError("the selected causal prefix contains no audio")
        encoded = BytesIO()
        sf.write(encoded, audio, decoded_rate, format="WAV", subtype="PCM_16")
        payload = encoded.getvalue()
        content_type = "audio/wav"
        payload_encoding = "wav_pcm16_causal_prefix"

    metadata = {
        "source_path": path.resolve().as_posix(),
        "source_sha256": bytes_sha256(source_bytes),
        "source_bytes": len(source_bytes),
        "source_duration_seconds": source_duration,
        "sample_rate_hz": info.samplerate,
        "channels": info.channels,
        "observation_end_seconds": observation_end,
        "causal_prefix": end_seconds is not None,
        "payload_encoding": payload_encoding,
        "payload_sha256": bytes_sha256(payload),
        "payload_bytes": len(payload),
    }
    return payload, content_type, metadata


def enforce_expected_service(readiness: dict[str, Any], args: argparse.Namespace) -> None:
    if (
        args.expect_model_version is not None
        and readiness.get("model_version") != args.expect_model_version
    ):
        raise RuntimeError(
            f"expected model {args.expect_model_version!r}, got {readiness.get('model_version')!r}"
        )
    if args.expect_threshold is not None:
        actual = float(readiness["threshold"])
        if abs(actual - args.expect_threshold) > 1e-12:
            raise RuntimeError(
                f"expected threshold {args.expect_threshold}, got {actual}"
            )


def run_inference(args: argparse.Namespace) -> dict[str, Any]:
    base_url = args.url.rstrip("/")
    total_started = time.perf_counter()
    with httpx.Client(timeout=args.timeout) as client:
        ready_started = time.perf_counter()
        readiness = response_json(client.get(f"{base_url}/ready"), "readiness check")
        ready_http_ms = (time.perf_counter() - ready_started) * 1000
        enforce_expected_service(readiness, args)

        preprocessing = readiness.get("preprocessing", {})
        payload, content_type, input_metadata = prepare_audio(
            args.audio,
            expected_sample_rate=int(preprocessing["sample_rate"]),
            end_seconds=args.end_seconds,
        )
        predict_started = time.perf_counter()
        upload_name = (
            f"{args.audio.stem}.prefix.wav"
            if input_metadata["causal_prefix"]
            else args.audio.name
        )
        response = client.post(
            f"{base_url}/predict",
            files={"audio": (upload_name, payload, content_type)},
        )
        predict_http_ms = (time.perf_counter() - predict_started) * 1000
        prediction = response_json(response, "prediction")

    if prediction.get("model_version") != readiness.get("model_version"):
        raise RuntimeError("model version changed between readiness and prediction")
    if abs(float(prediction["threshold"]) - float(readiness["threshold"])) > 1e-12:
        raise RuntimeError("threshold changed between readiness and prediction")

    server_pipeline_ms = float(prediction["timing_ms"]["total"])
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "endpoint": base_url,
        "input": input_metadata,
        "service": {
            "status": readiness["status"],
            "service_version": readiness["service_version"],
            "model_version": readiness["model_version"],
            "model_sha256": readiness["model_sha256"],
            "artifact_kind": readiness["artifact_kind"],
            "threshold": readiness["threshold"],
            "providers": readiness["providers"],
            "preprocessing": preprocessing,
        },
        "prediction": prediction,
        "client_timing_ms": {
            "readiness_http": ready_http_ms,
            "prediction_http": predict_http_ms,
            "server_pipeline": server_pipeline_ms,
            "prediction_http_minus_server_pipeline": max(
                0.0, predict_http_ms - server_pipeline_ms
            ),
            "complete_run": (time.perf_counter() - total_started) * 1000,
        },
    }


def main() -> None:
    args = parse_args()
    try:
        result = run_inference(args)
    except (OSError, ValueError, RuntimeError, httpx.HTTPError) as error:
        raise SystemExit(f"inference failed: {error}") from error
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
