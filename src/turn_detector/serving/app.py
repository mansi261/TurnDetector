"""FastAPI service for an exported turn detector."""

from __future__ import annotations

from collections import Counter, deque
from contextlib import asynccontextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from threading import Lock
import time

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
import numpy as np

from turn_detector.data_loader import decode_audio_window_array
from turn_detector.models import WaveformFeatureExtractor


SERVICE_VERSION = "1.2.0"
MAX_AUDIO_BYTES = 25 * 1024 * 1024
MAX_METRIC_EVENTS = 5000
STATIC_DIR = Path(__file__).parent / "static"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def percentile_summary(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"p50": None, "p95": None, "p99": None}
    return {
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
    }


class APIMetrics:
    def __init__(self, capacity: int = MAX_METRIC_EVENTS):
        self.started_at = time.time()
        self.capacity = capacity
        self._lock = Lock()
        self._events: deque[dict[str, object]] = deque(maxlen=capacity)
        self._total_requests = 0
        self._successful_requests = 0
        self._status_counts: Counter[int] = Counter()
        self._outcome_counts: Counter[str] = Counter()

    def record_success(
        self,
        result: dict[str, object],
        *,
        api_total_ms: float,
        request_bytes: int,
    ) -> None:
        timing = result["timing_ms"]
        assert isinstance(timing, dict)
        event = {
            "timestamp": time.time(),
            "status_code": 200,
            "outcome": "success",
            "api_total_ms": api_total_ms,
            "model_total_ms": float(timing["total"]),
            "preprocess_ms": float(timing["preprocess"]),
            "inference_ms": float(timing["inference"]),
            "request_bytes": request_bytes,
            "p_end": float(result["p_end"]),
            "decision": str(result["decision"]),
        }
        with self._lock:
            self._total_requests += 1
            self._successful_requests += 1
            self._status_counts[200] += 1
            self._outcome_counts["success"] += 1
            self._events.append(event)

    def record_error(
        self,
        status_code: int,
        outcome: str,
        *,
        api_total_ms: float,
        request_bytes: int = 0,
    ) -> None:
        event = {
            "timestamp": time.time(),
            "status_code": status_code,
            "outcome": outcome,
            "api_total_ms": api_total_ms,
            "request_bytes": request_bytes,
        }
        with self._lock:
            self._total_requests += 1
            self._status_counts[status_code] += 1
            self._outcome_counts[outcome] += 1
            self._events.append(event)

    def snapshot(
        self,
        *,
        model_version: str | None,
        model_sha256: str | None,
        threshold: float | None,
    ) -> dict[str, object]:
        now = time.time()
        with self._lock:
            events = list(self._events)
            total_requests = self._total_requests
            successful_requests = self._successful_requests
            status_counts = dict(self._status_counts)
            outcome_counts = dict(self._outcome_counts)

        successes = [event for event in events if event["status_code"] == 200]
        recent_minute = [event for event in events if float(event["timestamp"]) >= now - 60]
        decisions = Counter(str(event["decision"]) for event in successes)
        probabilities = [float(event["p_end"]) for event in successes]
        bins = np.linspace(0.0, 1.0, 11)
        histogram = np.histogram(probabilities, bins=bins)[0] if probabilities else np.zeros(10, dtype=int)
        error_requests = total_requests - successful_requests

        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "uptime_seconds": now - self.started_at,
            "service_version": SERVICE_VERSION,
            "model_version": model_version,
            "model_sha256": model_sha256,
            "threshold": threshold,
            "requests": {
                "total": total_requests,
                "successful": successful_requests,
                "errors": error_requests,
                "success_rate": successful_requests / total_requests if total_requests else None,
                "error_rate": error_requests / total_requests if total_requests else None,
                "requests_last_minute": len(recent_minute),
                "status_counts": {str(key): value for key, value in sorted(status_counts.items())},
                "outcome_counts": outcome_counts,
            },
            "latency_ms": {
                "api_total": percentile_summary([float(event["api_total_ms"]) for event in successes]),
                "model_total": percentile_summary([float(event["model_total_ms"]) for event in successes]),
                "preprocess": percentile_summary([float(event["preprocess_ms"]) for event in successes]),
                "inference": percentile_summary([float(event["inference_ms"]) for event in successes]),
            },
            "request_bytes": percentile_summary([float(event["request_bytes"]) for event in events]),
            "outputs": {
                "average_p_end": float(np.mean(probabilities)) if probabilities else None,
                "decisions": {"hold": decisions["hold"], "end": decisions["end"]},
                "histogram": [
                    {
                        "lower": float(bins[index]),
                        "upper": float(bins[index + 1]),
                        "count": int(count),
                    }
                    for index, count in enumerate(histogram)
                ],
            },
            "recent": events[-60:],
            "retained_events": len(events),
            "event_capacity": self.capacity,
        }


class Predictor:
    def __init__(
        self,
        checkpoint_path: Path,
        metadata_path: Path,
        *,
        threshold: float | None = None,
        model_version: str | None = None,
    ):
        import onnxruntime as ort

        started = time.perf_counter()
        self.checkpoint_path = checkpoint_path.resolve()
        self.metadata_path = metadata_path.resolve()
        self.metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        self.model_sha256 = file_sha256(self.checkpoint_path)
        declared_artifacts = self.metadata.get("artifacts", {})
        self.artifact_kind = next(
            (
                name
                for name, artifact in declared_artifacts.items()
                if artifact.get("sha256") == self.model_sha256
            ),
            None,
        )
        if declared_artifacts and self.artifact_kind is None:
            raise ValueError("checkpoint SHA-256 is not declared by the selected metadata")
        selected_artifact = declared_artifacts.get(self.artifact_kind, {})
        if selected_artifact.get("serving_eligible") is False or selected_artifact.get(
            "parity", {}
        ).get("within_tolerance") is False:
            raise ValueError("selected checkpoint did not pass export parity")
        self.model_version = (
            model_version
            or self.metadata.get("model_version")
            or f"sha256:{self.model_sha256[:12]}"
        )
        self.threshold = float(
            self.metadata["threshold"] if threshold is None else threshold
        )
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("threshold must be between 0 and 1")
        self.waveform_normalization = self.metadata.get(
            "waveform_normalization", "none"
        )
        if self.waveform_normalization not in {
            "none",
            "zero_mean_unit_variance_before_padding",
        }:
            raise ValueError("unsupported waveform normalization")

        self.session = ort.InferenceSession(
            self.checkpoint_path.as_posix(), providers=["CPUExecutionProvider"]
        )
        extractor_path = self.metadata_path.parent / self.metadata["feature_extractor_path"]
        self.extractor = WaveformFeatureExtractor(
            extractor_path.as_posix(),
            self.metadata["window_seconds"],
            self.metadata["sample_rate"],
            local_files_only=True,
            return_tensors="np",
        )
        self.startup_seconds = time.perf_counter() - started

    def predict(self, audio_bytes: bytes) -> dict[str, object]:
        total_started = time.perf_counter()
        preprocess_started = time.perf_counter()
        try:
            waveform = decode_audio_window_array(
                audio_bytes,
                end_seconds=None,
                window_seconds=self.metadata["window_seconds"],
                sample_rate=self.metadata["sample_rate"],
                normalize_waveform=(
                    self.waveform_normalization
                    == "zero_mean_unit_variance_before_padding"
                ),
            )
        except (RuntimeError, ValueError) as error:
            raise ValueError(str(error)) from error
        features = np.asarray(self.extractor([waveform]), dtype=np.float32)
        preprocess_ms = (time.perf_counter() - preprocess_started) * 1000

        inference_started = time.perf_counter()
        output = self.session.run(None, {"input_features": features})[0]
        inference_ms = (time.perf_counter() - inference_started) * 1000
        p_end = float(np.asarray(output).reshape(-1)[0])
        should_end = p_end >= self.threshold
        return {
            "p_end": p_end,
            "should_end_turn": should_end,
            "decision": "end" if should_end else "hold",
            "threshold": self.threshold,
            "model_version": self.model_version,
            "timing_ms": {
                "preprocess": preprocess_ms,
                "inference": inference_ms,
                "total": (time.perf_counter() - total_started) * 1000,
            },
        }


def create_app(
    checkpoint_path: str | Path | None = None,
    metadata_path: str | Path | None = None,
    *,
    threshold: float | None = None,
    model_version: str | None = None,
) -> FastAPI:
    configured_checkpoint = (
        checkpoint_path
        or os.environ.get("TURN_DETECTOR_CHECKPOINT")
        or os.environ.get("TURN_DETECTOR_MODEL")
        or "model/model.fp32.onnx"
    )
    configured_metadata = (
        metadata_path
        or os.environ.get("TURN_DETECTOR_METADATA")
        or "model/model_metadata.json"
    )
    configured_threshold = threshold
    if configured_threshold is None and "TURN_DETECTOR_THRESHOLD" in os.environ:
        configured_threshold = float(os.environ["TURN_DETECTOR_THRESHOLD"])
    configured_version = model_version or os.environ.get("TURN_DETECTOR_MODEL_VERSION")

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        application.state.predictor = None
        application.state.load_error = None
        try:
            application.state.predictor = await run_in_threadpool(
                Predictor,
                Path(configured_checkpoint),
                Path(configured_metadata),
                threshold=configured_threshold,
                model_version=configured_version,
            )
        except Exception as error:
            application.state.load_error = f"{type(error).__name__}: {error}"
        yield
        application.state.predictor = None

    application = FastAPI(
        title="Turn Detector",
        version=SERVICE_VERSION,
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    application.state.metrics = APIMetrics()

    @application.get("/", include_in_schema=False)
    def demo() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @application.get("/health")
    def health() -> dict[str, str]:
        return {"status": "alive", "service_version": SERVICE_VERSION}

    @application.get("/ready")
    def ready() -> dict[str, object]:
        predictor: Predictor | None = application.state.predictor
        if predictor is None:
            raise HTTPException(
                status_code=503,
                detail={"status": "not_ready", "error": application.state.load_error},
            )
        return {
            "status": "ready",
            "service_version": SERVICE_VERSION,
            "model_version": predictor.model_version,
            "model_sha256": predictor.model_sha256,
            "artifact_kind": predictor.artifact_kind,
            "threshold": predictor.threshold,
            "startup_seconds": predictor.startup_seconds,
            "providers": predictor.session.get_providers(),
            "preprocessing": {
                "sample_rate": predictor.metadata["sample_rate"],
                "channels": predictor.metadata["channels"],
                "window_seconds": predictor.metadata["window_seconds"],
                "padding": predictor.metadata["padding"],
                "truncation": predictor.metadata["truncation"],
                "waveform_normalization": predictor.waveform_normalization,
            },
        }

    @application.get("/metrics")
    def metrics() -> dict[str, object]:
        predictor: Predictor | None = application.state.predictor
        return application.state.metrics.snapshot(
            model_version=predictor.model_version if predictor else None,
            model_sha256=predictor.model_sha256 if predictor else None,
            threshold=predictor.threshold if predictor else None,
        )

    @application.post("/predict")
    async def predict(audio: UploadFile = File(...)) -> dict[str, object]:
        api_started = time.perf_counter()
        predictor: Predictor | None = application.state.predictor
        if predictor is None:
            application.state.metrics.record_error(
                503,
                "not_ready",
                api_total_ms=(time.perf_counter() - api_started) * 1000,
            )
            raise HTTPException(status_code=503, detail="model is not ready")
        payload = await audio.read(MAX_AUDIO_BYTES + 1)
        if len(payload) > MAX_AUDIO_BYTES:
            application.state.metrics.record_error(
                413,
                "oversized_input",
                api_total_ms=(time.perf_counter() - api_started) * 1000,
                request_bytes=len(payload),
            )
            raise HTTPException(status_code=413, detail="audio exceeds 25 MiB")
        if not payload:
            application.state.metrics.record_error(
                400,
                "empty_input",
                api_total_ms=(time.perf_counter() - api_started) * 1000,
            )
            raise HTTPException(status_code=400, detail="audio is empty")
        try:
            result = await run_in_threadpool(predictor.predict, payload)
        except ValueError as error:
            application.state.metrics.record_error(
                400,
                "invalid_audio",
                api_total_ms=(time.perf_counter() - api_started) * 1000,
                request_bytes=len(payload),
            )
            raise HTTPException(status_code=400, detail=str(error)) from error
        except Exception:
            application.state.metrics.record_error(
                500,
                "server_error",
                api_total_ms=(time.perf_counter() - api_started) * 1000,
                request_bytes=len(payload),
            )
            raise
        application.state.metrics.record_success(
            result,
            api_total_ms=(time.perf_counter() - api_started) * 1000,
            request_bytes=len(payload),
        )
        return result

    return application


app = create_app()
