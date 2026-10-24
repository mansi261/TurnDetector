"""FastAPI service for an exported turn detector."""

from __future__ import annotations

from collections import Counter, deque
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from threading import Lock
import time

from fastapi import FastAPI, File, HTTPException, Request, Response, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
import numpy as np

from turn_detector.data_loader import decode_audio_window_array
from turn_detector.models import WaveformFeatureExtractor
from turn_detector.serving.streaming import StreamingSession, StreamingSessionStore
from turn_detector.utils import file_sha256


SERVICE_VERSION = "1.7.0"
MAX_AUDIO_BYTES = 25 * 1024 * 1024
MAX_METRIC_EVENTS = 5000
ROLLING_METRIC_EVENTS = 60
STATIC_DIR = Path(__file__).parent / "static"


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    display_name: str
    checkpoint_path: Path
    metadata_path: Path
    model_version: str | None


@dataclass
class ModelRuntime:
    spec: ModelSpec
    predictor: "Predictor"
    streams: StreamingSessionStore | None


def load_model_specs(
    checkpoint_path: str | Path,
    metadata_path: str | Path,
    *,
    model_version: str | None,
    manifest_path: str | Path | None,
) -> tuple[str, list[ModelSpec]]:
    if manifest_path is None:
        model_id = os.environ.get("TURN_DETECTOR_MODEL_ID", "english")
        return model_id, [
            ModelSpec(
                model_id=model_id,
                display_name=os.environ.get(
                    "TURN_DETECTOR_MODEL_DISPLAY_NAME", "English"
                ),
                checkpoint_path=Path(checkpoint_path),
                metadata_path=Path(metadata_path),
                model_version=model_version,
            )
        ]

    manifest = Path(manifest_path).resolve()
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    default_model_id = str(payload["default_model"])
    entries = payload.get("models")
    if not isinstance(entries, list) or not entries:
        raise ValueError("model manifest must contain a non-empty models list")

    specs = []
    seen = set()
    for entry in entries:
        model_id = str(entry["id"])
        if not model_id or model_id in seen:
            raise ValueError("model manifest IDs must be unique and non-empty")
        seen.add(model_id)
        checkpoint = Path(entry["checkpoint"])
        metadata = Path(entry["metadata"])
        specs.append(
            ModelSpec(
                model_id=model_id,
                display_name=str(entry.get("display_name", model_id)),
                checkpoint_path=(manifest.parent / checkpoint).resolve()
                if not checkpoint.is_absolute()
                else checkpoint,
                metadata_path=(manifest.parent / metadata).resolve()
                if not metadata.is_absolute()
                else metadata,
                model_version=entry.get("model_version"),
            )
        )
    if default_model_id not in seen:
        raise ValueError("default model is not declared in the model manifest")
    return default_model_id, specs


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
        self._model_total_requests: Counter[str] = Counter()
        self._model_successful_requests: Counter[str] = Counter()
        self._model_status_counts: dict[str, Counter[int]] = {}
        self._model_outcome_counts: dict[str, Counter[str]] = {}

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
            "model_id": result.get("model_id"),
            "model_version": result.get("model_version"),
        }
        with self._lock:
            self._total_requests += 1
            self._successful_requests += 1
            self._status_counts[200] += 1
            self._outcome_counts["success"] += 1
            model_id = result.get("model_id")
            if model_id is not None:
                key = str(model_id)
                self._model_total_requests[key] += 1
                self._model_successful_requests[key] += 1
                self._model_status_counts.setdefault(key, Counter())[200] += 1
                self._model_outcome_counts.setdefault(key, Counter())["success"] += 1
            self._events.append(event)

    def record_error(
        self,
        status_code: int,
        outcome: str,
        *,
        api_total_ms: float,
        request_bytes: int = 0,
        model_id: str | None = None,
    ) -> None:
        event = {
            "timestamp": time.time(),
            "status_code": status_code,
            "outcome": outcome,
            "api_total_ms": api_total_ms,
            "request_bytes": request_bytes,
            "model_id": model_id,
        }
        with self._lock:
            self._total_requests += 1
            self._status_counts[status_code] += 1
            self._outcome_counts[outcome] += 1
            if model_id is not None:
                self._model_total_requests[model_id] += 1
                self._model_status_counts.setdefault(model_id, Counter())[status_code] += 1
                self._model_outcome_counts.setdefault(model_id, Counter())[outcome] += 1
            self._events.append(event)

    def snapshot(
        self,
        *,
        model_version: str | None,
        model_sha256: str | None,
        threshold: float | None,
        model_id: str | None = None,
    ) -> dict[str, object]:
        now = time.time()
        with self._lock:
            all_events = list(self._events)
            if model_id is None:
                events = all_events
                total_requests = self._total_requests
                successful_requests = self._successful_requests
                status_counts = dict(self._status_counts)
                outcome_counts = dict(self._outcome_counts)
            else:
                events = [event for event in all_events if event.get("model_id") == model_id]
                total_requests = self._model_total_requests[model_id]
                successful_requests = self._model_successful_requests[model_id]
                status_counts = dict(self._model_status_counts.get(model_id, Counter()))
                outcome_counts = dict(self._model_outcome_counts.get(model_id, Counter()))

        successes = [event for event in events if event["status_code"] == 200]
        rolling_successes = successes[-ROLLING_METRIC_EVENTS:]
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
            "model_id": model_id,
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
            "rolling_latency_ms": {
                "window_size": ROLLING_METRIC_EVENTS,
                "sample_count": len(rolling_successes),
                "api_total": percentile_summary(
                    [float(event["api_total_ms"]) for event in rolling_successes]
                ),
                "model_total": percentile_summary(
                    [float(event["model_total_ms"]) for event in rolling_successes]
                ),
                "preprocess": percentile_summary(
                    [float(event["preprocess_ms"]) for event in rolling_successes]
                ),
                "inference": percentile_summary(
                    [float(event["inference_ms"]) for event in rolling_successes]
                ),
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
        ort_intra_op_threads: int | str = 2,
        ort_inter_op_threads: int | str = 1,
        ort_execution_mode: str = "sequential",
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

        try:
            intra_op_threads = int(ort_intra_op_threads)
            inter_op_threads = int(ort_inter_op_threads)
        except (TypeError, ValueError) as error:
            raise ValueError("ONNX Runtime thread counts must be integers") from error
        if intra_op_threads < 0 or inter_op_threads < 0:
            raise ValueError("ONNX Runtime thread counts must be zero or greater")
        execution_mode = ort_execution_mode.strip().lower()
        if execution_mode not in {"sequential", "parallel"}:
            raise ValueError(
                "ONNX Runtime execution mode must be sequential or parallel"
            )

        session_options = ort.SessionOptions()
        session_options.intra_op_num_threads = intra_op_threads
        session_options.inter_op_num_threads = inter_op_threads
        session_options.execution_mode = (
            ort.ExecutionMode.ORT_SEQUENTIAL
            if execution_mode == "sequential"
            else ort.ExecutionMode.ORT_PARALLEL
        )
        session_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        self.session = ort.InferenceSession(
            self.checkpoint_path.as_posix(),
            sess_options=session_options,
            providers=["CPUExecutionProvider"],
        )
        self.ort_config = {
            "execution_mode": execution_mode,
            "graph_optimization_level": "all",
            "intra_op_num_threads": intra_op_threads,
            "inter_op_num_threads": inter_op_threads,
            "zero_threads_mean_runtime_default": True,
            "visible_cpu_count": os.cpu_count(),
        }
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

        return self.predict_features(
            features,
            total_started=total_started,
            preprocess_ms=preprocess_ms,
        )

    def predict_features(
        self,
        features: np.ndarray,
        *,
        total_started: float,
        preprocess_ms: float,
    ) -> dict[str, object]:
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


def predict_stream_chunk(
    predictor: Predictor,
    session: StreamingSession,
    payload: bytes,
    model_id: str,
) -> dict[str, object]:
    with session.lock:
        total_started = time.perf_counter()
        preprocess_started = time.perf_counter()
        features, stream_details = session.features.append_pcm_s16le(payload)
        preprocess_ms = (time.perf_counter() - preprocess_started) * 1000
        result = predictor.predict_features(
            features,
            total_started=total_started,
            preprocess_ms=preprocess_ms,
        )
        result["model_id"] = model_id
        result["stream"] = {
            "session_id": session.session_id,
            **stream_details,
        }
        return result


def create_app(
    checkpoint_path: str | Path | None = None,
    metadata_path: str | Path | None = None,
    *,
    model_manifest: str | Path | None = None,
    threshold: float | None = None,
    model_version: str | None = None,
    ort_intra_op_threads: int | str | None = None,
    ort_inter_op_threads: int | str | None = None,
    ort_execution_mode: str | None = None,
) -> FastAPI:
    configured_checkpoint = (
        checkpoint_path
        or os.environ.get("TURN_DETECTOR_CHECKPOINT")
        or "model/model.fp32.onnx"
    )
    configured_metadata = (
        metadata_path
        or os.environ.get("TURN_DETECTOR_METADATA")
        or "model/model_metadata.json"
    )
    configured_manifest = model_manifest
    if configured_manifest is None and checkpoint_path is None and metadata_path is None:
        configured_manifest = os.environ.get("TURN_DETECTOR_MODEL_MANIFEST")
    configured_threshold = threshold
    if configured_threshold is None and "TURN_DETECTOR_THRESHOLD" in os.environ:
        configured_threshold = float(os.environ["TURN_DETECTOR_THRESHOLD"])
    configured_version = model_version or os.environ.get("TURN_DETECTOR_MODEL_VERSION")
    configured_ort_intra_op_threads = (
        ort_intra_op_threads
        if ort_intra_op_threads is not None
        else os.environ.get("TURN_DETECTOR_ORT_INTRA_OP_THREADS", "2")
    )
    configured_ort_inter_op_threads = (
        ort_inter_op_threads
        if ort_inter_op_threads is not None
        else os.environ.get("TURN_DETECTOR_ORT_INTER_OP_THREADS", "1")
    )
    configured_ort_execution_mode = (
        ort_execution_mode
        or os.environ.get("TURN_DETECTOR_ORT_EXECUTION_MODE", "sequential")
    )

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        application.state.predictor = None
        application.state.streams = None
        application.state.runtimes = {}
        application.state.default_model_id = None
        application.state.load_error = None
        application.state.load_errors = {}
        try:
            default_model_id, specs = load_model_specs(
                configured_checkpoint,
                configured_metadata,
                model_version=configured_version,
                manifest_path=configured_manifest,
            )
            application.state.default_model_id = default_model_id
            for spec in specs:
                try:
                    predictor = await run_in_threadpool(
                        Predictor,
                        spec.checkpoint_path,
                        spec.metadata_path,
                        threshold=configured_threshold,
                        model_version=spec.model_version,
                        ort_intra_op_threads=configured_ort_intra_op_threads,
                        ort_inter_op_threads=configured_ort_inter_op_threads,
                        ort_execution_mode=configured_ort_execution_mode,
                    )
                    streams = (
                        StreamingSessionStore(predictor.extractor)
                        if predictor.waveform_normalization == "none"
                        else None
                    )
                    application.state.runtimes[spec.model_id] = ModelRuntime(
                        spec=spec,
                        predictor=predictor,
                        streams=streams,
                    )
                except Exception as error:
                    application.state.load_errors[spec.model_id] = (
                        f"{type(error).__name__}: {error}"
                    )
            default_runtime = application.state.runtimes.get(default_model_id)
            if default_runtime is None:
                raise RuntimeError(
                    application.state.load_errors.get(
                        default_model_id, "default model did not load"
                    )
                )
            application.state.predictor = default_runtime.predictor
            application.state.streams = default_runtime.streams
        except Exception as error:
            application.state.load_error = f"{type(error).__name__}: {error}"
        yield
        for runtime in application.state.runtimes.values():
            if runtime.streams is not None:
                runtime.streams.clear()
        application.state.runtimes = {}
        application.state.streams = None
        application.state.predictor = None

    application = FastAPI(
        title="Turn Detector",
        version=SERVICE_VERSION,
        lifespan=lifespan,
        openapi_url=None,
    )
    application.state.metrics = APIMetrics()

    def runtime_summary(runtime: ModelRuntime) -> dict[str, object]:
        predictor = runtime.predictor
        return {
            "id": runtime.spec.model_id,
            "display_name": runtime.spec.display_name,
            "model_version": predictor.model_version,
            "model_sha256": predictor.model_sha256,
            "artifact_kind": predictor.artifact_kind,
            "language_scope": predictor.metadata.get("language_scope", ["eng"]),
            "threshold": predictor.threshold,
            "startup_seconds": predictor.startup_seconds,
            "streaming_enabled": runtime.streams is not None,
        }

    def select_runtime(model_id: str | None = None) -> ModelRuntime:
        selected = model_id or application.state.default_model_id
        runtime = application.state.runtimes.get(selected)
        if runtime is None:
            if application.state.load_error and not application.state.runtimes:
                raise HTTPException(status_code=503, detail="model is not ready")
            raise HTTPException(status_code=404, detail=f"model is not available: {selected}")
        return runtime

    def find_stream(session_id: str) -> tuple[ModelRuntime, StreamingSession] | None:
        for runtime in application.state.runtimes.values():
            if runtime.streams is None:
                continue
            session = runtime.streams.get(session_id)
            if session is not None:
                return runtime, session
        return None

    @application.get("/", include_in_schema=False)
    def demo() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @application.get("/health")
    def health() -> dict[str, str]:
        return {"status": "alive", "service_version": SERVICE_VERSION}

    @application.get("/ready")
    def ready() -> dict[str, object]:
        try:
            runtime = select_runtime()
        except HTTPException:
            raise HTTPException(
                status_code=503,
                detail={"status": "not_ready", "error": application.state.load_error},
            )
        predictor = runtime.predictor
        return {
            "status": "ready",
            "service_version": SERVICE_VERSION,
            "default_model_id": runtime.spec.model_id,
            "models": [
                runtime_summary(item)
                for item in application.state.runtimes.values()
            ],
            "model_load_errors": application.state.load_errors,
            "model_version": predictor.model_version,
            "model_sha256": predictor.model_sha256,
            "artifact_kind": predictor.artifact_kind,
            "language_scope": predictor.metadata.get("language_scope", ["eng"]),
            "threshold": predictor.threshold,
            "startup_seconds": predictor.startup_seconds,
            "providers": predictor.session.get_providers(),
            "onnx_runtime": predictor.ort_config,
            "preprocessing": {
                "sample_rate": predictor.metadata["sample_rate"],
                "channels": predictor.metadata["channels"],
                "window_seconds": predictor.metadata["window_seconds"],
                "padding": predictor.metadata["padding"],
                "truncation": predictor.metadata["truncation"],
                "waveform_normalization": predictor.waveform_normalization,
            },
            "streaming": {
                "enabled": runtime.streams is not None,
                "encoding": "pcm_s16le",
                "feature_hop_ms": 10,
            },
        }

    @application.get("/metrics")
    def metrics(model: str | None = None) -> dict[str, object]:
        runtime = select_runtime(model) if model is not None else None
        predictor: Predictor | None = (
            runtime.predictor if runtime is not None else application.state.predictor
        )
        result = application.state.metrics.snapshot(
            model_version=predictor.model_version if predictor else None,
            model_sha256=predictor.model_sha256 if predictor else None,
            threshold=predictor.threshold if predictor else None,
            model_id=runtime.spec.model_id if runtime is not None else None,
        )
        result["selected_model_id"] = runtime.spec.model_id if runtime is not None else None
        result["default_model_id"] = application.state.default_model_id
        result["models"] = [
            runtime_summary(runtime)
            for runtime in application.state.runtimes.values()
        ]
        stream_snapshots = {
            model_id: runtime.streams.snapshot()
            for model_id, runtime in application.state.runtimes.items()
            if runtime.streams is not None
        }
        result["streams"] = stream_snapshots
        return result

    @application.post("/predict")
    async def predict(
        audio: UploadFile = File(...), model: str | None = None
    ) -> dict[str, object]:
        api_started = time.perf_counter()
        try:
            runtime = select_runtime(model)
        except HTTPException as error:
            application.state.metrics.record_error(
                error.status_code,
                "model_unavailable",
                api_total_ms=(time.perf_counter() - api_started) * 1000,
            )
            raise
        predictor = runtime.predictor
        payload = await audio.read(MAX_AUDIO_BYTES + 1)
        if len(payload) > MAX_AUDIO_BYTES:
            application.state.metrics.record_error(
                413,
                "oversized_input",
                api_total_ms=(time.perf_counter() - api_started) * 1000,
                request_bytes=len(payload),
                model_id=runtime.spec.model_id,
            )
            raise HTTPException(status_code=413, detail="audio exceeds 25 MiB")
        if not payload:
            application.state.metrics.record_error(
                400,
                "empty_input",
                api_total_ms=(time.perf_counter() - api_started) * 1000,
                model_id=runtime.spec.model_id,
            )
            raise HTTPException(status_code=400, detail="audio is empty")
        try:
            result = await run_in_threadpool(predictor.predict, payload)
            result["model_id"] = runtime.spec.model_id
        except ValueError as error:
            application.state.metrics.record_error(
                400,
                "invalid_audio",
                api_total_ms=(time.perf_counter() - api_started) * 1000,
                request_bytes=len(payload),
                model_id=runtime.spec.model_id,
            )
            raise HTTPException(status_code=400, detail=str(error)) from error
        except Exception:
            application.state.metrics.record_error(
                500,
                "server_error",
                api_total_ms=(time.perf_counter() - api_started) * 1000,
                request_bytes=len(payload),
                model_id=runtime.spec.model_id,
            )
            raise
        application.state.metrics.record_success(
            result,
            api_total_ms=(time.perf_counter() - api_started) * 1000,
            request_bytes=len(payload),
        )
        return result

    @application.post("/stream")
    def create_stream(model: str | None = None) -> dict[str, object]:
        runtime = select_runtime(model)
        predictor = runtime.predictor
        streams = runtime.streams
        if streams is None:
            raise HTTPException(
                status_code=409,
                detail="streaming feature reuse is unavailable for this preprocessing",
            )
        try:
            session = streams.create()
        except RuntimeError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error
        return {
            "session_id": session.session_id,
            "model_id": runtime.spec.model_id,
            "model_version": predictor.model_version,
            "display_name": runtime.spec.display_name,
            "encoding": "pcm_s16le",
            "sample_rate": predictor.metadata["sample_rate"],
            "channels": predictor.metadata["channels"],
            "window_seconds": predictor.metadata["window_seconds"],
            "feature_hop_samples": session.features.hop_length,
        }

    @application.post("/stream/{session_id}/predict")
    async def stream_predict(session_id: str, request: Request) -> dict[str, object]:
        api_started = time.perf_counter()
        if not application.state.runtimes:
            raise HTTPException(status_code=503, detail="streaming is not ready")
        matched = find_stream(session_id)
        if matched is None:
            raise HTTPException(status_code=404, detail="stream not found or expired")
        runtime, session = matched
        predictor = runtime.predictor

        maximum_bytes = int(
            predictor.metadata["sample_rate"]
            * predictor.metadata["window_seconds"]
            * 2
        )
        payload = bytearray()
        async for chunk in request.stream():
            payload.extend(chunk)
            if len(payload) > maximum_bytes:
                application.state.metrics.record_error(
                    413,
                    "oversized_stream_chunk",
                    api_total_ms=(time.perf_counter() - api_started) * 1000,
                    request_bytes=len(payload),
                    model_id=runtime.spec.model_id,
                )
                raise HTTPException(status_code=413, detail="stream chunk exceeds one window")
        try:
            result = await run_in_threadpool(
                predict_stream_chunk,
                predictor,
                session,
                bytes(payload),
                runtime.spec.model_id,
            )
        except ValueError as error:
            application.state.metrics.record_error(
                400,
                "invalid_stream_audio",
                api_total_ms=(time.perf_counter() - api_started) * 1000,
                request_bytes=len(payload),
                model_id=runtime.spec.model_id,
            )
            raise HTTPException(status_code=400, detail=str(error)) from error
        except Exception:
            application.state.metrics.record_error(
                500,
                "server_error",
                api_total_ms=(time.perf_counter() - api_started) * 1000,
                request_bytes=len(payload),
                model_id=runtime.spec.model_id,
            )
            raise
        application.state.metrics.record_success(
            result,
            api_total_ms=(time.perf_counter() - api_started) * 1000,
            request_bytes=len(payload),
        )
        return result

    @application.delete("/stream/{session_id}", status_code=204)
    def delete_stream(session_id: str) -> Response:
        for runtime in application.state.runtimes.values():
            if runtime.streams is not None and runtime.streams.delete(session_id):
                return Response(status_code=204)
        raise HTTPException(status_code=404, detail="stream not found or expired")

    return application


app = create_app()
