"""Verify raw-audio prediction parity across PyTorch, ONNX, and FastAPI."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from io import BytesIO
import json
import numpy as np
from pathlib import Path
import wave

from fastapi.testclient import TestClient
import torch
import soundfile as sf

from turn_detector.data_loader import CanonicalEvaluationDataset, decode_audio_window
from turn_detector.models import load_model_checkpoint
from turn_detector.serving.app import Predictor, create_app
from turn_detector.utils import file_sha256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-checkpoint", type=Path, required=True)
    parser.add_argument("--serving-checkpoint", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--audio", type=Path, action="append", default=[])
    parser.add_argument("--canonical-index", type=Path)
    parser.add_argument("--canonical-records", type=int, default=2)
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--ort-intra-op-threads", type=int, default=2)
    parser.add_argument("--ort-inter-op-threads", type=int, default=1)
    parser.add_argument(
        "--ort-execution-mode",
        choices=("sequential", "parallel"),
        default="sequential",
    )
    parser.add_argument("--tolerance", type=float, default=1e-4)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def silence_wav(duration_seconds: float = 0.25, sample_rate: int = 16000) -> bytes:
    output = BytesIO()
    with wave.open(output, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(sample_rate)
        audio.writeframes(b"\x00\x00" * round(duration_seconds * sample_rate))
    return output.getvalue()


def causal_prefix_wav(
    audio_source: bytes | Path,
    *,
    end_seconds: float,
    sample_rate: int,
) -> bytes:
    source = BytesIO(audio_source) if isinstance(audio_source, bytes) else audio_source
    waveform, decoded_rate = sf.read(source, dtype="float32", always_2d=True)
    if decoded_rate != sample_rate:
        raise ValueError(f"expected {sample_rate} Hz, got {decoded_rate} Hz")
    waveform = waveform[: min(len(waveform), round(end_seconds * sample_rate))]
    output = BytesIO()
    sf.write(output, waveform, sample_rate, format="WAV", subtype="PCM_16")
    return output.getvalue()


def streaming_pcm(audio_source: bytes, sample_rate: int) -> tuple[bytes, bytes]:
    waveform, decoded_rate = sf.read(
        BytesIO(audio_source), dtype="float32", always_2d=True
    )
    if decoded_rate != sample_rate:
        raise ValueError(f"expected {sample_rate} Hz, got {decoded_rate} Hz")
    mono = np.mean(waveform, axis=1, dtype=np.float32)
    sample_count = len(mono) // 160 * 160
    mono = mono[:sample_count]
    integers = np.clip(np.rint(mono * 32768.0), -32768, 32767).astype("<i2")
    pcm = integers.tobytes()
    output = BytesIO()
    with wave.open(output, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(sample_rate)
        audio.writeframes(pcm)
    return pcm, output.getvalue()


def main() -> None:
    args = parse_args()
    direct = Predictor(
        args.serving_checkpoint,
        args.metadata,
        threshold=args.threshold,
        model_version="parity-check",
        ort_intra_op_threads=args.ort_intra_op_threads,
        ort_inter_op_threads=args.ort_inter_op_threads,
        ort_execution_mode=args.ort_execution_mode,
    )
    model, _ = load_model_checkpoint(args.training_checkpoint, device="cpu")
    app = create_app(
        args.serving_checkpoint,
        args.metadata,
        threshold=args.threshold,
        model_version="parity-check",
        ort_intra_op_threads=args.ort_intra_op_threads,
        ort_inter_op_threads=args.ort_inter_op_threads,
        ort_execution_mode=args.ort_execution_mode,
    )
    samples = [(path.name, path.read_bytes()) for path in args.audio]
    if args.canonical_index:
        dataset = CanonicalEvaluationDataset(
            args.canonical_index,
            language="eng",
                partition="ENG_evaluation_dev",
                window_seconds=direct.metadata["window_seconds"],
                sample_rate=direct.metadata["sample_rate"],
                normalize_waveform=(
                    direct.waveform_normalization
                    == "zero_mean_unit_variance_before_padding"
                ),
        )
        selected_labels = set()
        for decision in dataset.decisions:
            if decision.label in selected_labels:
                continue
            audio_source = dataset._audio_source(decision)
            payload = causal_prefix_wav(
                audio_source,
                end_seconds=decision.silence_end_s,
                sample_rate=direct.metadata["sample_rate"],
            )
            samples.append(
                (f"ENG_evaluation_dev:{decision.source_id}:{decision.label}", payload)
            )
            selected_labels.add(decision.label)
            if len(selected_labels) >= args.canonical_records:
                break
    if not samples:
        raise SystemExit("provide --audio or --canonical-index")
    samples.append(("boundary_250ms_silence.wav", silence_wav()))
    records = []

    with TestClient(app) as client, torch.inference_mode():
        readiness = client.get("/ready")
        readiness.raise_for_status()
        for name, payload in samples:
            waveform = decode_audio_window(
                payload,
                end_seconds=None,
                window_seconds=direct.metadata["window_seconds"],
                sample_rate=direct.metadata["sample_rate"],
                normalize_waveform=(
                    direct.waveform_normalization
                    == "zero_mean_unit_variance_before_padding"
                ),
            )
            features = torch.from_numpy(np.asarray(direct.extractor([waveform])))
            pytorch_p_end = float(torch.sigmoid(model(features)).reshape(-1)[0])
            direct_result = direct.predict(payload)
            response = client.post(
                "/predict",
                files={"audio": (name, payload, "audio/wav")},
            )
            response.raise_for_status()
            api_result = response.json()
            pcm, streaming_reference_audio = streaming_pcm(
                payload, direct.metadata["sample_rate"]
            )
            streaming_reference = direct.predict(streaming_reference_audio)
            stream_response = client.post("/stream")
            stream_response.raise_for_status()
            session_id = stream_response.json()["session_id"]
            streaming_result = None
            chunk_bytes = 5600 * 2
            for offset in range(0, len(pcm), chunk_bytes):
                streaming_response = client.post(
                    f"/stream/{session_id}/predict",
                    content=pcm[offset : offset + chunk_bytes],
                    headers={"Content-Type": "application/octet-stream"},
                )
                streaming_response.raise_for_status()
                streaming_result = streaming_response.json()
            client.delete(f"/stream/{session_id}").raise_for_status()
            if streaming_result is None:
                raise RuntimeError(f"sample {name} is shorter than one feature hop")
            direct_p_end = float(direct_result["p_end"])
            api_p_end = float(api_result["p_end"])
            streaming_reference_p_end = float(streaming_reference["p_end"])
            streaming_api_p_end = float(streaming_result["p_end"])
            records.append(
                {
                    "sample": name,
                    "bytes": len(payload),
                    "pytorch_p_end": pytorch_p_end,
                    "direct_onnx_p_end": direct_p_end,
                    "api_p_end": api_p_end,
                    "streaming_reference_p_end": streaming_reference_p_end,
                    "streaming_api_p_end": streaming_api_p_end,
                    "pytorch_onnx_absolute_difference": abs(pytorch_p_end - direct_p_end),
                    "direct_api_absolute_difference": abs(direct_p_end - api_p_end),
                    "streaming_api_absolute_difference": abs(
                        streaming_reference_p_end - streaming_api_p_end
                    ),
                    "pytorch_decision": pytorch_p_end >= direct.threshold,
                    "direct_decision": direct_result["should_end_turn"],
                    "api_decision": api_result["should_end_turn"],
                    "streaming_reference_decision": streaming_reference[
                        "should_end_turn"
                    ],
                    "streaming_api_decision": streaming_result["should_end_turn"],
                    "streaming_final_update": streaming_result["stream"],
                }
            )

    maximum_pytorch_onnx = max(
        record["pytorch_onnx_absolute_difference"] for record in records
    )
    maximum_direct_api = max(record["direct_api_absolute_difference"] for record in records)
    maximum_streaming_api = max(
        record["streaming_api_absolute_difference"] for record in records
    )
    decision_mismatches = sum(
        not (
            record["pytorch_decision"]
            == record["direct_decision"]
            == record["api_decision"]
        )
        for record in records
    )
    streaming_decision_mismatches = sum(
        record["streaming_reference_decision"]
        != record["streaming_api_decision"]
        for record in records
    )
    passed = (
        maximum_pytorch_onnx <= args.tolerance
        and maximum_direct_api <= args.tolerance
        and maximum_streaming_api <= args.tolerance
        and decision_mismatches == 0
        and streaming_decision_mismatches == 0
    )
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "passed": passed,
        "tolerance": args.tolerance,
        "threshold": direct.threshold,
        "records": records,
        "summary": {
            "samples": len(records),
            "maximum_pytorch_onnx_absolute_difference": maximum_pytorch_onnx,
            "maximum_direct_api_absolute_difference": maximum_direct_api,
            "maximum_streaming_api_absolute_difference": maximum_streaming_api,
            "decision_mismatches": decision_mismatches,
            "streaming_decision_mismatches": streaming_decision_mismatches,
        },
        "artifacts": {
            "training_checkpoint": args.training_checkpoint.resolve().as_posix(),
            "training_checkpoint_sha256": file_sha256(args.training_checkpoint),
            "serving_checkpoint": args.serving_checkpoint.resolve().as_posix(),
            "serving_checkpoint_sha256": direct.model_sha256,
            "metadata": args.metadata.resolve().as_posix(),
            "metadata_sha256": file_sha256(args.metadata),
            "canonical_index": args.canonical_index.resolve().as_posix()
            if args.canonical_index
            else None,
        },
        "preprocessing": {
            "implementation": "stateless full-window and stateful incremental Whisper features",
            "sample_rate": direct.metadata["sample_rate"],
            "channels": direct.metadata["channels"],
            "window_seconds": direct.metadata["window_seconds"],
            "padding": direct.metadata["padding"],
            "truncation": direct.metadata["truncation"],
            "waveform_normalization": direct.waveform_normalization,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
