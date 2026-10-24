"""Verify PyTorch, ONNX, and FastAPI parity for all multilingual languages."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from io import BytesIO
import json
from pathlib import Path

from fastapi.testclient import TestClient
import numpy as np
import soundfile as sf
import torch

from turn_detector.data_loader import MultilingualEvaluationDataset, decode_audio_window
from turn_detector.models import load_model_checkpoint
from turn_detector.serving.app import Predictor, create_app
from turn_detector.training.train import read_config
from turn_detector.utils import file_sha256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/multilingual_whisper_tiny_v1.json")
    )
    parser.add_argument("--training-checkpoint", type=Path, required=True)
    parser.add_argument("--serving-checkpoint", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--tolerance", type=float, default=1e-4)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def wav_prefix(dataset: MultilingualEvaluationDataset, index: int) -> bytes:
    decision = dataset.decisions[index]
    waveform = dataset._decoded_audio(
        decision.audio_path, decision.parquet_row_group, decision.parquet_row_index
    )
    waveform = waveform[: round(decision.silence_end_s * dataset.sample_rate)]
    output = BytesIO()
    sf.write(output, waveform, dataset.sample_rate, format="WAV", subtype="PCM_16")
    return output.getvalue()


def main() -> None:
    args = parse_args()
    config = read_config(args.config)
    dataset_config = config["dataset"]
    dataset = MultilingualEvaluationDataset(
        dataset_config["index_path"], languages=dataset_config["languages"],
        partition=dataset_config["evaluation_partition"],
        window_seconds=dataset_config["window_seconds"],
        sample_rate=dataset_config["sample_rate"],
        normalize_waveform=bool(dataset_config.get("normalize_waveform", False)),
    )
    selected = {}
    for index, decision in enumerate(dataset.decisions):
        selected.setdefault(decision.language, index)
    if set(selected) != set(dataset_config["languages"]):
        raise RuntimeError("parity sample does not cover every language")

    model, _ = load_model_checkpoint(args.training_checkpoint, device="cpu")
    model.eval()
    predictor = Predictor(
        args.serving_checkpoint, args.metadata, model_version="multilingual-parity",
        ort_intra_op_threads=2, ort_inter_op_threads=1,
    )
    app = create_app(
        args.serving_checkpoint, args.metadata, model_version="multilingual-parity",
        ort_intra_op_threads=2, ort_inter_op_threads=1,
    )
    records = []
    with TestClient(app) as client, torch.inference_mode():
        client.get("/ready").raise_for_status()
        for language, index in sorted(selected.items()):
            payload = wav_prefix(dataset, index)
            waveform = decode_audio_window(
                payload, end_seconds=None, window_seconds=predictor.metadata["window_seconds"],
                sample_rate=predictor.metadata["sample_rate"], normalize_waveform=False,
            )
            features = torch.from_numpy(np.asarray(predictor.extractor([waveform])))
            pytorch_score = float(torch.sigmoid(model(features)).reshape(-1)[0])
            direct = predictor.predict(payload)
            response = client.post(
                "/predict", files={"audio": (f"{language}.wav", payload, "audio/wav")}
            )
            response.raise_for_status()
            api = response.json()
            direct_score = float(direct["p_end"])
            api_score = float(api["p_end"])
            records.append(
                {
                    "language": language,
                    "decision_id": dataset.decisions[index].decision_id,
                    "bytes": len(payload),
                    "pytorch_p_end": pytorch_score,
                    "direct_onnx_p_end": direct_score,
                    "api_p_end": api_score,
                    "pytorch_onnx_absolute_difference": abs(pytorch_score - direct_score),
                    "direct_api_absolute_difference": abs(direct_score - api_score),
                    "decision_match": (
                        (pytorch_score >= 0.5)
                        == bool(direct["should_end_turn"])
                        == bool(api["should_end_turn"])
                    ),
                }
            )
    maximum_pytorch_onnx = max(
        record["pytorch_onnx_absolute_difference"] for record in records
    )
    maximum_direct_api = max(record["direct_api_absolute_difference"] for record in records)
    mismatches = sum(not record["decision_match"] for record in records)
    passed = (
        maximum_pytorch_onnx <= args.tolerance
        and maximum_direct_api <= args.tolerance
        and mismatches == 0
    )
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "passed": passed,
        "tolerance": args.tolerance,
        "threshold": predictor.threshold,
        "languages": dataset_config["languages"],
        "summary": {
            "samples": len(records),
            "maximum_pytorch_onnx_absolute_difference": maximum_pytorch_onnx,
            "maximum_direct_api_absolute_difference": maximum_direct_api,
            "decision_mismatches": mismatches,
        },
        "records": records,
        "artifacts": {
            "training_checkpoint_sha256": file_sha256(args.training_checkpoint),
            "serving_checkpoint_sha256": file_sha256(args.serving_checkpoint),
            "metadata_sha256": file_sha256(args.metadata),
            "canonical_index_sha256": file_sha256(Path(dataset_config["index_path"])),
        },
        "preprocessing": {
            "sample_rate": predictor.metadata["sample_rate"],
            "channels": predictor.metadata["channels"],
            "window_seconds": predictor.metadata["window_seconds"],
            "padding": predictor.metadata["padding"],
            "truncation": predictor.metadata["truncation"],
            "waveform_normalization": predictor.waveform_normalization,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
