"""Export a selected checkpoint to verified FP32 and INT8 ONNX models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from turn_detector.data_loader import CanonicalPipecatDataset
from turn_detector.models import WaveformFeatureExtractor, load_model_checkpoint
from turn_detector.training.train import FeatureCollator, file_sha256, read_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/english_whisper_tiny_v1.json"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


class ProbabilityModel(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.model(input_features))


class CalibrationReader:
    def __init__(self, loader: DataLoader):
        self.loader = loader
        self.iterator = iter(loader)

    def get_next(self):
        try:
            batch = next(self.iterator)
        except StopIteration:
            return None
        return {"input_features": batch["input_features"].numpy()}

    def rewind(self):
        self.iterator = iter(self.loader)


def inference_session(path: Path):
    import onnxruntime as ort

    return ort.InferenceSession(path.as_posix(), providers=["CPUExecutionProvider"])


def verify_export(model, session, batches, tolerance: float) -> dict[str, Any]:
    maximum_difference = 0.0
    decisions = 0
    mismatches = 0
    with torch.inference_mode():
        for features in batches:
            expected = torch.sigmoid(model(features)).numpy()
            actual = session.run(None, {"input_features": features.numpy()})[0]
            maximum_difference = max(maximum_difference, float(np.max(np.abs(expected - actual))))
            mismatches += int(np.sum((expected >= 0.5) != (actual >= 0.5)))
            decisions += len(expected)
    return {
        "records": decisions,
        "maximum_absolute_difference": maximum_difference,
        "decisions_different_at_0_5": mismatches,
        "within_tolerance": maximum_difference <= tolerance,
        "tolerance": tolerance,
    }


def main() -> None:
    args = parse_args()
    config = read_config(args.config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model, payload = load_model_checkpoint(args.checkpoint, device="cpu")
    model.eval()
    wrapper = ProbabilityModel(model).eval()
    dataset_config = config["dataset"]
    extractor = WaveformFeatureExtractor(
        config["model"]["backbone"],
        dataset_config["window_seconds"],
        dataset_config["sample_rate"],
        config["model"].get("backbone_revision"),
        True,
    )
    preprocessing_dir = args.output_dir / "preprocessor"
    extractor.save_pretrained(preprocessing_dir.as_posix())
    dummy = torch.zeros(
        1,
        80,
        round(dataset_config["window_seconds"] * 100),
        dtype=torch.float32,
    )
    fp32_path = args.output_dir / "model.fp32.onnx"
    torch.onnx.export(
        wrapper,
        (dummy,),
        fp32_path,
        input_names=["input_features"],
        output_names=["end_probability"],
        dynamic_axes={"input_features": {0: "batch"}, "end_probability": {0: "batch"}},
        opset_version=config["export"]["opset"],
        do_constant_folding=True,
        dynamo=False,
    )

    validation = CanonicalPipecatDataset(
        dataset_config["index_path"],
        split="validation",
        language=dataset_config["language"],
        window_seconds=dataset_config["window_seconds"],
        sample_rate=dataset_config["sample_rate"],
        shuffle=False,
        max_examples=16,
        normalize_waveform=bool(dataset_config.get("normalize_waveform", False)),
    )
    validation_loader = DataLoader(
        validation,
        batch_size=4,
        num_workers=0,
        collate_fn=FeatureCollator(
            config["model"]["backbone"],
            dataset_config["window_seconds"],
            dataset_config["sample_rate"],
            config["model"].get("backbone_revision"),
            True,
        ),
    )
    validation_batches = [batch["input_features"] for batch in validation_loader]
    fp32_parity = verify_export(model, inference_session(fp32_path), validation_batches, 1e-4)
    if not fp32_parity["within_tolerance"]:
        raise RuntimeError(f"FP32 ONNX parity failed: {fp32_parity}")

    calibration = CanonicalPipecatDataset(
        dataset_config["index_path"],
        split="train",
        language=dataset_config["language"],
        window_seconds=dataset_config["window_seconds"],
        sample_rate=dataset_config["sample_rate"],
        shuffle=True,
        shuffle_buffer=256,
        seed=config["training"]["seed"],
        max_examples=config["export"]["calibration_examples"],
        normalize_waveform=bool(dataset_config.get("normalize_waveform", False)),
    )
    calibration_loader = DataLoader(
        calibration,
        batch_size=8,
        num_workers=0,
        collate_fn=FeatureCollator(
            config["model"]["backbone"],
            dataset_config["window_seconds"],
            dataset_config["sample_rate"],
            config["model"].get("backbone_revision"),
            True,
        ),
    )
    int8_path = args.output_dir / "model.int8.onnx"
    from onnxruntime.quantization import CalibrationMethod, QuantFormat, QuantType, quantize_static

    quantize_static(
        fp32_path.as_posix(),
        int8_path.as_posix(),
        CalibrationReader(calibration_loader),
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
        calibrate_method=CalibrationMethod.MinMax,
    )
    int8_parity = verify_export(model, inference_session(int8_path), validation_batches, 0.08)
    metadata = {
        "format_version": 1,
        "backbone": config["model"]["backbone"],
        "backbone_revision": config["model"].get("backbone_revision"),
        "feature_extractor_path": preprocessing_dir.name,
        "sample_rate": dataset_config["sample_rate"],
        "channels": 1,
        "window_seconds": dataset_config["window_seconds"],
        "padding": "left_zero",
        "truncation": "latest_samples",
        "waveform_normalization": (
            "zero_mean_unit_variance_before_padding"
            if dataset_config.get("normalize_waveform", False)
            else "none"
        ),
        "output": "end_probability",
        "threshold": float(payload["threshold"]),
        "recommended_artifact": "fp32",
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "artifacts": {
            "fp32": {
                "path": fp32_path.name,
                "sha256": file_sha256(fp32_path),
                "bytes": fp32_path.stat().st_size,
                "parity": fp32_parity,
            },
            "int8": {
                "path": int8_path.name,
                "sha256": file_sha256(int8_path),
                "bytes": int8_path.stat().st_size,
                "parity": int8_parity,
                "serving_eligible": bool(int8_parity["within_tolerance"]),
            },
        },
    }
    (args.output_dir / "model_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
