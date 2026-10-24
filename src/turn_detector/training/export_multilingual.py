"""Export and verify the multilingual candidate as FP32 ONNX."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from turn_detector.models import WaveformFeatureExtractor, load_model_checkpoint
from turn_detector.training.train import read_config
from turn_detector.training.train_multilingual import (
    MultilingualFeatureCollator,
    make_dataset,
    make_loader,
)
from turn_detector.utils import file_sha256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/multilingual_whisper_tiny_v1.json")
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("model/multilingual_v1"))
    return parser.parse_args()


class ProbabilityModel(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.model(input_features))


def session(path: Path):
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(
        path.as_posix(), sess_options=options, providers=["CPUExecutionProvider"]
    )


def verify(model, runtime, loader, tolerance: float) -> dict[str, Any]:
    maximum = 0.0
    mismatches = 0
    records = 0
    by_language: dict[str, dict[str, Any]] = {}
    with torch.inference_mode():
        for batch in loader:
            features = batch["input_features"]
            expected = torch.sigmoid(model(features)).numpy()
            actual = runtime.run(None, {"input_features": features.numpy()})[0]
            for language, left, right in zip(
                batch["languages"], expected.tolist(), actual.tolist(), strict=True
            ):
                difference = abs(float(left) - float(right))
                values = by_language.setdefault(
                    language,
                    {"records": 0, "maximum_absolute_difference": 0.0, "decision_mismatches": 0},
                )
                values["records"] += 1
                values["maximum_absolute_difference"] = max(
                    values["maximum_absolute_difference"], difference
                )
                values["decision_mismatches"] += int((left >= 0.5) != (right >= 0.5))
                maximum = max(maximum, difference)
                mismatches += int((left >= 0.5) != (right >= 0.5))
                records += 1
    return {
        "records": records,
        "maximum_absolute_difference": maximum,
        "decision_mismatches_at_0_5": mismatches,
        "within_tolerance": maximum <= tolerance and mismatches == 0,
        "tolerance": tolerance,
        "by_language": by_language,
    }


def main() -> None:
    args = parse_args()
    config = read_config(args.config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model, payload = load_model_checkpoint(args.checkpoint, device="cpu")
    model.eval()
    if float(payload["threshold"]) != float(config["threshold"]["value"]):
        raise RuntimeError("checkpoint threshold differs from export config")
    wrapper = ProbabilityModel(model).eval()
    dataset = config["dataset"]
    extractor = WaveformFeatureExtractor(
        config["model"]["backbone"], dataset["window_seconds"], dataset["sample_rate"],
        config["model"].get("backbone_revision"), True,
    )
    preprocessing_dir = args.output_dir / "preprocessor"
    extractor.save_pretrained(preprocessing_dir.as_posix())
    dummy = torch.zeros(
        1, 80, round(dataset["window_seconds"] * 100), dtype=torch.float32
    )
    onnx_path = args.output_dir / "model.fp32.onnx"
    torch.onnx.export(
        wrapper, (dummy,), onnx_path,
        input_names=["input_features"], output_names=["end_probability"],
        dynamic_axes={"input_features": {0: "batch"}, "end_probability": {0: "batch"}},
        opset_version=config["export"]["opset"], do_constant_folding=True, dynamo=False,
    )

    validation = make_dataset(config, "validation", smoke=True)
    validation_loader = make_loader(
        validation, config, MultilingualFeatureCollator(config), "validation"
    )
    tolerance = config["objective"]["promotion_criteria"][
        "onnx_probability_maximum_absolute_difference"
    ]
    parity = verify(model, session(onnx_path), validation_loader, tolerance)
    if not parity["within_tolerance"]:
        raise RuntimeError(f"ONNX parity failed: {parity}")
    metadata = {
        "format_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model_version": "multilingual_whisper_tiny_v1",
        "language_scope": dataset["languages"],
        "backbone": config["model"]["backbone"],
        "backbone_revision": config["model"].get("backbone_revision"),
        "feature_extractor_path": preprocessing_dir.name,
        "sample_rate": dataset["sample_rate"],
        "channels": 1,
        "window_seconds": dataset["window_seconds"],
        "padding": "left_zero",
        "truncation": "latest_samples",
        "waveform_normalization": "none",
        "output": "end_probability",
        "threshold": float(payload["threshold"]),
        "recommended_artifact": "fp32",
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "artifacts": {
            "fp32": {
                "path": onnx_path.name,
                "sha256": file_sha256(onnx_path),
                "bytes": onnx_path.stat().st_size,
                "parity": parity,
                "serving_eligible": True,
            }
        },
    }
    metadata_path = args.output_dir / "model_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
