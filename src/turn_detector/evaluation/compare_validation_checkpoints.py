"""Compare two checkpoints on the fixed English validation split."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sqlite3
import subprocess
import sys
import time
from typing import Any

import numpy as np
import torch
from torch import nn

from turn_detector.models import load_model_checkpoint
from turn_detector.training.train import (
    FeatureCollator,
    binary_metrics,
    file_sha256,
    make_loader,
    read_config,
    select_device,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path("experiments/english_whisper_tiny_v1")
    parser.add_argument(
        "--config", type=Path, default=Path("configs/english_whisper_tiny_v1.json")
    )
    parser.add_argument(
        "--baseline-checkpoint",
        type=Path,
        default=root / "full_interrupted_drop_last/checkpoints/best.pt",
    )
    parser.add_argument(
        "--candidate-checkpoint",
        type=Path,
        default=root / "full_three_epoch_v1/checkpoints/latest.pt",
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument(
        "--output-dir", type=Path, default=root / "validation_comparison_v1"
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        default=Path("reports/ENG_epoch3_validation_comparison_v1.md"),
    )
    return parser.parse_args()


def command_output(*command: str) -> str | None:
    try:
        return subprocess.run(
            command, check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def calibration(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    labels_float = labels.astype(np.float64)
    clipped = np.clip(probabilities.astype(np.float64), 1e-7, 1 - 1e-7)
    bins = []
    ece = 0.0
    for index in range(10):
        lower = index / 10
        upper = (index + 1) / 10
        mask = (probabilities >= lower) & (
            probabilities <= upper if index == 9 else probabilities < upper
        )
        count = int(mask.sum())
        if count:
            mean_score = float(probabilities[mask].mean())
            observed_rate = float(labels_float[mask].mean())
            gap = abs(mean_score - observed_rate)
            ece += count / len(labels) * gap
        else:
            mean_score = observed_rate = gap = None
        bins.append(
            {
                "range": [lower, upper],
                "count": count,
                "mean_p_end": mean_score,
                "observed_end_rate": observed_rate,
                "absolute_gap": gap,
            }
        )
    return {
        "brier_score": float(np.mean((probabilities - labels_float) ** 2)),
        "log_loss_from_probabilities": float(
            -np.mean(labels_float * np.log(clipped) + (1 - labels_float) * np.log(1 - clipped))
        ),
        "expected_calibration_error_10_bins": float(ece),
        "bins": bins,
    }


def score_distribution(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    def summarize(values: np.ndarray) -> dict[str, float]:
        return {
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "p10": float(np.percentile(values, 10)),
            "p90": float(np.percentile(values, 90)),
        }

    return {
        "hold": summarize(probabilities[labels == 0]),
        "end": summarize(probabilities[labels == 1]),
    }


def evaluate_checkpoint(
    checkpoint: Path,
    loader,
    *,
    threshold: float,
    device: torch.device,
) -> tuple[dict[str, Any], list[str], np.ndarray, np.ndarray]:
    model, payload = load_model_checkpoint(checkpoint, device=device)
    model.eval()
    criterion = nn.BCEWithLogitsLoss(reduction="sum")
    losses = 0.0
    identifiers: list[str] = []
    label_batches = []
    probability_batches = []
    first_batch_checked = False
    started = time.perf_counter()
    with torch.inference_mode():
        for batch in loader:
            inputs = batch["input_features"].to(device)
            targets = batch["labels"].to(device)
            logits = model(inputs)
            if not first_batch_checked:
                if logits.shape != targets.shape or not torch.isfinite(logits).all():
                    raise RuntimeError("checkpoint inference smoke check failed")
                first_batch_checked = True
            losses += float(criterion(logits, targets).cpu())
            identifiers.extend(batch["ids"])
            label_batches.append(targets.cpu().numpy())
            probability_batches.append(torch.sigmoid(logits).cpu().numpy())
    labels = np.concatenate(label_batches)
    probabilities = np.concatenate(probability_batches)
    metrics = binary_metrics(labels, probabilities, threshold)
    metrics.update(
        {
            "loss": losses / len(labels),
            "records": len(labels),
            "calibration": calibration(labels, probabilities),
            "score_distribution": score_distribution(labels, probabilities),
        }
    )
    record = {
        "checkpoint": checkpoint.resolve().as_posix(),
        "checkpoint_sha256": file_sha256(checkpoint),
        "checkpoint_epoch": int(payload["training_state"]["epoch"]),
        "stored_threshold": float(payload["threshold"]),
        "evaluation_threshold": threshold,
        "model_config": payload["model_config"],
        "metrics": metrics,
        "elapsed_seconds": time.perf_counter() - started,
        "inference_smoke_check": {
            "passed": first_batch_checked,
            "mode": "model.eval with torch.inference_mode",
        },
    }
    del model
    if device.type == "mps":
        torch.mps.empty_cache()
    return record, identifiers, labels, probabilities


def transition_counts(
    labels: np.ndarray,
    baseline_probabilities: np.ndarray,
    candidate_probabilities: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    baseline = baseline_probabilities >= threshold
    candidate = candidate_probabilities >= threshold
    holds = labels == 0
    endings = labels == 1
    return {
        "hold": {
            "correct_under_both": int(np.sum(holds & ~baseline & ~candidate)),
            "wrong_under_both": int(np.sum(holds & baseline & candidate)),
            "corrected_by_candidate": int(np.sum(holds & baseline & ~candidate)),
            "regressed_under_candidate": int(np.sum(holds & ~baseline & candidate)),
        },
        "end": {
            "correct_under_both": int(np.sum(endings & baseline & candidate)),
            "wrong_under_both": int(np.sum(endings & ~baseline & ~candidate)),
            "corrected_by_candidate": int(np.sum(endings & ~baseline & candidate)),
            "regressed_under_candidate": int(np.sum(endings & baseline & ~candidate)),
        },
    }


def split_identity(index_path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    labels = {"hold": 0, "end": 0}
    with sqlite3.connect(index_path) as connection:
        rows = connection.execute(
            """
            SELECT decision_id, label FROM decisions
            WHERE split = 'validation' AND language = 'eng'
            ORDER BY decision_id
            """
        )
        records = 0
        for decision_id, label in rows:
            digest.update(decision_id.encode())
            digest.update(b"\n")
            labels[label] += 1
            records += 1
        metadata = dict(connection.execute("SELECT key, value FROM index_metadata"))
    return {
        "split": "validation",
        "language": "eng",
        "records": records,
        "labels": labels,
        "decision_id_sha256": digest.hexdigest(),
        "index_sha256": file_sha256(index_path),
        "pipecat_revision": metadata["pipecat_revision"],
        "canonical_schema_version": int(metadata["schema_version"]),
    }


def metric_delta(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, float]:
    keys = (
        "loss",
        "accuracy",
        "precision",
        "end_recall",
        "f1",
        "auroc",
        "hold_false_positive_rate",
        "true_positive",
        "false_positive",
        "true_negative",
        "false_negative",
    )
    result = {key: float(candidate[key]) - float(baseline[key]) for key in keys}
    for key in ("brier_score", "expected_calibration_error_10_bins"):
        result[key] = float(candidate["calibration"][key]) - float(
            baseline["calibration"][key]
        )
    return result


def percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def percentage_delta(value: float) -> str:
    return f"{value * 100:+.2f} pp"


def markdown_report(report: dict[str, Any]) -> str:
    baseline = report["models"]["baseline_v0"]
    candidate = report["models"]["epoch_3_candidate"]
    left = baseline["metrics"]
    right = candidate["metrics"]
    delta = report["candidate_minus_baseline"]
    transitions = report["prediction_transitions"]
    return f"""# ENG epoch-3 validation comparison v1

## Decision

This is a fixed-threshold internal validation comparison. Both checkpoints were
scored at `0.5`; the automatically calibrated threshold was ignored. Epoch 3
is a development candidate and is not promoted by this report.

## Frozen protocol

- Dataset: Pipecat English `validation` split at revision `{report['dataset']['pipecat_revision']}`.
- Records: {report['dataset']['records']} (`{report['dataset']['labels']['hold']} hold`, `{report['dataset']['labels']['end']} end`).
- Decision-ID hash: `{report['dataset']['decision_id_sha256']}`.
- Input: latest 8 seconds of 16 kHz mono audio, left-zero-padded when shorter.
- Evaluation: `model.eval()` and `torch.inference_mode()`; dropout and gradients disabled.
- Threshold: `0.5` for both checkpoints.
- No LiveKit development or holdout examples were accessed.

## Checkpoints

| Role | Epoch | SHA-256 | Stored threshold | Evaluation threshold |
|---|---:|---|---:|---:|
| Baseline v0 | {baseline['checkpoint_epoch']} | `{baseline['checkpoint_sha256']}` | {baseline['stored_threshold']:.4f} | {baseline['evaluation_threshold']:.1f} |
| Fully fine-tuned candidate | {candidate['checkpoint_epoch']} | `{candidate['checkpoint_sha256']}` | {candidate['stored_threshold']:.4f} | {candidate['evaluation_threshold']:.1f} |

The candidate is `latest.pt` from epoch 3. It is deliberately not described as
the lowest-loss checkpoint: epoch 1 still has the lowest validation loss.

## Results at threshold 0.5

| Metric | Baseline epoch 1 | Candidate epoch 3 | Candidate change |
|---|---:|---:|---:|
| BCE loss | {left['loss']:.5f} | {right['loss']:.5f} | {delta['loss']:+.5f} |
| Accuracy | {percent(left['accuracy'])} | {percent(right['accuracy'])} | {percentage_delta(delta['accuracy'])} |
| End precision | {percent(left['precision'])} | {percent(right['precision'])} | {percentage_delta(delta['precision'])} |
| End recall | {percent(left['end_recall'])} | {percent(right['end_recall'])} | {percentage_delta(delta['end_recall'])} |
| F1 | {percent(left['f1'])} | {percent(right['f1'])} | {percentage_delta(delta['f1'])} |
| AUROC | {percent(left['auroc'])} | {percent(right['auroc'])} | {percentage_delta(delta['auroc'])} |
| Hold false-positive rate | {percent(left['hold_false_positive_rate'])} | {percent(right['hold_false_positive_rate'])} | {percentage_delta(delta['hold_false_positive_rate'])} |
| Brier score | {left['calibration']['brier_score']:.5f} | {right['calibration']['brier_score']:.5f} | {delta['brier_score']:+.5f} |
| Calibration error, 10 bins | {percent(left['calibration']['expected_calibration_error_10_bins'])} | {percent(right['calibration']['expected_calibration_error_10_bins'])} | {percentage_delta(delta['expected_calibration_error_10_bins'])} |

| Actual / predicted | Baseline epoch 1 | Candidate epoch 3 |
|---|---:|---:|
| Hold predicted hold | {left['true_negative']} | {right['true_negative']} |
| Hold predicted end | {left['false_positive']} | {right['false_positive']} |
| End predicted hold | {left['false_negative']} | {right['false_negative']} |
| End predicted end | {left['true_positive']} | {right['true_positive']} |

The candidate corrected {transitions['hold']['corrected_by_candidate']} baseline
hold errors and {transitions['end']['corrected_by_candidate']} baseline end
errors, while introducing {transitions['hold']['regressed_under_candidate']} new
hold errors and {transitions['end']['regressed_under_candidate']} new end errors.

## Interpretation

Epoch 3 improves every threshold-based classification metric at `0.5`, including
accuracy, precision, recall, F1, AUROC, and the hold false-positive rate. Its
probability-quality results are mixed: Brier score improves, while BCE loss and
10-bin calibration error worsen. The candidate reduces average squared error
but has more calibration mismatch and a small number of more confidently wrong
predictions. `p_end` should continue to be treated as a model score rather than
a literal calibrated probability.

This validation split was observed during model development and cannot establish
external turn-taking quality. The candidate must next be compared with Baseline
v0 on the fixed causal `ENG_evaluation_dev` protocol at threshold `0.5`, measuring
premature responses, missed endings, and detection delay. Neither holdout should
be used yet.

## Reproduce

```bash
PYTHONPATH=src .venv/bin/python -m turn_detector.evaluation.compare_validation_checkpoints --device mps
```

Raw artifacts:
`experiments/english_whisper_tiny_v1/validation_comparison_v1/`.
"""


def main() -> None:
    args = parse_args()
    if args.threshold != 0.5:
        raise ValueError("this comparison is frozen at threshold 0.5")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {args.output_dir}")
    config_path = args.config.resolve()
    config = read_config(config_path)
    device = select_device(args.device)
    collator = FeatureCollator(
        config["model"]["backbone"],
        config["dataset"]["window_seconds"],
        config["dataset"]["sample_rate"],
        config["model"].get("backbone_revision"),
        True,
    )
    _, validation_loader = make_loader(
        config, split="validation", collator=collator, smoke=False
    )
    baseline, baseline_ids, labels, baseline_probabilities = evaluate_checkpoint(
        args.baseline_checkpoint,
        validation_loader,
        threshold=args.threshold,
        device=device,
    )
    candidate, candidate_ids, candidate_labels, candidate_probabilities = (
        evaluate_checkpoint(
            args.candidate_checkpoint,
            validation_loader,
            threshold=args.threshold,
            device=device,
        )
    )
    if baseline_ids != candidate_ids or not np.array_equal(labels, candidate_labels):
        raise RuntimeError("checkpoints were not evaluated on identical records")
    if baseline["model_config"] != candidate["model_config"]:
        raise RuntimeError("model configurations differ")
    dataset = split_identity(Path(config["dataset"]["index_path"]))
    if len(labels) != dataset["records"]:
        raise RuntimeError("evaluated record count differs from the canonical index")

    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "development_comparison_only",
        "candidate_promoted": False,
        "threshold": args.threshold,
        "dataset": dataset,
        "models": {
            "baseline_v0": baseline,
            "epoch_3_candidate": candidate,
        },
        "candidate_minus_baseline": metric_delta(
            baseline["metrics"], candidate["metrics"]
        ),
        "prediction_transitions": transition_counts(
            labels, baseline_probabilities, candidate_probabilities, args.threshold
        ),
        "data_access": {
            "training_split_scored": False,
            "validation_split_scored": True,
            "ENG_evaluation_dev_scored": False,
            "holdouts_scored": [],
        },
        "limitations": [
            "Validation informed model development and is not a final test set.",
            "The epoch-3 continuation did not restore the epoch-1 RNG state.",
            "Offline classification does not measure causal turn-taking behavior.",
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = args.output_dir / "predictions.npz"
    np.savez_compressed(
        predictions_path,
        ids=np.asarray(baseline_ids),
        labels=labels,
        baseline_probabilities=baseline_probabilities,
        candidate_probabilities=candidate_probabilities,
    )
    report_path = args.output_dir / "report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest = {
        "created_utc": report["created_utc"],
        "invocation": [sys.executable, *sys.argv],
        "config": config_path.as_posix(),
        "config_sha256": file_sha256(config_path),
        "scoring_code_sha256": file_sha256(Path(__file__)),
        "report_sha256": file_sha256(report_path),
        "predictions_sha256": file_sha256(predictions_path),
        "baseline_checkpoint_sha256": baseline["checkpoint_sha256"],
        "candidate_checkpoint_sha256": candidate["checkpoint_sha256"],
        "dataset": dataset,
        "threshold": args.threshold,
        "device": str(device),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": importlib.metadata.version("torch"),
            "numpy": importlib.metadata.version("numpy"),
            "transformers": importlib.metadata.version("transformers"),
        },
        "git_head": command_output("git", "rev-parse", "HEAD"),
        "git_diff_sha256": hashlib.sha256(
            (command_output("git", "diff", "--no-ext-diff", "HEAD") or "").encode()
        ).hexdigest(),
        "randomness": None,
        "holdouts_scored": [],
    }
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text(markdown_report(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "report": report_path.resolve().as_posix(),
                "markdown": args.markdown.resolve().as_posix(),
                "threshold": args.threshold,
                "candidate_promoted": False,
                "holdouts_scored": [],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
