"""Tune the decision threshold from saved causal development predictions."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import statistics
import sys
from typing import Any

import numpy as np

from turn_detector.training.train import binary_metrics
from turn_detector.utils import file_sha256, git_value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/english_threshold_optimization_v1.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/english_whisper_tiny_v1/threshold_optimization_v1"),
    )
    return parser.parse_args()


def threshold_grid(search: dict[str, Any]) -> list[float]:
    minimum = float(search["minimum"])
    maximum = float(search["maximum"])
    step = float(search["step"])
    count = round((maximum - minimum) / step)
    return [round(minimum + index * step, 6) for index in range(count + 1)]


def rescore_event_records(
    events: list[dict[str, Any]], threshold: float
) -> list[dict[str, Any]]:
    rescored = []
    for event in events:
        positives = [
            point
            for point in event["timeline"]
            if float(point["end_probability"]) >= threshold
        ]
        first_trigger = float(positives[0]["timestamp_s"]) if positives else None
        rescored.append(
            {
                "decision_id": event["decision_id"],
                "source_id": event["source_id"],
                "group_id": event["group_id"],
                "label": event["label"],
                "pause_duration_s": event["pause_duration_s"],
                "pause_bucket": event["pause_bucket"],
                "speech_source": event["speech_source"],
                "triggered": bool(positives),
                "first_trigger_s": first_trigger,
                "detection_delay_s": (
                    first_trigger - float(event["timeline"][0]["timestamp_s"])
                    if first_trigger is not None
                    else None
                ),
                "positive_frames_after_first": max(0, len(positives) - 1),
            }
        )

    return rescored


def summarize_event_records(
    events: list[dict[str, Any]], *, include_breakdowns: bool
) -> dict[str, Any]:
    holds = [event for event in events if event["label"] == "hold"]
    endings = [event for event in events if event["label"] == "end"]
    delays = [
        float(event["detection_delay_s"])
        for event in endings
        if event["detection_delay_s"] is not None
    ]
    premature = sum(bool(event["triggered"]) for event in holds)
    detected = sum(bool(event["triggered"]) for event in endings)
    missed = len(endings) - detected
    emitted = premature + detected
    summary = {
        "events": len(events),
        "hold_events": len(holds),
        "end_events": len(endings),
        "premature_responses": premature,
        "premature_response_rate": premature / max(1, len(holds)),
        "detected_endings": detected,
        "missed_endings": missed,
        "missed_ending_rate": missed / max(1, len(endings)),
        "total_causal_event_errors": premature + missed,
        "premature_fraction_of_emitted": premature / max(1, emitted),
        "detected_end_delay_median_s": statistics.median(delays) if delays else None,
        "detected_end_delay_p95_s": float(np.percentile(delays, 95)) if delays else None,
    }
    if include_breakdowns:
        for output_key, event_key in (
            ("by_pause_duration", "pause_bucket"),
            ("by_speech_source", "speech_source"),
        ):
            grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for event in events:
                grouped[str(event[event_key])].append(event)
            summary[output_key] = {
                name: summarize_event_records(items, include_breakdowns=False)
                for name, items in sorted(grouped.items())
            }
    return summary


def rescore_events(events: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    summary = summarize_event_records(
        rescore_event_records(events, threshold), include_breakdowns=True
    )
    summary["threshold"] = threshold
    return summary


def add_full_span_metrics(
    result: dict[str, Any], labels: np.ndarray, probabilities: np.ndarray
) -> None:
    metrics = binary_metrics(labels, probabilities, float(result["threshold"]))
    result["full_span"] = metrics


def selection_key(result: dict[str, Any]) -> tuple[float, ...]:
    delay = result["detected_end_delay_median_s"]
    return (
        float(result["total_causal_event_errors"]),
        float(result["premature_responses"]),
        float(result["missed_endings"]),
        float("inf") if delay is None else float(delay),
    )


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    config = json.loads(config_path.read_text(encoding="utf-8"))
    baseline_config = config["baseline"]
    report_path = Path(baseline_config["causal_report"])
    events_path = Path(baseline_config["causal_events"])
    predictions_path = Path(baseline_config["full_span_predictions"])
    checkpoint_path = Path(baseline_config["checkpoint"])
    source_report = json.loads(report_path.read_text(encoding="utf-8"))
    if source_report["evaluation_partition"] != config["data_policy"]["tuning_partition"]:
        raise ValueError("causal report does not match the configured tuning partition")
    if config["data_policy"]["score_holdouts"]:
        raise ValueError("holdout scoring is disabled for threshold optimization")

    events = json.loads(events_path.read_text(encoding="utf-8"))
    arrays = np.load(predictions_path)
    labels = arrays["labels"]
    probabilities = arrays["probabilities"]
    results = []
    for threshold in threshold_grid(config["search"]):
        result = rescore_events(events, threshold)
        add_full_span_metrics(result, labels, probabilities)
        results.append(result)

    baseline_threshold = float(baseline_config["threshold"])
    baseline = next(result for result in results if result["threshold"] == baseline_threshold)
    candidate = min(results, key=selection_key)
    comparison_fields = (
        "total_causal_event_errors",
        "premature_responses",
        "premature_response_rate",
        "missed_endings",
        "missed_ending_rate",
        "detected_end_delay_median_s",
        "detected_end_delay_p95_s",
    )
    deltas = {
        field: (
            None
            if baseline[field] is None or candidate[field] is None
            else float(candidate[field]) - float(baseline[field])
        )
        for field in comparison_fields
    }
    comparison = {
        "baseline": baseline,
        "candidate": candidate,
        "candidate_minus_baseline": deltas,
        "success_criterion_met": (
            candidate["total_causal_event_errors"]
            < baseline["total_causal_event_errors"]
        ),
    }
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "invocation": [str(Path(__file__).resolve()), "--config", str(config_path), "--output-dir", str(output_dir)],
        "config": config,
        "config_sha256": file_sha256(config_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "causal_report_sha256": file_sha256(report_path),
        "causal_events_sha256": file_sha256(events_path),
        "full_span_predictions_sha256": file_sha256(predictions_path),
        "scoring_code_sha256": file_sha256(Path(__file__)),
        "git_head": git_value("rev-parse", "HEAD"),
        "git_diff_sha256": hashlib.sha256(
            (git_value("diff", "--no-ext-diff", "HEAD") or "").encode()
        ).hexdigest(),
        "dataset_revision": source_report["dataset_revision"],
        "canonical_schema_version": source_report["canonical_schema_version"],
        "partition": source_report["evaluation_partition"],
        "checkpoint": checkpoint_path.resolve().as_posix(),
        "thresholds": threshold_grid(config["search"]),
        "holdouts_scored": [],
        "execution": "Cached causal probabilities were rescored; model inference and training were not run.",
        "randomness": None,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": importlib.metadata.version("numpy"),
        },
    }
    policy = {
        "model_version": "english-whisper-tiny-baseline-v0-threshold-v1",
        "checkpoint": checkpoint_path.resolve().as_posix(),
        "checkpoint_sha256": manifest["checkpoint_sha256"],
        "threshold": candidate["threshold"],
        "selection_partition": manifest["partition"],
        "selection_metric": config["search"]["primary_metric"],
        "holdouts_scored": [],
    }
    candidate_events = rescore_event_records(events, float(candidate["threshold"]))
    candidate_report = {
        "created_utc": manifest["created_utc"],
        "language": source_report["language"],
        "language_label": source_report["language_label"],
        "evaluation_partition": manifest["partition"],
        "dataset_revision": manifest["dataset_revision"],
        "canonical_schema_version": manifest["canonical_schema_version"],
        "checkpoint": manifest["checkpoint"],
        "checkpoint_sha256": manifest["checkpoint_sha256"],
        "threshold": candidate["threshold"],
        "threshold_selection": {
            "partition": manifest["partition"],
            "metric": config["search"]["primary_metric"],
            "candidate_count": len(results),
        },
        "protocol": {
            **source_report["protocol"],
            "threshold": candidate["threshold"],
        },
        "causal_replay": {
            key: value for key, value in candidate.items() if key != "full_span"
        },
        "full_span_metrics": candidate["full_span"],
        "baseline_comparison": comparison,
        "holdouts_scored": [],
    }

    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "sweep_results.json").write_text(
        json.dumps({"results": results, "comparison": comparison}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "selected_policy.json").write_text(
        json.dumps(policy, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "candidate_development_report.json").write_text(
        json.dumps(candidate_report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "candidate_development_events.json").write_text(
        json.dumps(candidate_events, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(comparison, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
