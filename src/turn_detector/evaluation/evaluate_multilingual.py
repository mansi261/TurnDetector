"""Compare the English baseline and multilingual candidate by causal replay."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import gzip
import json
import math
from pathlib import Path
import sqlite3
import statistics
import time
from typing import Any

import numpy as np
import torch

from turn_detector.data_loader import MultilingualEvaluationDataset
from turn_detector.evaluation.evaluate import pause_bucket, replay_times
from turn_detector.evaluation.metrics import calibration, score_distribution
from turn_detector.models import WaveformFeatureExtractor, load_model_checkpoint
from turn_detector.training.train import binary_metrics, read_config, select_device
from turn_detector.utils import file_sha256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/multilingual_whisper_tiny_v1.json")
    )
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--candidate-checkpoint", type=Path, required=True)
    parser.add_argument("--partition", choices=("MULTI_evaluation", "MULTI_holdout"), default="MULTI_evaluation")
    parser.add_argument("--allow-holdout", action="store_true")
    parser.add_argument("--freeze-manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def enriched_summary(events: list[dict[str, Any]]) -> dict[str, Any]:
    holds = [event for event in events if event["label"] == "hold"]
    endings = [event for event in events if event["label"] == "end"]
    premature = sum(event["triggered"] for event in holds)
    detected = sum(event["triggered"] for event in endings)
    missed = len(endings) - detected
    true_negative = len(holds) - premature
    emitted = premature + detected
    delays = [
        float(event["detection_delay_s"])
        for event in endings
        if event["detection_delay_s"] is not None
    ]
    return {
        "events": len(events),
        "hold_events": len(holds),
        "end_events": len(endings),
        "true_negative_holds": true_negative,
        "premature_responses": premature,
        "premature_response_rate": premature / max(1, len(holds)),
        "detected_endings": detected,
        "missed_endings": missed,
        "missed_ending_rate": missed / max(1, len(endings)),
        "end_precision": detected / max(1, emitted),
        "end_recall": detected / max(1, len(endings)),
        "event_accuracy": (detected + true_negative) / max(1, len(events)),
        "total_causal_event_errors": premature + missed,
        "total_causal_event_error_rate": (premature + missed) / max(1, len(events)),
        "emitted_responses": emitted,
        "premature_fraction_of_emitted": premature / max(1, emitted),
        "detected_end_delay_median_s": statistics.median(delays) if delays else None,
        "detected_end_delay_p95_s": float(np.percentile(delays, 95)) if delays else None,
        "repeat_trigger_opportunities": sum(
            int(event["positive_frames_after_first"]) for event in events
        ),
    }


def breakdown(events: list[dict[str, Any]], key: str) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        grouped[str(event[key])].append(event)
    return {name: enriched_summary(items) for name, items in sorted(grouped.items())}


def macro_language(by_language: dict[str, dict[str, Any]]) -> dict[str, float]:
    metrics = (
        "premature_response_rate", "missed_ending_rate", "end_precision",
        "end_recall", "event_accuracy", "total_causal_event_error_rate",
        "detected_end_delay_median_s", "detected_end_delay_p95_s",
    )
    return {
        metric: float(
            np.mean(
                [values[metric] for values in by_language.values() if values[metric] is not None]
            )
        )
        for metric in metrics
    }


def model_report(
    events: list[dict[str, Any]], labels: np.ndarray, full_span_scores: np.ndarray
) -> dict[str, Any]:
    by_language = breakdown(events, "language")
    result = {
        "causal_micro": enriched_summary(events),
        "causal_macro_language": macro_language(by_language),
        "causal_by_language": by_language,
        "causal_by_pause_duration": breakdown(events, "pause_bucket"),
    }
    full_span = binary_metrics(labels, full_span_scores, 0.5)
    full_span["score_distribution"] = score_distribution(labels, full_span_scores)
    full_span["calibration"] = calibration(labels, full_span_scores)
    result["full_span_micro"] = full_span
    full_by_language = {}
    event_languages = np.asarray([event["language"] for event in events])
    for language in sorted(set(event_languages.tolist())):
        mask = event_languages == language
        values = binary_metrics(labels[mask], full_span_scores[mask], 0.5)
        values["score_distribution"] = score_distribution(
            labels[mask], full_span_scores[mask]
        )
        values["calibration"] = calibration(labels[mask], full_span_scores[mask])
        full_by_language[language] = values
    result["full_span_by_language"] = full_by_language
    return result


def fixed_silence(dataset: MultilingualEvaluationDataset, seconds: float) -> dict[str, Any]:
    events = []
    for decision in dataset.decisions:
        duration = decision.silence_end_s - decision.silence_start_s
        events.append(
            {
                "label": decision.label,
                "language": decision.language,
                "pause_bucket": pause_bucket(duration),
                "triggered": duration >= seconds,
                "detection_delay_s": seconds if duration >= seconds else None,
                "positive_frames_after_first": 0,
            }
        )
    by_language = breakdown(events, "language")
    return {
        "silence_threshold_s": seconds,
        "micro": enriched_summary(events),
        "macro_language": macro_language(by_language),
        "by_language": by_language,
        "by_pause_duration": breakdown(events, "pause_bucket"),
    }


def replay_models(
    dataset: MultilingualEvaluationDataset,
    models: dict[str, Any],
    extractor,
    device,
    *,
    threshold: float,
    initial: float,
    step: float,
    batch_size: int,
) -> dict[str, list[dict[str, Any]]]:
    scores: dict[str, dict[int, list[tuple[float, float]]]] = {
        name: defaultdict(list) for name in models
    }
    waveforms = []
    locations: list[tuple[int, float]] = []

    def flush() -> None:
        if not waveforms:
            return
        features = extractor(waveforms).to(device)
        with torch.inference_mode():
            for name, model in models.items():
                probabilities = torch.sigmoid(model(features)).cpu().numpy()
                for (event_index, timestamp), probability in zip(locations, probabilities):
                    scores[name][event_index].append((timestamp, float(probability)))
        waveforms.clear()
        locations.clear()

    for index, decision in enumerate(dataset.decisions):
        for timestamp in replay_times(
            decision.silence_start_s, decision.silence_end_s, initial, step
        ):
            waveforms.append(dataset.waveform_at(index, timestamp))
            locations.append((index, timestamp))
            if len(waveforms) >= batch_size:
                flush()
    flush()

    outputs = {}
    for name in models:
        events = []
        for index, decision in enumerate(dataset.decisions):
            timeline = scores[name][index]
            positives = [(timestamp, score) for timestamp, score in timeline if score >= threshold]
            first_trigger = positives[0][0] if positives else None
            duration = decision.silence_end_s - decision.silence_start_s
            events.append(
                {
                    "decision_id": decision.decision_id,
                    "source_id": decision.source_id,
                    "group_id": decision.group_id,
                    "language": decision.language,
                    "label": decision.label,
                    "pause_duration_s": duration,
                    "pause_bucket": pause_bucket(duration),
                    "speech_source": decision.provenance,
                    "triggered": bool(positives),
                    "first_trigger_s": first_trigger,
                    "detection_delay_s": (
                        first_trigger - decision.silence_start_s
                        if first_trigger is not None
                        else None
                    ),
                    "positive_frames_after_first": max(0, len(positives) - 1),
                    "maximum_p_end": max(score for _, score in timeline),
                    "full_span_p_end": timeline[-1][1],
                    "timeline": [
                        {"timestamp_s": timestamp, "end_probability": score}
                        for timestamp, score in timeline
                    ],
                }
            )
        outputs[name] = events
    return outputs


def comparison(
    baseline: dict[str, Any], candidate: dict[str, Any], criteria: dict[str, Any]
) -> dict[str, Any]:
    left = baseline["causal_macro_language"]
    right = candidate["causal_macro_language"]
    deltas = {metric: right[metric] - left[metric] for metric in right}
    language_error_deltas = {
        language: candidate["causal_by_language"][language]["total_causal_event_error_rate"]
        - baseline["causal_by_language"][language]["total_causal_event_error_rate"]
        for language in baseline["causal_by_language"]
    }
    checks = {
        "lower_macro_causal_error": deltas["total_causal_event_error_rate"]
        <= criteria["macro_causal_event_error_delta_maximum"],
        "macro_premature_guardrail": deltas["premature_response_rate"]
        <= criteria["macro_premature_response_rate_delta_maximum"],
        "macro_missed_guardrail": deltas["missed_ending_rate"]
        <= criteria["macro_missed_ending_rate_delta_maximum"],
        "worst_language_error_guardrail": max(language_error_deltas.values())
        <= criteria["worst_language_causal_error_regression_maximum"],
    }
    return {
        "candidate_minus_baseline_macro": deltas,
        "candidate_minus_baseline_language_error": language_error_deltas,
        "development_quality_checks": checks,
        "development_quality_acceptable": all(checks.values()),
        "serving_checks_pending": True,
    }


def main() -> None:
    args = parse_args()
    if args.partition == "MULTI_holdout" and not args.allow_holdout:
        raise RuntimeError("holdout evaluation requires --allow-holdout")
    if args.partition == "MULTI_holdout" and args.freeze_manifest is None:
        raise RuntimeError("holdout evaluation requires --freeze-manifest")
    config = read_config(args.config)
    if args.freeze_manifest is not None:
        frozen = json.loads(args.freeze_manifest.read_text(encoding="utf-8"))
        checks = {
            "config": file_sha256(args.config),
            "baseline_checkpoint": file_sha256(args.baseline_checkpoint),
            "candidate_checkpoint": file_sha256(args.candidate_checkpoint),
            "scoring_code": file_sha256(Path(__file__)),
        }
        if checks != frozen["sha256"]:
            raise RuntimeError("current holdout inputs differ from the frozen manifest")
        if frozen["protocol"] != {
            "threshold": 0.5,
            "first_score_offset_seconds": config["evaluation"]["first_score_offset_seconds"],
            "replay_step_seconds": config["evaluation"]["replay_step_seconds"],
            "dispatch": "first threshold crossing per annotated pause, latched once",
        }:
            raise RuntimeError("current holdout protocol differs from the frozen manifest")
    device = select_device(args.device)
    baseline, baseline_payload = load_model_checkpoint(args.baseline_checkpoint, device=device)
    candidate, candidate_payload = load_model_checkpoint(args.candidate_checkpoint, device=device)
    for model in (baseline, candidate):
        model.eval()
    if float(baseline_payload["threshold"]) != 0.5 or float(candidate_payload["threshold"]) != 0.5:
        raise RuntimeError("the frozen evaluation protocol requires threshold 0.5")
    if baseline_payload["model_config"] != candidate_payload["model_config"]:
        raise RuntimeError("baseline and candidate architectures differ")

    dataset_config = config["dataset"]
    dataset = MultilingualEvaluationDataset(
        dataset_config["index_path"], languages=dataset_config["languages"],
        partition=args.partition, window_seconds=dataset_config["window_seconds"],
        sample_rate=dataset_config["sample_rate"],
        normalize_waveform=bool(dataset_config.get("normalize_waveform", False)),
    )
    extractor = WaveformFeatureExtractor(
        config["model"]["backbone"], dataset_config["window_seconds"],
        dataset_config["sample_rate"], config["model"].get("backbone_revision"), True,
    )
    protocol = config["evaluation"]
    started = time.perf_counter()
    events = replay_models(
        dataset, {"baseline_v0": baseline, "multilingual_v1": candidate},
        extractor, device, threshold=0.5,
        initial=protocol["first_score_offset_seconds"],
        step=protocol["replay_step_seconds"], batch_size=args.batch_size,
    )
    elapsed = time.perf_counter() - started
    labels = np.asarray([decision.label_id for decision in dataset.decisions], dtype=np.int64)
    reports = {
        name: model_report(
            model_events, labels,
            np.asarray([event["full_span_p_end"] for event in model_events]),
        )
        for name, model_events in events.items()
    }
    with sqlite3.connect(dataset_config["index_path"]) as connection:
        index_metadata = dict(connection.execute("SELECT key, value FROM index_metadata"))
    result = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "partition": args.partition,
        "holdout_access_authorized": bool(args.allow_holdout),
        "dataset_revision": index_metadata["livekit_revision"],
        "canonical_index_sha256": file_sha256(Path(dataset_config["index_path"])),
        "scoring_code_sha256": file_sha256(Path(__file__)),
        "checkpoints": {
            "baseline_v0": {
                "path": args.baseline_checkpoint.resolve().as_posix(),
                "sha256": file_sha256(args.baseline_checkpoint),
            },
            "multilingual_v1": {
                "path": args.candidate_checkpoint.resolve().as_posix(),
                "sha256": file_sha256(args.candidate_checkpoint),
            },
        },
        "runtime": {
            "device": str(device),
            "elapsed_seconds": elapsed,
            "events_per_second_for_two_models": len(dataset) / elapsed,
        },
        "counts": {
            "events": len(dataset),
            "turns": len({decision.source_id for decision in dataset.decisions}),
            "conversation_groups": len({decision.group_id for decision in dataset.decisions}),
            "hold": int(np.sum(labels == 0)),
            "end": int(np.sum(labels == 1)),
            "by_language": {
                language: sum(decision.language == language for decision in dataset.decisions)
                for language in dataset_config["languages"]
            },
        },
        "protocol": {
            "causal_audio_only": True,
            "window_seconds": dataset_config["window_seconds"],
            "sample_rate_hz": dataset_config["sample_rate"],
            "waveform_normalization": dataset_config.get("normalize_waveform", False),
            "threshold": 0.5,
            "first_score_offset_seconds": protocol["first_score_offset_seconds"],
            "replay_step_seconds": protocol["replay_step_seconds"],
            "dispatch": "first threshold crossing per annotated pause, latched once",
            "detection_deadline": "end of annotated silence span",
            "event_time_delay_excludes_compute_and_network_latency": True,
        },
        "models": reports,
        "comparison": comparison(
            reports["baseline_v0"], reports["multilingual_v1"],
            config["objective"]["promotion_criteria"],
        ),
        "fixed_silence_baselines": {
            str(seconds): fixed_silence(dataset, float(seconds))
            for seconds in protocol["fixed_silence_seconds"]
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    events_path = args.output.with_name(args.output.stem + "_events.json.gz")
    with gzip.open(events_path, "wt", encoding="utf-8") as stream:
        json.dump(events, stream, separators=(",", ":"))
    print(
        json.dumps(
            {
                "partition": args.partition,
                "runtime": result["runtime"],
                "baseline": reports["baseline_v0"]["causal_macro_language"],
                "candidate": reports["multilingual_v1"]["causal_macro_language"],
                "comparison": result["comparison"],
                "report": args.output.as_posix(),
                "events": events_path.as_posix(),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
