"""Evaluate a checkpoint on causal pause replays and fixed-silence baselines."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import statistics
from typing import Any

import numpy as np
import torch

from turn_detector.data_loader import CanonicalEvaluationDataset
from turn_detector.models import WaveformFeatureExtractor, load_model_checkpoint
from turn_detector.training.train import binary_metrics, read_config, select_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/english_whisper_tiny_v1.json"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--partition",
        choices=("ENG_evaluation_dev", "ENG_holdout_a", "ENG_holdout_b"),
        default="ENG_evaluation_dev",
    )
    parser.add_argument("--allow-holdout", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    return parser.parse_args()


def pause_bucket(duration: float) -> str:
    if duration < 0.5:
        return "under_500ms"
    if duration < 1.0:
        return "500ms_to_1s"
    return "at_least_1s"


def replay_times(start: float, end: float, initial: float, step: float) -> list[float]:
    if end <= start:
        return [end]
    first = min(end, start + initial)
    count = max(0, math.floor((end - first) / step + 1e-8))
    times = [first + index * step for index in range(count + 1)]
    if times[-1] < end - 1e-6:
        times.append(end)
    return [min(end, value) for value in times]


def predict_waveforms(model, extractor, waveforms, device, batch_size: int) -> np.ndarray:
    results = []
    with torch.inference_mode():
        for start in range(0, len(waveforms), batch_size):
            features = extractor(waveforms[start : start + batch_size]).to(device)
            results.append(torch.sigmoid(model(features)).cpu().numpy())
    return np.concatenate(results) if results else np.empty(0, dtype=np.float32)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def summarize_events(events: list[dict[str, Any]]) -> dict[str, Any]:
    holds = [event for event in events if event["label"] == "hold"]
    endings = [event for event in events if event["label"] == "end"]
    delays = [event["detection_delay_s"] for event in endings if event["detection_delay_s"] is not None]
    premature = sum(event["triggered"] for event in holds)
    detected_endings = sum(event["triggered"] for event in endings)
    emitted = premature + detected_endings
    return {
        "events": len(events),
        "hold_events": len(holds),
        "end_events": len(endings),
        "premature_responses": premature,
        "premature_response_rate": premature / max(1, len(holds)),
        "emitted_responses": emitted,
        "premature_fraction_of_emitted": premature / max(1, emitted),
        "detected_endings": detected_endings,
        "missed_endings": sum(not event["triggered"] for event in endings),
        "missed_ending_rate": sum(not event["triggered"] for event in endings) / max(1, len(endings)),
        "repeat_trigger_opportunities": sum(event["positive_frames_after_first"] for event in events),
        "detected_end_delay_median_s": statistics.median(delays) if delays else None,
        "detected_end_delay_p95_s": float(np.percentile(delays, 95)) if delays else None,
    }


def breakdown(events: list[dict[str, Any]], key: str) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        grouped[str(event[key])].append(event)
    return {name: summarize_events(items) for name, items in sorted(grouped.items())}


def evaluate_model_replay(
    dataset: CanonicalEvaluationDataset,
    model,
    extractor,
    device,
    *,
    threshold: float,
    initial: float,
    step: float,
    batch_size: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    waveforms = []
    locations = []
    by_event: dict[int, list[tuple[float, float]]] = defaultdict(list)

    def flush() -> None:
        if not waveforms:
            return
        probabilities = predict_waveforms(model, extractor, waveforms, device, batch_size)
        for (event_index, timestamp), probability in zip(locations, probabilities):
            by_event[event_index].append((timestamp, float(probability)))
        waveforms.clear()
        locations.clear()

    for index, decision in enumerate(dataset.decisions):
        times = replay_times(decision.silence_start_s, decision.silence_end_s, initial, step)
        for timestamp in times:
            waveforms.append(dataset.waveform_at(index, timestamp))
            locations.append((index, timestamp))
            if len(waveforms) >= batch_size:
                flush()
    flush()

    events = []
    for index, decision in enumerate(dataset.decisions):
        timeline = by_event[index]
        positives = [(timestamp, score) for timestamp, score in timeline if score >= threshold]
        first_trigger = positives[0][0] if positives else None
        duration = decision.silence_end_s - decision.silence_start_s
        events.append(
            {
                "decision_id": decision.decision_id,
                "source_id": decision.source_id,
                "group_id": decision.group_id,
                "label": decision.label,
                "pause_duration_s": duration,
                "pause_bucket": pause_bucket(duration),
                "speech_source": decision.provenance,
                "triggered": bool(positives),
                "first_trigger_s": first_trigger,
                "detection_delay_s": (
                    first_trigger - decision.silence_start_s if first_trigger is not None else None
                ),
                "positive_frames_after_first": max(0, len(positives) - 1),
                "timeline": [
                    {"timestamp_s": timestamp, "end_probability": score}
                    for timestamp, score in timeline
                ],
            }
        )
    report = summarize_events(events)
    report["threshold"] = threshold
    report["first_score_offset_s"] = initial
    report["observation_cadence_s"] = step
    report["by_pause_duration"] = breakdown(events, "pause_bucket")
    report["by_speech_source"] = breakdown(events, "speech_source")
    return report, events


def evaluate_fixed_silence(dataset: CanonicalEvaluationDataset, seconds: float) -> dict[str, Any]:
    events = []
    for decision in dataset.decisions:
        duration = decision.silence_end_s - decision.silence_start_s
        triggered = duration >= seconds
        events.append(
            {
                "label": decision.label,
                "triggered": triggered,
                "detection_delay_s": seconds if triggered else None,
                "positive_frames_after_first": 0,
                "pause_bucket": pause_bucket(duration),
                "speech_source": decision.provenance,
            }
        )
    result = summarize_events(events)
    result["silence_threshold_s"] = seconds
    result["by_pause_duration"] = breakdown(events, "pause_bucket")
    return result


def full_span_metrics(dataset, model, extractor, device, threshold, batch_size):
    probability_batches = []
    for start in range(0, len(dataset.decisions), batch_size):
        stop = min(len(dataset.decisions), start + batch_size)
        waveforms = [
            dataset.waveform_at(index, dataset.decisions[index].silence_end_s)
            for index in range(start, stop)
        ]
        probability_batches.append(
            predict_waveforms(model, extractor, waveforms, device, batch_size)
        )
    probabilities = np.concatenate(probability_batches)
    labels = np.asarray([item.label_id for item in dataset.decisions], dtype=np.int64)
    return binary_metrics(labels, probabilities, threshold), labels, probabilities


def main() -> None:
    args = parse_args()
    if args.partition != "ENG_evaluation_dev" and not args.allow_holdout:
        raise RuntimeError("holdout evaluation requires the explicit --allow-holdout flag")
    config = read_config(args.config)
    device = select_device(args.device)
    model, payload = load_model_checkpoint(args.checkpoint, device=device)
    threshold = float(payload["threshold"])
    with sqlite3.connect(config["dataset"]["index_path"]) as connection:
        index_metadata = dict(connection.execute("SELECT key, value FROM index_metadata"))
    dataset = CanonicalEvaluationDataset(
        config["dataset"]["index_path"],
        language=config["dataset"]["language"],
        partition=args.partition,
        window_seconds=config["dataset"]["window_seconds"],
        sample_rate=config["dataset"]["sample_rate"],
        normalize_waveform=bool(config["dataset"].get("normalize_waveform", False)),
    )
    extractor = WaveformFeatureExtractor(
        config["model"]["backbone"],
        config["dataset"]["window_seconds"],
        config["dataset"]["sample_rate"],
        config["model"].get("backbone_revision"),
        True,
    )
    span_metrics, labels, probabilities = full_span_metrics(
        dataset, model, extractor, device, threshold, args.batch_size
    )
    evaluation = config["evaluation"]
    replay, events = evaluate_model_replay(
        dataset,
        model,
        extractor,
        device,
        threshold=threshold,
        initial=evaluation["first_score_offset_seconds"],
        step=evaluation["replay_step_seconds"],
        batch_size=args.batch_size,
    )
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": args.checkpoint.resolve().as_posix(),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "checkpoint_threshold": threshold,
        "language": config["dataset"]["language"],
        "language_label": "ENG",
        "evaluation_partition": args.partition,
        "dataset_revision": index_metadata["livekit_revision"],
        "canonical_schema_version": int(index_metadata["schema_version"]),
        "scoring_code_sha256": file_sha256(Path(__file__)),
        "causal_contract": "Every score uses audio samples at or before its timestamp.",
        "protocol": {
            "window_seconds": config["dataset"]["window_seconds"],
            "sample_rate_hz": config["dataset"]["sample_rate"],
            "label_semantics": {"hold": 0, "end": 1},
            "threshold": threshold,
            "first_score_offset_seconds": evaluation["first_score_offset_seconds"],
            "observation_cadence_seconds": evaluation["replay_step_seconds"],
            "dispatch": "first threshold crossing per annotated pause; response latch enabled",
            "deadline": "end of the annotated silence span",
            "smoothing": None,
            "debounce": None,
            "timeout": None,
        },
        "full_span_metrics": span_metrics,
        "causal_replay": replay,
        "fixed_silence_baselines": {
            str(seconds): evaluate_fixed_silence(dataset, seconds)
            for seconds in (0.2, 0.5, 1.0)
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output.with_name(args.output.stem + "_events.json").write_text(
        json.dumps(events, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    np.savez_compressed(
        args.output.with_name(args.output.stem + "_full_span_predictions.npz"),
        labels=labels,
        probabilities=probabilities,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
