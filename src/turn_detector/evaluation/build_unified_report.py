"""Build the three-level English development evaluation report."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

import numpy as np

from turn_detector.training.train import binary_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path("experiments/english_whisper_tiny_v1")
    parser.add_argument(
        "--baseline-report",
        type=Path,
        default=root / "baseline_v0/evaluation/ENG_evaluation_dev_report.json",
    )
    parser.add_argument(
        "--events",
        type=Path,
        default=root / "baseline_v0/evaluation/ENG_evaluation_dev_report_events.json",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=root
        / "baseline_v0/evaluation/ENG_evaluation_dev_report_full_span_predictions.npz",
    )
    parser.add_argument(
        "--candidate-report",
        type=Path,
        default=root / "threshold_optimization_v1/candidate_development_report.json",
    )
    parser.add_argument(
        "--serving-parity",
        type=Path,
        default=root / "baseline_v0/serving_parity.json",
    )
    parser.add_argument(
        "--api-benchmark",
        type=Path,
        default=root / "baseline_v0/api_benchmark_serving_v1.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "unified_evaluation_v1",
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        default=Path("reports/ENG_unified_three_level_evaluation_v1.md"),
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=1337)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_value(*args: str) -> str | None:
    try:
        return subprocess.run(
            ("git", *args), check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def calibration(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    clipped = np.clip(probabilities.astype(np.float64), 1e-7, 1.0 - 1e-7)
    labels_float = labels.astype(np.float64)
    bins = []
    ece = 0.0
    for index in range(10):
        lower = index / 10
        upper = (index + 1) / 10
        mask = (probabilities >= lower) & (
            probabilities <= upper if index == 9 else probabilities < upper
        )
        count = int(mask.sum())
        if not count:
            bins.append(
                {
                    "range": [lower, upper],
                    "count": 0,
                    "mean_p_end": None,
                    "observed_end_rate": None,
                    "absolute_gap": None,
                }
            )
            continue
        mean_score = float(probabilities[mask].mean())
        end_rate = float(labels_float[mask].mean())
        gap = abs(mean_score - end_rate)
        ece += count / len(labels) * gap
        bins.append(
            {
                "range": [lower, upper],
                "count": count,
                "mean_p_end": mean_score,
                "observed_end_rate": end_rate,
                "absolute_gap": gap,
            }
        )
    return {
        "brier_score": float(np.mean((probabilities - labels_float) ** 2)),
        "log_loss": float(
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


def classification_policy(
    labels: np.ndarray, probabilities: np.ndarray, threshold: float
) -> dict[str, Any]:
    result = binary_metrics(labels, probabilities, threshold)
    result["specificity"] = result["true_negative"] / max(
        1, result["true_negative"] + result["false_positive"]
    )
    result["balanced_accuracy"] = (result["end_recall"] + result["specificity"]) / 2
    return result


def rescore_events(events: list[dict[str, Any]], threshold: float) -> list[dict[str, Any]]:
    result = []
    for event in events:
        positives = [
            point
            for point in event["timeline"]
            if float(point["end_probability"]) >= threshold
        ]
        first_trigger = float(positives[0]["timestamp_s"]) if positives else None
        first_observation = float(event["timeline"][0]["timestamp_s"])
        result.append(
            {
                "decision_id": event["decision_id"],
                "source_id": event["source_id"],
                "group_id": event["group_id"],
                "label": event["label"],
                "pause_duration_s": float(event["pause_duration_s"]),
                "pause_bucket": event["pause_bucket"],
                "speech_source": event["speech_source"],
                "triggered": bool(positives),
                "first_trigger_s": first_trigger,
                "detection_delay_s": (
                    first_trigger - first_observation if first_trigger is not None else None
                ),
                "positive_frames_after_first": max(0, len(positives) - 1),
                "maximum_p_end": max(
                    float(point["end_probability"]) for point in event["timeline"]
                ),
            }
        )
    return result


def causal_summary(events: list[dict[str, Any]]) -> dict[str, Any]:
    holds = [event for event in events if event["label"] == "hold"]
    endings = [event for event in events if event["label"] == "end"]
    premature = sum(event["triggered"] for event in holds)
    detected = sum(event["triggered"] for event in endings)
    missed = len(endings) - detected
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
        "premature_responses": premature,
        "premature_response_rate": premature / max(1, len(holds)),
        "detected_endings": detected,
        "missed_endings": missed,
        "missed_ending_rate": missed / max(1, len(endings)),
        "total_causal_event_errors": premature + missed,
        "total_causal_event_error_rate": (premature + missed) / max(1, len(events)),
        "emitted_responses": emitted,
        "premature_fraction_of_emitted": premature / max(1, emitted),
        "detected_end_delay_median_s": float(np.median(delays)) if delays else None,
        "detected_end_delay_p95_s": float(np.percentile(delays, 95)) if delays else None,
        "repeat_trigger_opportunities": sum(
            int(event["positive_frames_after_first"]) for event in events
        ),
    }


def causal_breakdown(events: list[dict[str, Any]], key: str) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        grouped[str(event[key])].append(event)
    return {name: causal_summary(items) for name, items in sorted(grouped.items())}


def policy_transitions(
    baseline: list[dict[str, Any]], candidate: list[dict[str, Any]]
) -> dict[str, Any]:
    pairs = list(zip(baseline, candidate, strict=True))
    for left, right in pairs:
        if left["decision_id"] != right["decision_id"]:
            raise ValueError("baseline and candidate event order differs")

    hold_pairs = [(left, right) for left, right in pairs if left["label"] == "hold"]
    end_pairs = [(left, right) for left, right in pairs if left["label"] == "end"]
    resolved = [
        left for left, right in hold_pairs if left["triggered"] and not right["triggered"]
    ]
    new_misses = [
        left for left, right in end_pairs if left["triggered"] and not right["triggered"]
    ]
    return {
        "hold_events": {
            "correct_under_both": sum(
                not left["triggered"] and not right["triggered"]
                for left, right in hold_pairs
            ),
            "premature_under_both": sum(
                left["triggered"] and right["triggered"] for left, right in hold_pairs
            ),
            "premature_resolved_by_candidate": len(resolved),
            "new_premature_under_candidate": sum(
                not left["triggered"] and right["triggered"] for left, right in hold_pairs
            ),
        },
        "end_events": {
            "detected_under_both": sum(
                left["triggered"] and right["triggered"] for left, right in end_pairs
            ),
            "missed_under_both": sum(
                not left["triggered"] and not right["triggered"]
                for left, right in end_pairs
            ),
            "new_misses_under_candidate": len(new_misses),
            "recovered_by_candidate": sum(
                not left["triggered"] and right["triggered"] for left, right in end_pairs
            ),
        },
        "resolved_premature_examples": [
            {
                "decision_id": event["decision_id"],
                "pause_duration_s": event["pause_duration_s"],
                "maximum_p_end": event["maximum_p_end"],
            }
            for event in sorted(resolved, key=lambda item: item["maximum_p_end"], reverse=True)[:5]
        ],
        "new_missed_ending_examples": [
            {
                "decision_id": event["decision_id"],
                "baseline_delay_s": event["detection_delay_s"],
                "maximum_p_end": event["maximum_p_end"],
            }
            for event in sorted(new_misses, key=lambda item: item["maximum_p_end"])[:5]
        ],
    }


def percentile_interval(values: list[float]) -> dict[str, float]:
    return {
        "lower_95": float(np.percentile(values, 2.5)),
        "upper_95": float(np.percentile(values, 97.5)),
    }


def grouped_bootstrap(
    baseline: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
    *,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    by_group_baseline: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_group_candidate: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in baseline:
        by_group_baseline[event["group_id"]].append(event)
    for event in candidate:
        by_group_candidate[event["group_id"]].append(event)
    groups = sorted(by_group_baseline)
    if groups != sorted(by_group_candidate):
        raise ValueError("baseline and candidate groups differ")

    metric_names = (
        "premature_response_rate",
        "missed_ending_rate",
        "total_causal_event_error_rate",
        "premature_fraction_of_emitted",
    )
    samples = {
        policy: {metric: [] for metric in metric_names}
        for policy in ("baseline", "candidate", "candidate_minus_baseline")
    }
    rng = np.random.default_rng(seed)
    for _ in range(replicates):
        sampled = rng.choice(groups, size=len(groups), replace=True)
        baseline_sample = [
            event for group in sampled for event in by_group_baseline[str(group)]
        ]
        candidate_sample = [
            event for group in sampled for event in by_group_candidate[str(group)]
        ]
        left = causal_summary(baseline_sample)
        right = causal_summary(candidate_sample)
        for metric in metric_names:
            samples["baseline"][metric].append(float(left[metric]))
            samples["candidate"][metric].append(float(right[metric]))
            samples["candidate_minus_baseline"][metric].append(
                float(right[metric]) - float(left[metric])
            )
    return {
        "method": "percentile bootstrap resampling inferred conversation groups",
        "groups": len(groups),
        "replicates": replicates,
        "seed": seed,
        "conditional_on_selected_candidate": True,
        "intervals": {
            policy: {
                metric: percentile_interval(values) for metric, values in metrics.items()
            }
            for policy, metrics in samples.items()
        },
    }


def validate_inputs(
    baseline_report: dict[str, Any],
    candidate_report: dict[str, Any],
    labels: np.ndarray,
    probabilities: np.ndarray,
    events: list[dict[str, Any]],
) -> None:
    if baseline_report["evaluation_partition"] != "ENG_evaluation_dev":
        raise ValueError("baseline report is not ENG_evaluation_dev")
    if candidate_report["evaluation_partition"] != "ENG_evaluation_dev":
        raise ValueError("candidate report is not ENG_evaluation_dev")
    for key in ("dataset_revision", "canonical_schema_version", "checkpoint_sha256"):
        if baseline_report[key] != candidate_report[key]:
            raise ValueError(f"baseline and candidate {key} differ")
    if candidate_report.get("holdouts_scored") != []:
        raise ValueError("candidate report indicates holdout access")
    if float(baseline_report["protocol"]["threshold"]) != 0.5:
        raise ValueError("expected Baseline v0 threshold 0.5")
    if float(candidate_report["threshold"]) != 0.825:
        raise ValueError("expected candidate threshold 0.825")
    if not (len(labels) == len(probabilities) == len(events)):
        raise ValueError("prediction and event counts differ")
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("labels must contain hold=0 and end=1")


def format_percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def format_delta(value: float) -> str:
    return f"{value * 100:+.2f} pp"


def markdown_report(report: dict[str, Any]) -> str:
    level1 = report["level_1_full_span_classification"]
    level2 = report["level_2_causal_event_replay"]
    level3 = report["level_3_serving_and_policy"]
    baseline_class = level1["policies"]["baseline_v0"]
    candidate_class = level1["policies"]["threshold_0_825_candidate"]
    baseline_event = level2["policies"]["baseline_v0"]
    candidate_event = level2["policies"]["threshold_0_825_candidate"]
    transitions = level2["policy_transitions"]
    intervals = level2["grouped_bootstrap"]["intervals"]
    benchmark = level3["measured_baseline_api"]
    parity = level3["measured_baseline_parity"]

    classification_rows = []
    for label, key in (
        ("Accuracy", "accuracy"),
        ("Balanced accuracy", "balanced_accuracy"),
        ("End precision", "precision"),
        ("End recall", "end_recall"),
        ("F1", "f1"),
        ("Hold false-positive rate", "hold_false_positive_rate"),
    ):
        classification_rows.append(
            f"| {label} | {format_percent(baseline_class[key])} | "
            f"{format_percent(candidate_class[key])} | "
            f"{format_delta(candidate_class[key] - baseline_class[key])} |"
        )

    pause_rows = []
    for bucket, label in (
        ("under_500ms", "Under 500 ms"),
        ("500ms_to_1s", "500 ms to 1 s"),
        ("at_least_1s", "At least 1 s"),
    ):
        left = level2["by_pause_duration"]["baseline_v0"][bucket]
        right = level2["by_pause_duration"]["threshold_0_825_candidate"][bucket]
        pause_rows.append(
            f"| {label} | {left['premature_responses']}/{left['hold_events']} "
            f"({format_percent(left['premature_response_rate'])}) | "
            f"{right['premature_responses']}/{right['hold_events']} "
            f"({format_percent(right['premature_response_rate'])}) | "
            f"{left['missed_endings']}/{left['end_events']} | "
            f"{right['missed_endings']}/{right['end_events']} |"
        )

    latency_rows = []
    for item in benchmark["warm"]:
        latency_rows.append(
            f"| {item['concurrency']} | {item['requests']} | "
            f"{item['client_latency_ms']['p50']:.1f} ms | "
            f"{item['client_latency_ms']['p95']:.1f} ms | "
            f"{item['throughput_completed_per_second']:.1f}/s | "
            f"{format_percent(item['error_rate'])} |"
        )

    new_misses = "\n".join(
        f"- `{item['decision_id']}` — max `p_end={item['maximum_p_end']:.4f}`; "
        f"Baseline delay {item['baseline_delay_s'] * 1000:.0f} ms."
        for item in transitions["new_missed_ending_examples"]
    )
    return f"""# ENG unified three-level development evaluation v1

## Decision

Baseline v0 at threshold `0.5` remains the serving policy. Threshold `0.825`
is compared as a development candidate only and is **not promoted**. Both
holdout groups remain unscored.

The candidate removes 55 premature responses but creates 27 additional missed
endings. Its lower total causal error count does not establish a general
improvement because it was selected and measured on the same development split.

## Frozen comparison

- Partition: `ENG_evaluation_dev` only.
- Data revision: `{report['protocol']['dataset_revision']}`.
- Examples: {report['protocol']['turns']} turns, {report['protocol']['groups']} inferred conversation groups, {report['protocol']['events']} decisions.
- Labels: {report['protocol']['hold_events']} continuation pauses and {report['protocol']['end_events']} true endings.
- Model checkpoint SHA-256: `{report['protocol']['checkpoint_sha256']}`.
- Model and preprocessing: identical for both policies; only the decision threshold changes.
- Input: latest 8 seconds of 16 kHz mono audio available at each decision timestamp.
- Replay: silence onset and every 100 ms through the annotated span.
- Dispatch: first threshold crossing, latched once per annotated event.
- No smoothing or debounce. Event-time delay excludes network and compute time.

## Level 1 — full-span classification

This view scores one prediction at the end of each annotated span. It measures
class separation and probability quality, but it cannot reveal an earlier
premature trigger.

| Metric | Baseline 0.5 | Candidate 0.825 | Change |
|---|---:|---:|---:|
{chr(10).join(classification_rows)}

| Confusion count | Baseline 0.5 | Candidate 0.825 |
|---|---:|---:|
| Hold predicted hold | {baseline_class['true_negative']} | {candidate_class['true_negative']} |
| Hold predicted end | {baseline_class['false_positive']} | {candidate_class['false_positive']} |
| End predicted hold | {baseline_class['false_negative']} | {candidate_class['false_negative']} |
| End predicted end | {baseline_class['true_positive']} | {candidate_class['true_positive']} |

Threshold-independent AUROC is {level1['shared_probability_quality']['auroc']:.4f}.
The Brier score is {level1['shared_probability_quality']['calibration']['brier_score']:.4f},
10-bin expected calibration error is
{level1['shared_probability_quality']['calibration']['expected_calibration_error_10_bins']:.4f},
and log loss is {level1['shared_probability_quality']['calibration']['log_loss']:.4f}.
The 14.30-point calibration error means `p_end` should currently be treated as
a model score, not a literal probability. Changing the threshold does not
change calibration.

## Level 2 — causal event replay

This is the primary behavior view. Each decision uses only audio available at
that timestamp, and a response is emitted on the first crossing.

| Metric | Baseline 0.5 | Candidate 0.825 | Change |
|---|---:|---:|---:|
| Premature responses | {baseline_event['premature_responses']}/{baseline_event['hold_events']} ({format_percent(baseline_event['premature_response_rate'])}) | {candidate_event['premature_responses']}/{candidate_event['hold_events']} ({format_percent(candidate_event['premature_response_rate'])}) | {candidate_event['premature_responses'] - baseline_event['premature_responses']:+d} |
| Missed endings | {baseline_event['missed_endings']}/{baseline_event['end_events']} ({format_percent(baseline_event['missed_ending_rate'])}) | {candidate_event['missed_endings']}/{candidate_event['end_events']} ({format_percent(candidate_event['missed_ending_rate'])}) | {candidate_event['missed_endings'] - baseline_event['missed_endings']:+d} |
| Total causal errors | {baseline_event['total_causal_event_errors']} | {candidate_event['total_causal_event_errors']} | {candidate_event['total_causal_event_errors'] - baseline_event['total_causal_event_errors']:+d} |
| Premature share of emitted responses | {format_percent(baseline_event['premature_fraction_of_emitted'])} | {format_percent(candidate_event['premature_fraction_of_emitted'])} | {format_delta(candidate_event['premature_fraction_of_emitted'] - baseline_event['premature_fraction_of_emitted'])} |
| Detected-end delay, median | {baseline_event['detected_end_delay_median_s'] * 1000:.0f} ms | {candidate_event['detected_end_delay_median_s'] * 1000:.0f} ms | 0 ms |
| Detected-end delay, p95 | {baseline_event['detected_end_delay_p95_s'] * 1000:.0f} ms | {candidate_event['detected_end_delay_p95_s'] * 1000:.0f} ms | +100 ms |

The transition table explains the trade-off directly:

- Holds: {transitions['hold_events']['correct_under_both']} correct under both,
  {transitions['hold_events']['premature_under_both']} premature under both, and
  {transitions['hold_events']['premature_resolved_by_candidate']} premature responses fixed by `0.825`.
- Endings: {transitions['end_events']['detected_under_both']} detected under both,
  {transitions['end_events']['missed_under_both']} missed under both, and
  {transitions['end_events']['new_misses_under_candidate']} new misses caused by `0.825`.

### Pause-duration slices

| Pause duration | Baseline premature holds | Candidate premature holds | Baseline missed ends | Candidate missed ends |
|---|---:|---:|---:|---:|
{chr(10).join(pause_rows)}

The source field has only `livekit/eot-bench-data`, so a recording-condition or
microphone-source comparison is unavailable. The grouped bootstrap resamples
the 46 inferred conversation groups, but its intervals are conditional on a
candidate already selected on this same data and are therefore optimistic.

| Group-bootstrap 95% interval | Baseline 0.5 | Candidate 0.825 | Candidate change |
|---|---:|---:|---:|
| Premature-response rate | {format_percent(intervals['baseline']['premature_response_rate']['lower_95'])} to {format_percent(intervals['baseline']['premature_response_rate']['upper_95'])} | {format_percent(intervals['candidate']['premature_response_rate']['lower_95'])} to {format_percent(intervals['candidate']['premature_response_rate']['upper_95'])} | {format_delta(intervals['candidate_minus_baseline']['premature_response_rate']['lower_95'])} to {format_delta(intervals['candidate_minus_baseline']['premature_response_rate']['upper_95'])} |
| Missed-ending rate | {format_percent(intervals['baseline']['missed_ending_rate']['lower_95'])} to {format_percent(intervals['baseline']['missed_ending_rate']['upper_95'])} | {format_percent(intervals['candidate']['missed_ending_rate']['lower_95'])} to {format_percent(intervals['candidate']['missed_ending_rate']['upper_95'])} | {format_delta(intervals['candidate_minus_baseline']['missed_ending_rate']['lower_95'])} to {format_delta(intervals['candidate_minus_baseline']['missed_ending_rate']['upper_95'])} |
| Total causal-error rate | {format_percent(intervals['baseline']['total_causal_event_error_rate']['lower_95'])} to {format_percent(intervals['baseline']['total_causal_event_error_rate']['upper_95'])} | {format_percent(intervals['candidate']['total_causal_event_error_rate']['lower_95'])} to {format_percent(intervals['candidate']['total_causal_event_error_rate']['upper_95'])} | {format_delta(intervals['candidate_minus_baseline']['total_causal_event_error_rate']['lower_95'])} to {format_delta(intervals['candidate_minus_baseline']['total_causal_event_error_rate']['upper_95'])} |

### Representative new candidate misses

{new_misses}

## Level 3 — serving and production policy

The production boundary is checked separately from model quality. Baseline v0
has measured PyTorch-to-ONNX and direct-to-API parity. The threshold candidate
uses identical weights and preprocessing, so it has the same `p_end` values,
but it has not been promoted or separately served.

- Direct model/API maximum absolute difference: `{parity['summary']['maximum_direct_api_absolute_difference']}` across {parity['summary']['samples']} parity samples.
- PyTorch/ONNX maximum absolute difference: `{parity['summary']['maximum_pytorch_onnx_absolute_difference']:.8f}`.
- Decision mismatches at the Baseline threshold: `{parity['summary']['decision_mismatches']}`.
- Cold HTTP request: `{benchmark['cold_inference']['client_latency_ms']:.1f} ms`.
- Warm latency target is met only at concurrency 1 in this measured run; queueing increases latency at concurrency 4 and 8.

| Concurrency | Requests | Client p50 | Client p95 | Throughput | Errors |
|---:|---:|---:|---:|---:|---:|
{chr(10).join(latency_rows)}

The API benchmark used one worker on `{benchmark['hardware']['platform']}` with
an {benchmark['request']['duration_seconds']:.1f}-second, {benchmark['request']['bytes']}-byte WAV request.
The service consumes only the latest 8 seconds. Latency is measured evidence
for Baseline v0, not a separate candidate benchmark.

## Recommendation

Keep `0.5` active. Treat `0.825` as evidence that thresholding alone can reduce
interruptions, but not as the next release: the missed-ending rate rises from
{format_percent(baseline_event['missed_ending_rate'])} to
{format_percent(candidate_event['missed_ending_rate'])}. The next controlled
experiment should change model training while keeping this evaluation protocol
fixed. Do not use either holdout until the model, preprocessing, and policy are
frozen.

## Reproduce

```bash
PYTHONPATH=src .venv/bin/python -m turn_detector.evaluation.build_unified_report
```

Machine-readable output:
`experiments/english_whisper_tiny_v1/unified_evaluation_v1/report.json`.
"""


def main() -> None:
    args = parse_args()
    source_paths = {
        "baseline_report": args.baseline_report,
        "causal_events": args.events,
        "full_span_predictions": args.predictions,
        "candidate_report": args.candidate_report,
        "serving_parity": args.serving_parity,
        "api_benchmark": args.api_benchmark,
    }
    baseline_report = json.loads(args.baseline_report.read_text(encoding="utf-8"))
    candidate_report = json.loads(args.candidate_report.read_text(encoding="utf-8"))
    events = json.loads(args.events.read_text(encoding="utf-8"))
    arrays = np.load(args.predictions)
    labels = arrays["labels"]
    probabilities = arrays["probabilities"]
    parity = json.loads(args.serving_parity.read_text(encoding="utf-8"))
    benchmark = json.loads(args.api_benchmark.read_text(encoding="utf-8"))
    validate_inputs(baseline_report, candidate_report, labels, probabilities, events)

    baseline_threshold = float(baseline_report["protocol"]["threshold"])
    candidate_threshold = float(candidate_report["threshold"])
    baseline_events = rescore_events(events, baseline_threshold)
    candidate_events = rescore_events(events, candidate_threshold)
    baseline_summary = causal_summary(baseline_events)
    candidate_summary = causal_summary(candidate_events)
    groups = len({event["group_id"] for event in events})
    turns = len({event["source_id"] for event in events})

    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "decision": {
            "active_policy": "baseline_v0_threshold_0.5",
            "candidate": "threshold_0.825_development_only",
            "candidate_promoted": False,
            "holdouts_scored": [],
        },
        "protocol": {
            "language": "ENG",
            "partition": "ENG_evaluation_dev",
            "dataset_revision": baseline_report["dataset_revision"],
            "canonical_schema_version": baseline_report["canonical_schema_version"],
            "checkpoint_sha256": baseline_report["checkpoint_sha256"],
            "turns": turns,
            "groups": groups,
            "events": len(events),
            "hold_events": int(np.sum(labels == 0)),
            "end_events": int(np.sum(labels == 1)),
            "audio": baseline_report["protocol"],
            "causal_contract": baseline_report["causal_contract"],
        },
        "level_1_full_span_classification": {
            "description": "One prediction at the end of each annotated span.",
            "shared_probability_quality": {
                "auroc": float(baseline_report["full_span_metrics"]["auroc"]),
                "calibration": calibration(labels, probabilities),
                "score_distribution": score_distribution(labels, probabilities),
            },
            "policies": {
                "baseline_v0": classification_policy(
                    labels, probabilities, baseline_threshold
                ),
                "threshold_0_825_candidate": classification_policy(
                    labels, probabilities, candidate_threshold
                ),
            },
        },
        "level_2_causal_event_replay": {
            "description": "First threshold crossing using only audio available at each timestamp.",
            "policies": {
                "baseline_v0": baseline_summary,
                "threshold_0_825_candidate": candidate_summary,
            },
            "candidate_minus_baseline": {
                key: candidate_summary[key] - baseline_summary[key]
                for key in (
                    "premature_responses",
                    "premature_response_rate",
                    "missed_endings",
                    "missed_ending_rate",
                    "total_causal_event_errors",
                    "total_causal_event_error_rate",
                    "premature_fraction_of_emitted",
                    "detected_end_delay_median_s",
                    "detected_end_delay_p95_s",
                )
            },
            "policy_transitions": policy_transitions(
                baseline_events, candidate_events
            ),
            "by_pause_duration": {
                "baseline_v0": causal_breakdown(baseline_events, "pause_bucket"),
                "threshold_0_825_candidate": causal_breakdown(
                    candidate_events, "pause_bucket"
                ),
            },
            "by_speech_source": {
                "baseline_v0": causal_breakdown(baseline_events, "speech_source"),
                "threshold_0_825_candidate": causal_breakdown(
                    candidate_events, "speech_source"
                ),
            },
            "grouped_bootstrap": grouped_bootstrap(
                baseline_events,
                candidate_events,
                replicates=args.bootstrap_replicates,
                seed=args.seed,
            ),
        },
        "level_3_serving_and_policy": {
            "active_threshold": baseline_threshold,
            "candidate_threshold": candidate_threshold,
            "candidate_served": False,
            "same_weights_and_preprocessing": True,
            "measured_baseline_parity": parity,
            "measured_baseline_api": benchmark,
        },
        "limitations": [
            "The 0.825 threshold was selected and evaluated on the same development split.",
            "Conversation groups are inferred rather than verified speaker identities.",
            "All examples share one public speech-source provenance label.",
            "Event time excludes VAD, network, scheduling, and inference latency.",
            "No holdout group was scored.",
        ],
        "reproducibility": {
            "invocation": [str(Path(__file__).resolve())],
            "source_files": {
                name: {
                    "path": path.resolve().as_posix(),
                    "sha256": file_sha256(path),
                }
                for name, path in source_paths.items()
            },
            "scoring_code_sha256": file_sha256(Path(__file__)),
            "git_head": git_value("rev-parse", "HEAD"),
            "git_diff_sha256": hashlib.sha256(
                (git_value("diff", "--no-ext-diff", "HEAD") or "").encode()
            ).hexdigest(),
            "randomness": {
                "scope": "grouped bootstrap only",
                "seed": args.seed,
                "replicates": args.bootstrap_replicates,
            },
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "report.json"
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text(markdown_report(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "report": output_path.resolve().as_posix(),
                "markdown": args.markdown.resolve().as_posix(),
                "active_threshold": baseline_threshold,
                "candidate_threshold": candidate_threshold,
                "candidate_promoted": False,
                "holdouts_scored": [],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
