"""Shared causal replay summaries."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np


def rescore_events(
    events: list[dict[str, Any]], threshold: float
) -> list[dict[str, Any]]:
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
                    first_trigger - first_observation
                    if first_trigger is not None
                    else None
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


def causal_breakdown(
    events: list[dict[str, Any]], key: str
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        grouped[str(event[key])].append(event)
    return {name: causal_summary(items) for name, items in sorted(grouped.items())}


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
                metric: percentile_interval(values)
                for metric, values in metrics.items()
            }
            for policy, metrics in samples.items()
        },
    }
