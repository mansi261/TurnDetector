"""Compare Baseline v0 and epoch 3 on causal English development replay."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

from turn_detector.evaluation.causal import (
    causal_breakdown,
    causal_summary,
    grouped_bootstrap,
    rescore_events,
)
from turn_detector.evaluation.metrics import (
    format_percent as percent,
    format_percentage_delta as percentage_delta,
)
from turn_detector.utils import command_output, file_sha256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path("experiments/english_whisper_tiny_v1")
    parser.add_argument(
        "--baseline-report",
        type=Path,
        default=root / "baseline_v0/evaluation/ENG_evaluation_dev_report.json",
    )
    parser.add_argument(
        "--baseline-events",
        type=Path,
        default=root / "baseline_v0/evaluation/ENG_evaluation_dev_report_events.json",
    )
    parser.add_argument(
        "--candidate-report",
        type=Path,
        default=root / "full_three_epoch_v1/evaluation/ENG_evaluation_dev_report.json",
    )
    parser.add_argument(
        "--candidate-events",
        type=Path,
        default=root
        / "full_three_epoch_v1/evaluation/ENG_evaluation_dev_report_events.json",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=root / "causal_comparison_v1"
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        default=root / "causal_comparison_v1/report.md",
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=1337)
    return parser.parse_args()


def validate_inputs(
    baseline_report: dict[str, Any],
    candidate_report: dict[str, Any],
    baseline_events: list[dict[str, Any]],
    candidate_events: list[dict[str, Any]],
    threshold: float,
) -> None:
    if threshold != 0.5:
        raise ValueError("this comparison is frozen at threshold 0.5")
    for report in (baseline_report, candidate_report):
        if report["evaluation_partition"] != "ENG_evaluation_dev":
            raise ValueError("only ENG_evaluation_dev is allowed")
        if float(report["protocol"]["threshold"]) != threshold:
            raise ValueError("report threshold differs from the fixed comparison threshold")
    for key in (
        "dataset_revision",
        "canonical_schema_version",
        "scoring_code_sha256",
    ):
        if baseline_report[key] != candidate_report[key]:
            raise ValueError(f"baseline and candidate {key} differ")
    protocol_keys = (
        "window_seconds",
        "sample_rate_hz",
        "label_semantics",
        "first_score_offset_seconds",
        "observation_cadence_seconds",
        "dispatch",
        "deadline",
        "smoothing",
        "debounce",
        "timeout",
    )
    for key in protocol_keys:
        if baseline_report["protocol"][key] != candidate_report["protocol"][key]:
            raise ValueError(f"baseline and candidate protocol {key} differ")
    if len(baseline_events) != len(candidate_events):
        raise ValueError("event counts differ")
    for baseline, candidate in zip(baseline_events, candidate_events, strict=True):
        for key in (
            "decision_id",
            "source_id",
            "group_id",
            "label",
            "pause_duration_s",
            "pause_bucket",
            "speech_source",
        ):
            if baseline[key] != candidate[key]:
                raise ValueError(f"event field {key} differs")
        baseline_times = [point["timestamp_s"] for point in baseline["timeline"]]
        candidate_times = [point["timestamp_s"] for point in candidate["timeline"]]
        if baseline_times != candidate_times:
            raise ValueError("causal observation timestamps differ")


def policy_transitions(
    baseline: list[dict[str, Any]], candidate: list[dict[str, Any]]
) -> dict[str, Any]:
    pairs = list(zip(baseline, candidate, strict=True))
    hold_pairs = [(left, right) for left, right in pairs if left["label"] == "hold"]
    end_pairs = [(left, right) for left, right in pairs if left["label"] == "end"]
    fixed_holds = [
        left for left, right in hold_pairs if left["triggered"] and not right["triggered"]
    ]
    new_premature = [
        right for left, right in hold_pairs if not left["triggered"] and right["triggered"]
    ]
    recovered_ends = [
        right for left, right in end_pairs if not left["triggered"] and right["triggered"]
    ]
    new_misses = [
        left for left, right in end_pairs if left["triggered"] and not right["triggered"]
    ]
    return {
        "hold": {
            "correct_under_both": sum(
                not left["triggered"] and not right["triggered"]
                for left, right in hold_pairs
            ),
            "premature_under_both": sum(
                left["triggered"] and right["triggered"] for left, right in hold_pairs
            ),
            "premature_fixed_by_candidate": len(fixed_holds),
            "new_premature_under_candidate": len(new_premature),
        },
        "end": {
            "detected_under_both": sum(
                left["triggered"] and right["triggered"] for left, right in end_pairs
            ),
            "missed_under_both": sum(
                not left["triggered"] and not right["triggered"]
                for left, right in end_pairs
            ),
            "recovered_by_candidate": len(recovered_ends),
            "new_misses_under_candidate": len(new_misses),
        },
        "new_premature_examples": [
            {
                "decision_id": event["decision_id"],
                "pause_duration_s": event["pause_duration_s"],
                "maximum_p_end": event["maximum_p_end"],
                "first_trigger_delay_s": event["detection_delay_s"],
            }
            for event in sorted(
                new_premature, key=lambda item: item["maximum_p_end"], reverse=True
            )[:10]
        ],
        "recovered_ending_examples": [
            {
                "decision_id": event["decision_id"],
                "maximum_p_end": event["maximum_p_end"],
                "detection_delay_s": event["detection_delay_s"],
            }
            for event in sorted(
                recovered_ends, key=lambda item: item["detection_delay_s"]
            )[:10]
        ],
        "new_missed_ending_examples": [
            {
                "decision_id": event["decision_id"],
                "baseline_maximum_p_end": event["maximum_p_end"],
                "baseline_detection_delay_s": event["detection_delay_s"],
            }
            for event in new_misses[:10]
        ],
    }


def common_detection_delays(
    baseline: list[dict[str, Any]], candidate: list[dict[str, Any]]
) -> dict[str, Any]:
    differences = []
    for left, right in zip(baseline, candidate, strict=True):
        if left["label"] == "end" and left["triggered"] and right["triggered"]:
            differences.append(
                float(right["detection_delay_s"]) - float(left["detection_delay_s"])
            )
    values = np.asarray(differences)
    tolerance = 1e-9
    return {
        "common_detected_endings": len(values),
        "candidate_faster": int(np.sum(values < -tolerance)),
        "same_delay": int(np.sum(np.abs(values) <= tolerance)),
        "candidate_slower": int(np.sum(values > tolerance)),
        "candidate_minus_baseline_median_s": float(np.median(values)),
        "candidate_minus_baseline_p95_s": float(np.percentile(values, 95)),
    }


def difference(left: dict[str, Any], right: dict[str, Any]) -> dict[str, float]:
    fields = (
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
    return {field: float(right[field]) - float(left[field]) for field in fields}


def classification_difference(
    baseline: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, float]:
    return {
        key: float(candidate[key]) - float(baseline[key])
        for key in (
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
    }


def milliseconds(value: float | None) -> str:
    return "N/A" if value is None else f"{value * 1000:.0f} ms"


def markdown_report(report: dict[str, Any]) -> str:
    baseline = report["causal_replay"]["baseline_v0"]
    candidate = report["causal_replay"]["epoch_3_candidate"]
    delta = report["causal_replay"]["candidate_minus_baseline"]
    transitions = report["policy_transitions"]
    full_left = report["full_span_classification"]["baseline_v0"]
    full_right = report["full_span_classification"]["epoch_3_candidate"]
    full_delta = report["full_span_classification"]["candidate_minus_baseline"]
    intervals = report["grouped_bootstrap"]["intervals"]
    delay = report["common_detection_delay_comparison"]

    pause_rows = []
    for key, label in (
        ("under_500ms", "Under 500 ms"),
        ("500ms_to_1s", "500 ms to 1 s"),
        ("at_least_1s", "At least 1 s"),
    ):
        left = report["by_pause_duration"]["baseline_v0"][key]
        right = report["by_pause_duration"]["epoch_3_candidate"][key]
        pause_rows.append(
            f"| {label} | {left['premature_responses']}/{left['hold_events']} "
            f"({percent(left['premature_response_rate'])}) | "
            f"{right['premature_responses']}/{right['hold_events']} "
            f"({percent(right['premature_response_rate'])}) | "
            f"{left['missed_endings']}/{left['end_events']} | "
            f"{right['missed_endings']}/{right['end_events']} |"
        )

    premature_examples = "\n".join(
        f"- `{item['decision_id']}` — {item['pause_duration_s']:.1f}-second "
        f"continuation pause, max `p_end={item['maximum_p_end']:.4f}`, first crossing "
        f"at {item['first_trigger_delay_s'] * 1000:.0f} ms."
        for item in transitions["new_premature_examples"][:5]
    )
    recovered_examples = "\n".join(
        f"- `{item['decision_id']}` — max `p_end={item['maximum_p_end']:.4f}`, "
        f"detected after {item['detection_delay_s'] * 1000:.0f} ms."
        for item in transitions["recovered_ending_examples"][:5]
    )
    return f"""# ENG epoch-3 causal development comparison v1

## Decision

Epoch 3 is **not promoted**. At the unchanged threshold `0.5`, it recovers 11
missed endings but adds 11 premature responses, leaving total causal event
errors unchanged at 143. Baseline v0 remains the serving policy. Neither holdout
was scored.

## Frozen causal protocol

- Partition: `ENG_evaluation_dev` only.
- Data revision: `{report['protocol']['dataset_revision']}`.
- Examples: {report['protocol']['turns']} turns, {report['protocol']['groups']} inferred conversation groups, and {report['protocol']['events']} annotated events.
- Labels: {report['protocol']['hold_events']} continuation pauses and {report['protocol']['end_events']} true endings.
- Threshold: `0.5` for both checkpoints.
- Input: latest 8 seconds of 16 kHz mono audio available at each timestamp.
- Replay: first score at silence onset, then every 100 ms until the annotated span ends.
- Dispatch: first threshold crossing per event with response latching.
- No smoothing or debounce. Delay is simulated event time and excludes compute and network latency.

## Primary causal results

| Metric | Baseline epoch 1 | Candidate epoch 3 | Change |
|---|---:|---:|---:|
| Premature responses | {baseline['premature_responses']}/{baseline['hold_events']} ({percent(baseline['premature_response_rate'])}) | {candidate['premature_responses']}/{candidate['hold_events']} ({percent(candidate['premature_response_rate'])}) | {candidate['premature_responses'] - baseline['premature_responses']:+d} ({percentage_delta(delta['premature_response_rate'])}) |
| Missed endings | {baseline['missed_endings']}/{baseline['end_events']} ({percent(baseline['missed_ending_rate'])}) | {candidate['missed_endings']}/{candidate['end_events']} ({percent(candidate['missed_ending_rate'])}) | {candidate['missed_endings'] - baseline['missed_endings']:+d} ({percentage_delta(delta['missed_ending_rate'])}) |
| Total causal errors | {baseline['total_causal_event_errors']} | {candidate['total_causal_event_errors']} | {candidate['total_causal_event_errors'] - baseline['total_causal_event_errors']:+d} |
| Premature share of emitted responses | {percent(baseline['premature_fraction_of_emitted'])} | {percent(candidate['premature_fraction_of_emitted'])} | {percentage_delta(delta['premature_fraction_of_emitted'])} |
| Detected-end delay, median | {milliseconds(baseline['detected_end_delay_median_s'])} | {milliseconds(candidate['detected_end_delay_median_s'])} | {milliseconds(delta['detected_end_delay_median_s'])} |
| Detected-end delay, p95 | {milliseconds(baseline['detected_end_delay_p95_s'])} | {milliseconds(candidate['detected_end_delay_p95_s'])} | {milliseconds(delta['detected_end_delay_p95_s'])} |

The transition counts expose changes hidden by the equal total:

- Continuation pauses: {transitions['hold']['correct_under_both']} correct under both,
  {transitions['hold']['premature_under_both']} premature under both,
  {transitions['hold']['premature_fixed_by_candidate']} fixed by epoch 3, and
  {transitions['hold']['new_premature_under_candidate']} newly premature under epoch 3.
- True endings: {transitions['end']['detected_under_both']} detected under both,
  {transitions['end']['missed_under_both']} missed under both,
  {transitions['end']['recovered_by_candidate']} recovered by epoch 3, and
  {transitions['end']['new_misses_under_candidate']} newly missed under epoch 3.

Among {delay['common_detected_endings']} endings detected by both models, epoch 3
was faster on {delay['candidate_faster']}, equal on {delay['same_delay']}, and
slower on {delay['candidate_slower']}. The median candidate-minus-baseline delay
was {milliseconds(delay['candidate_minus_baseline_median_s'])}.

## Pause-duration slices

| Pause duration | Baseline premature holds | Epoch-3 premature holds | Baseline missed ends | Epoch-3 missed ends |
|---|---:|---:|---:|---:|
{chr(10).join(pause_rows)}

All examples have the same public source provenance, so a microphone or acoustic
source comparison is unavailable.

## Group-level uncertainty

The intervals resample the 46 inferred conversation groups. They measure sampling
variation on this development set, not final-test uncertainty.

| 95% group-bootstrap interval | Baseline | Epoch 3 | Candidate change |
|---|---:|---:|---:|
| Premature-response rate | {percent(intervals['baseline']['premature_response_rate']['lower_95'])} to {percent(intervals['baseline']['premature_response_rate']['upper_95'])} | {percent(intervals['candidate']['premature_response_rate']['lower_95'])} to {percent(intervals['candidate']['premature_response_rate']['upper_95'])} | {percentage_delta(intervals['candidate_minus_baseline']['premature_response_rate']['lower_95'])} to {percentage_delta(intervals['candidate_minus_baseline']['premature_response_rate']['upper_95'])} |
| Missed-ending rate | {percent(intervals['baseline']['missed_ending_rate']['lower_95'])} to {percent(intervals['baseline']['missed_ending_rate']['upper_95'])} | {percent(intervals['candidate']['missed_ending_rate']['lower_95'])} to {percent(intervals['candidate']['missed_ending_rate']['upper_95'])} | {percentage_delta(intervals['candidate_minus_baseline']['missed_ending_rate']['lower_95'])} to {percentage_delta(intervals['candidate_minus_baseline']['missed_ending_rate']['upper_95'])} |
| Total causal-error rate | {percent(intervals['baseline']['total_causal_event_error_rate']['lower_95'])} to {percent(intervals['baseline']['total_causal_event_error_rate']['upper_95'])} | {percent(intervals['candidate']['total_causal_event_error_rate']['lower_95'])} to {percent(intervals['candidate']['total_causal_event_error_rate']['upper_95'])} | {percentage_delta(intervals['candidate_minus_baseline']['total_causal_event_error_rate']['lower_95'])} to {percentage_delta(intervals['candidate_minus_baseline']['total_causal_event_error_rate']['upper_95'])} |

## Full-span diagnostic

| Metric | Baseline epoch 1 | Candidate epoch 3 | Change |
|---|---:|---:|---:|
| Accuracy | {percent(full_left['accuracy'])} | {percent(full_right['accuracy'])} | {percentage_delta(full_delta['accuracy'])} |
| End precision | {percent(full_left['precision'])} | {percent(full_right['precision'])} | {percentage_delta(full_delta['precision'])} |
| End recall | {percent(full_left['end_recall'])} | {percent(full_right['end_recall'])} | {percentage_delta(full_delta['end_recall'])} |
| F1 | {percent(full_left['f1'])} | {percent(full_right['f1'])} | {percentage_delta(full_delta['f1'])} |
| AUROC | {percent(full_left['auroc'])} | {percent(full_right['auroc'])} | {percentage_delta(full_delta['auroc'])} |
| Hold false-positive rate | {percent(full_left['hold_false_positive_rate'])} | {percent(full_right['hold_false_positive_rate'])} | {percentage_delta(full_delta['hold_false_positive_rate'])} |

Full-span classification improves, especially end recall, but causal replay shows
that epoch 3 crosses the threshold on more continuation pauses. One prediction at
the end of a span therefore overstates the operational improvement.

## Representative changes

### New premature responses

{premature_examples}

### Recovered endings

{recovered_examples}

## Recommendation

Keep Baseline v0 active. Epoch 3 is useful evidence that encoder fine-tuning
improves end sensitivity, but it does not improve the primary interruption-versus-
miss trade-off at threshold `0.5`. The next model experiment should train on more
realistic causal pause examples or use an objective that explicitly penalizes
continuation-pause crossings. Calibration alone will not fix the 11 additional
premature responses because the decision threshold is already frozen.

Do not score either holdout until a model and policy improve development behavior
and are frozen.

## Reproduce

```bash
PYTHONPATH=src .venv/bin/python -m turn_detector.evaluation.compare_causal_checkpoints
```

Raw artifacts:
`experiments/english_whisper_tiny_v1/causal_comparison_v1/`.
"""


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {args.output_dir}")
    baseline_report = json.loads(args.baseline_report.read_text(encoding="utf-8"))
    candidate_report = json.loads(args.candidate_report.read_text(encoding="utf-8"))
    baseline_source_events = json.loads(
        args.baseline_events.read_text(encoding="utf-8")
    )
    candidate_source_events = json.loads(
        args.candidate_events.read_text(encoding="utf-8")
    )
    validate_inputs(
        baseline_report,
        candidate_report,
        baseline_source_events,
        candidate_source_events,
        args.threshold,
    )
    baseline_events = rescore_events(baseline_source_events, args.threshold)
    candidate_events = rescore_events(candidate_source_events, args.threshold)
    baseline_summary = causal_summary(baseline_events)
    candidate_summary = causal_summary(candidate_events)
    groups = len({event["group_id"] for event in baseline_events})
    turns = len({event["source_id"] for event in baseline_events})
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "development_comparison_only",
        "decision": {
            "active_policy": "baseline_v0_epoch_1_threshold_0.5",
            "candidate": "epoch_3_threshold_0.5",
            "candidate_promoted": False,
            "holdouts_scored": [],
        },
        "protocol": {
            "language": "ENG",
            "partition": "ENG_evaluation_dev",
            "dataset_revision": baseline_report["dataset_revision"],
            "canonical_schema_version": baseline_report["canonical_schema_version"],
            "turns": turns,
            "groups": groups,
            "events": len(baseline_events),
            "hold_events": baseline_summary["hold_events"],
            "end_events": baseline_summary["end_events"],
            "threshold": args.threshold,
            "audio_and_policy": baseline_report["protocol"],
            "causal_contract": baseline_report["causal_contract"],
        },
        "checkpoints": {
            "baseline_v0": {
                "path": baseline_report["checkpoint"],
                "sha256": baseline_report["checkpoint_sha256"],
            },
            "epoch_3_candidate": {
                "path": candidate_report["checkpoint"],
                "sha256": candidate_report["checkpoint_sha256"],
            },
        },
        "causal_replay": {
            "baseline_v0": baseline_summary,
            "epoch_3_candidate": candidate_summary,
            "candidate_minus_baseline": difference(
                baseline_summary, candidate_summary
            ),
        },
        "policy_transitions": policy_transitions(
            baseline_events, candidate_events
        ),
        "common_detection_delay_comparison": common_detection_delays(
            baseline_events, candidate_events
        ),
        "by_pause_duration": {
            "baseline_v0": causal_breakdown(baseline_events, "pause_bucket"),
            "epoch_3_candidate": causal_breakdown(
                candidate_events, "pause_bucket"
            ),
        },
        "by_speech_source": {
            "baseline_v0": causal_breakdown(baseline_events, "speech_source"),
            "epoch_3_candidate": causal_breakdown(
                candidate_events, "speech_source"
            ),
        },
        "grouped_bootstrap": grouped_bootstrap(
            baseline_events,
            candidate_events,
            replicates=args.bootstrap_replicates,
            seed=args.seed,
        ),
        "full_span_classification": {
            "baseline_v0": baseline_report["full_span_metrics"],
            "epoch_3_candidate": candidate_report["full_span_metrics"],
            "candidate_minus_baseline": classification_difference(
                baseline_report["full_span_metrics"],
                candidate_report["full_span_metrics"],
            ),
        },
        "fixed_silence_references": baseline_report["fixed_silence_baselines"],
        "limitations": [
            "The development partition has informed prior model and threshold decisions.",
            "Conversation groups are inferred rather than verified speaker identities.",
            "All records share one public speech-source provenance value.",
            "Simulated event time excludes inference, network, VAD, and scheduling latency.",
            "The epoch-3 training continuation did not restore epoch-1 RNG state.",
            "No holdout group was scored.",
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    source_paths = {
        "baseline_report": args.baseline_report,
        "baseline_events": args.baseline_events,
        "candidate_report": args.candidate_report,
        "candidate_events": args.candidate_events,
    }
    manifest = {
        "created_utc": report["created_utc"],
        "invocation": [sys.executable, *sys.argv],
        "threshold": args.threshold,
        "dataset_revision": report["protocol"]["dataset_revision"],
        "partition": report["protocol"]["partition"],
        "scoring_code_sha256": file_sha256(Path(__file__)),
        "source_files": {
            name: {
                "path": path.resolve().as_posix(),
                "sha256": file_sha256(path),
            }
            for name, path in source_paths.items()
        },
        "report_sha256": file_sha256(report_path),
        "bootstrap": {"replicates": args.bootstrap_replicates, "seed": args.seed},
        "git_head": command_output("git", "rev-parse", "HEAD"),
        "git_diff_sha256": hashlib.sha256(
            (command_output("git", "diff", "--no-ext-diff", "HEAD") or "").encode()
        ).hexdigest(),
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
                "candidate_promoted": False,
                "holdouts_scored": [],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
