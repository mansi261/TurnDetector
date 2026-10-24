# ENG Baseline v0 — Holdout A evaluation

## Frozen protocol

- Partition: `ENG_holdout_a` only
- Dataset revision: `ca9d98a9686b920a2d8c9eb984224ba9be74e4dd`
- Canonical schema: version 3
- Checkpoint: `experiments/english_whisper_tiny_v1/full_interrupted_drop_last/checkpoints/best.pt`
- Checkpoint SHA-256: `ef2315dd6bad380ccdc123a8cc73c4638f7503c57cf3eb0258d608d159979b35`
- Threshold: `0.5`
- Input: latest eight seconds of 16 kHz mono audio available at each score time
- Waveform normalization: disabled, matching Baseline v0 training
- Replay: first score at silence onset, followed by scores every 100 ms
- Dispatch: first threshold crossing per pause, with subsequent crossings latched
- Deadline: end of the annotated silence span
- Runtime: CPU, PyTorch 2.14.1, NumPy 2.5.4

The run contains 291 events from 102 turns and 25 conversation groups: 189
continuation pauses (`hold`) and 102 completed turns (`end`). The output IDs
exactly match the Holdout A index. Holdout B was not scored.

## Causal event results

| Actual / predicted action | Hold | End |
|---|---:|---:|
| Hold event | 144 | 45 |
| End event | 8 | 94 |

| Metric | Result | Denominator |
|---|---:|---:|
| Event accuracy | 81.79% | 238 / 291 |
| End precision | 67.63% | 94 / 139 emitted responses |
| End recall | 92.16% | 94 / 102 endings |
| Premature-response rate | 23.81% | 45 / 189 continuation pauses |
| Premature fraction of emitted responses | 32.37% | 45 / 139 responses |
| Missed-ending rate | 7.84% | 8 / 102 endings |
| Total event-error rate | 18.21% | 53 / 291 events |
| Detection-delay median | 0 ms | 94 detected endings |
| Detection-delay p95 | 300 ms | 94 detected endings |

Delay is simulated event time from silence onset. It excludes VAD, network,
scheduling, and model-compute latency.

## Pause-duration breakdown

| Pause duration | Hold events | Premature | Premature rate | End events | Detected | Missed |
|---|---:|---:|---:|---:|---:|---:|
| Under 500 ms | 131 | 23 | 17.56% | 0 | 0 | 0 |
| 500 ms to 1 s | 45 | 15 | 33.33% | 0 | 0 | 0 |
| At least 1 s | 13 | 7 | 53.85% | 102 | 94 | 8 |

All records have the same public-data provenance, so the speech-source slice is
identical to the overall result. The increasing premature rate across longer
continuation pauses is directly observed and remains the main quality risk.

## Full-span diagnostic

This view scores once at the end of each annotated silence. It is useful for
classification diagnostics but does not represent early production decisions.

| Metric | Result |
|---|---:|
| Accuracy | 78.35% |
| End precision | 66.39% |
| End recall | 77.45% |
| F1 | 71.49% |
| AUROC | 0.8742 |
| Hold false-positive rate | 21.16% |
| Brier score | 0.1588 |
| Expected calibration error, 10 bins | 12.48% |
| Log loss | 0.5366 |

The full-span confusion counts are TN 149, FP 40, FN 23, and TP 79. Hold
scores have mean 0.2211 and median 0.0156; end scores have mean 0.7549 and
median 0.9309. Their tails overlap: the hold p90 is 0.8740 and the end p10 is
0.1288.

## Development comparison

This comparison is descriptive only. It does not alter the frozen policy.

| Causal metric | Development | Holdout A |
|---|---:|---:|
| Event accuracy | 78.72% | 81.79% |
| End precision | 59.72% | 67.63% |
| End recall | 85.35% | 92.16% |
| Premature-response rate | 24.05% | 23.81% |
| Missed-ending rate | 14.65% | 7.84% |
| Premature fraction of emitted responses | 40.28% | 32.37% |
| Detection-delay p95 | 400 ms | 300 ms |

Holdout A is consistent with the development finding that Baseline v0 detects
most endings quickly but emits too often during continuation pauses. The
holdout result is better on several point estimates, but the model and
threshold remain unchanged and the partition will not be used for tuning.

## Fixed-silence reference policies

| Policy | Premature | Missed | Total event errors | Median delay |
|---|---:|---:|---:|---:|
| Model, threshold 0.5 | 45 | 8 | 53 / 291 (18.21%) | 0 ms |
| 200 ms silence | 136 | 0 | 136 / 291 (46.74%) | 200 ms |
| 500 ms silence | 58 | 0 | 58 / 291 (19.93%) | 500 ms |
| 1,000 ms silence | 13 | 0 | 13 / 291 (4.47%) | 1,000 ms |

The one-second rule has fewer errors on this partition but imposes a fixed
one-second response delay. The model is much faster, with a higher premature
response cost. This is an error-versus-delay trade-off, not an unqualified
model win.

## Reproduction and artifacts

```bash
PYTHONPATH=src .venv/bin/python -m turn_detector.evaluation.evaluate \
  --config configs/english_whisper_tiny_v1.json \
  --checkpoint experiments/english_whisper_tiny_v1/full_interrupted_drop_last/checkpoints/best.pt \
  --partition ENG_holdout_a \
  --allow-holdout \
  --device cpu \
  --batch-size 16 \
  --output experiments/english_whisper_tiny_v1/baseline_v0/evaluation/ENG_holdout_a_report.json
```

- Machine-readable report: `experiments/english_whisper_tiny_v1/baseline_v0/evaluation/ENG_holdout_a_report.json`
- Per-event causal timelines: `experiments/english_whisper_tiny_v1/baseline_v0/evaluation/ENG_holdout_a_report_events.json`
- Full-span labels and probabilities: `experiments/english_whisper_tiny_v1/baseline_v0/evaluation/ENG_holdout_a_report_full_span_predictions.npz`
- Scoring-code SHA-256: `91bdfd7dbf0cc171557d958c95b0b1aed25d3c8f0b3c94dd9b1a122e7f6dd812`
