# ENG unified three-level development evaluation v1

## Decision

Baseline v0 at threshold `0.5` remains the serving policy. Threshold `0.825`
is compared as a development candidate only and is **not promoted**. Both
holdout groups remain unscored.

The candidate removes 55 premature responses but creates 27 additional missed
endings. Its lower total causal error count does not establish a general
improvement because it was selected and measured on the same development split.

## Frozen comparison

- Partition: `ENG_evaluation_dev` only.
- Data revision: `ca9d98a9686b920a2d8c9eb984224ba9be74e4dd`.
- Examples: 198 turns, 46 inferred conversation groups, 672 decisions.
- Labels: 474 continuation pauses and 198 true endings.
- Model checkpoint SHA-256: `ef2315dd6bad380ccdc123a8cc73c4638f7503c57cf3eb0258d608d159979b35`.
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
| Accuracy | 76.93% | 78.72% | +1.79 pp |
| Balanced accuracy | 76.15% | 71.53% | -4.62 pp |
| End precision | 58.57% | 67.30% | +8.73 pp |
| End recall | 74.24% | 54.04% | -20.20 pp |
| F1 | 65.48% | 59.94% | -5.53 pp |
| Hold false-positive rate | 21.94% | 10.97% | -10.97 pp |

| Confusion count | Baseline 0.5 | Candidate 0.825 |
|---|---:|---:|
| Hold predicted hold | 370 | 422 |
| Hold predicted end | 104 | 52 |
| End predicted hold | 51 | 91 |
| End predicted end | 147 | 107 |

Threshold-independent AUROC is 0.8477.
The Brier score is 0.1725,
10-bin expected calibration error is
0.1430,
and log loss is 0.5637.
The 14.30-point calibration error means `p_end` should currently be treated as
a model score, not a literal probability. Changing the threshold does not
change calibration.

## Level 2 — causal event replay

This is the primary behavior view. Each decision uses only audio available at
that timestamp, and a response is emitted on the first crossing.

| Metric | Baseline 0.5 | Candidate 0.825 | Change |
|---|---:|---:|---:|
| Premature responses | 114/474 (24.05%) | 59/474 (12.45%) | -55 |
| Missed endings | 29/198 (14.65%) | 56/198 (28.28%) | +27 |
| Total causal errors | 143 | 115 | -28 |
| Premature share of emitted responses | 40.28% | 29.35% | -10.93 pp |
| Detected-end delay, median | 0 ms | 0 ms | 0 ms |
| Detected-end delay, p95 | 400 ms | 500 ms | +100 ms |

The transition table explains the trade-off directly:

- Holds: 360 correct under both,
  59 premature under both, and
  55 premature responses fixed by `0.825`.
- Endings: 142 detected under both,
  29 missed under both, and
  27 new misses caused by `0.825`.

### Pause-duration slices

| Pause duration | Baseline premature holds | Candidate premature holds | Baseline missed ends | Candidate missed ends |
|---|---:|---:|---:|---:|
| Under 500 ms | 64/320 (20.00%) | 30/320 (9.38%) | 0/0 | 0/0 |
| 500 ms to 1 s | 35/113 (30.97%) | 19/113 (16.81%) | 0/0 | 0/0 |
| At least 1 s | 15/41 (36.59%) | 10/41 (24.39%) | 29/198 | 56/198 |

The source field has only `livekit/eot-bench-data`, so a recording-condition or
microphone-source comparison is unavailable. The grouped bootstrap resamples
the 46 inferred conversation groups, but its intervals are conditional on a
candidate already selected on this same data and are therefore optimistic.

| Group-bootstrap 95% interval | Baseline 0.5 | Candidate 0.825 | Candidate change |
|---|---:|---:|---:|
| Premature-response rate | 18.87% to 29.41% | 8.99% to 16.45% | -14.80 pp to -8.64 pp |
| Missed-ending rate | 9.27% to 20.00% | 21.50% to 35.42% | +8.59 pp to +19.59 pp |
| Total causal-error rate | 17.64% to 25.15% | 13.90% to 20.53% | -6.99 pp to -1.39 pp |

### Representative new candidate misses

- `livekit:en__1115_49__user_turn_008:pause_02` — max `p_end=0.5040`; Baseline delay 500 ms.
- `livekit:en__287_49__user_turn_002:pause_11` — max `p_end=0.5384`; Baseline delay 400 ms.
- `livekit:en__1115_49__user_turn_000:pause_05` — max `p_end=0.5691`; Baseline delay 100 ms.
- `livekit:en__219_75__user_turn_001:pause_07` — max `p_end=0.5806`; Baseline delay 400 ms.
- `livekit:en__215_67__user_turn_007:pause_01` — max `p_end=0.5935`; Baseline delay 0 ms.

## Level 3 — serving and production policy

The production boundary is checked separately from model quality. Baseline v0
has measured PyTorch-to-ONNX and direct-to-API parity. The threshold candidate
uses identical weights and preprocessing, so it has the same `p_end` values,
but it has not been promoted or separately served.

- Direct model/API maximum absolute difference: `0.0` across 3 parity samples.
- PyTorch/ONNX maximum absolute difference: `0.00000238`.
- Decision mismatches at the Baseline threshold: `0`.
- Cold HTTP request: `178.0 ms`.
- Warm latency target is met only at concurrency 1 in this measured run; queueing increases latency at concurrency 4 and 8.

| Concurrency | Requests | Client p50 | Client p95 | Throughput | Errors |
|---:|---:|---:|---:|---:|---:|
| 1 | 50 | 42.1 ms | 96.3 ms | 18.9/s | 0.00% |
| 4 | 50 | 113.3 ms | 144.2 ms | 34.5/s | 0.00% |
| 8 | 50 | 213.7 ms | 313.8 ms | 34.0/s | 0.00% |

The API benchmark used one worker on `macOS-15.5-arm64-arm-64bit` with
an 11.0-second, 352044-byte WAV request.
The service consumes only the latest 8 seconds. Latency is measured evidence
for Baseline v0, not a separate candidate benchmark.

## Recommendation

Keep `0.5` active. Treat `0.825` as evidence that thresholding alone can reduce
interruptions, but not as the next release: the missed-ending rate rises from
14.65% to
28.28%. The next controlled
experiment should change model training while keeping this evaluation protocol
fixed. Do not use either holdout until the model, preprocessing, and policy are
frozen.

## Reproduce

```bash
PYTHONPATH=src .venv/bin/python -m turn_detector.evaluation.build_unified_report
```

Machine-readable output:
`experiments/english_whisper_tiny_v1/unified_evaluation_v1/report.json`.
