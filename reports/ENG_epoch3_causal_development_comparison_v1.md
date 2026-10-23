# ENG epoch-3 causal development comparison v1

## Decision

Epoch 3 is **not promoted**. At the unchanged threshold `0.5`, it recovers 11
missed endings but adds 11 premature responses, leaving total causal event
errors unchanged at 143. Baseline v0 remains the serving policy. Neither holdout
was scored.

## Frozen causal protocol

- Partition: `ENG_evaluation_dev` only.
- Data revision: `ca9d98a9686b920a2d8c9eb984224ba9be74e4dd`.
- Examples: 198 turns, 46 inferred conversation groups, and 672 annotated events.
- Labels: 474 continuation pauses and 198 true endings.
- Threshold: `0.5` for both checkpoints.
- Input: latest 8 seconds of 16 kHz mono audio available at each timestamp.
- Replay: first score at silence onset, then every 100 ms until the annotated span ends.
- Dispatch: first threshold crossing per event with response latching.
- No smoothing or debounce. Delay is simulated event time and excludes compute and network latency.

## Primary causal results

| Metric | Baseline epoch 1 | Candidate epoch 3 | Change |
|---|---:|---:|---:|
| Premature responses | 114/474 (24.05%) | 125/474 (26.37%) | +11 (+2.32 pp) |
| Missed endings | 29/198 (14.65%) | 18/198 (9.09%) | -11 (-5.56 pp) |
| Total causal errors | 143 | 143 | +0 |
| Premature share of emitted responses | 40.28% | 40.98% | +0.70 pp |
| Detected-end delay, median | 0 ms | 0 ms | 0 ms |
| Detected-end delay, p95 | 400 ms | 400 ms | 0 ms |

The transition counts expose changes hidden by the equal total:

- Continuation pauses: 332 correct under both,
  97 premature under both,
  17 fixed by epoch 3, and
  28 newly premature under epoch 3.
- True endings: 166 detected under both,
  15 missed under both,
  14 recovered by epoch 3, and
  3 newly missed under epoch 3.

Among 166 endings detected by both models, epoch 3
was faster on 23, equal on 111, and
slower on 32. The median candidate-minus-baseline delay
was 0 ms.

## Pause-duration slices

| Pause duration | Baseline premature holds | Epoch-3 premature holds | Baseline missed ends | Epoch-3 missed ends |
|---|---:|---:|---:|---:|
| Under 500 ms | 64/320 (20.00%) | 70/320 (21.88%) | 0/0 | 0/0 |
| 500 ms to 1 s | 35/113 (30.97%) | 41/113 (36.28%) | 0/0 | 0/0 |
| At least 1 s | 15/41 (36.59%) | 14/41 (34.15%) | 29/198 | 18/198 |

All examples have the same public source provenance, so a microphone or acoustic
source comparison is unavailable.

## Group-level uncertainty

The intervals resample the 46 inferred conversation groups. They measure sampling
variation on this development set, not final-test uncertainty.

| 95% group-bootstrap interval | Baseline | Epoch 3 | Candidate change |
|---|---:|---:|---:|
| Premature-response rate | 18.87% to 29.41% | 20.93% to 31.79% | -0.63 pp to +5.47 pp |
| Missed-ending rate | 9.27% to 20.00% | 5.44% to 13.33% | -9.95 pp to -1.08 pp |
| Total causal-error rate | 17.64% to 25.15% | 17.21% to 25.52% | -2.75 pp to +2.77 pp |

## Full-span diagnostic

| Metric | Baseline epoch 1 | Candidate epoch 3 | Change |
|---|---:|---:|---:|
| Accuracy | 76.93% | 78.27% | +1.34 pp |
| End precision | 58.57% | 59.42% | +0.85 pp |
| End recall | 74.24% | 82.83% | +8.59 pp |
| F1 | 65.48% | 69.20% | +3.72 pp |
| AUROC | 84.77% | 89.24% | +4.47 pp |
| Hold false-positive rate | 21.94% | 23.63% | +1.69 pp |

Full-span classification improves, especially end recall, but causal replay shows
that epoch 3 crosses the threshold on more continuation pauses. One prediction at
the end of a span therefore overstates the operational improvement.

## Representative changes

### New premature responses

- `livekit:en__1115_49__user_turn_008:pause_00` — 0.8-second continuation pause, max `p_end=0.9964`, first crossing at 300 ms.
- `livekit:en__907_96__user_turn_004:pause_01` — 0.2-second continuation pause, max `p_end=0.9933`, first crossing at 0 ms.
- `livekit:en__875_82__user_turn_000:pause_03` — 0.9-second continuation pause, max `p_end=0.9894`, first crossing at 400 ms.
- `livekit:en__989_144__user_turn_009:pause_00` — 0.1-second continuation pause, max `p_end=0.9816`, first crossing at 0 ms.
- `livekit:en__708_120__user_turn_003:pause_01` — 0.2-second continuation pause, max `p_end=0.9780`, first crossing at 0 ms.

### Recovered endings

- `livekit:en__576_127__user_turn_004:pause_07` — max `p_end=0.9166`, detected after 0 ms.
- `livekit:en__723_75__user_turn_006:pause_01` — max `p_end=0.9324`, detected after 0 ms.
- `livekit:en__829_87__user_turn_005:pause_03` — max `p_end=0.9847`, detected after 0 ms.
- `livekit:en__875_82__user_turn_007:pause_02` — max `p_end=0.9867`, detected after 0 ms.
- `livekit:en__447_20__user_turn_000:pause_09` — max `p_end=0.9964`, detected after 100 ms.

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
