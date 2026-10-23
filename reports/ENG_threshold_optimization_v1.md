# ENG threshold optimization v1

## Decision

Threshold `0.825` minimized total causal event errors on `ENG_evaluation_dev`.
It is retained as a development candidate, not promoted to the serving default.
It substantially reduced premature responses but nearly doubled missed endings.
Both frozen holdouts remain unscored.

## Fixed experiment

- Checkpoint SHA-256: `ef2315dd6bad380ccdc123a8cc73c4638f7503c57cf3eb0258d608d159979b35`
- Data revision: `ca9d98a9686b920a2d8c9eb984224ba9be74e4dd`
- Canonical schema: `3`
- Tuning partition: `ENG_evaluation_dev`, 672 events across 46 groups
- Search: 19 thresholds from `0.500` through `0.950` in `0.025` steps
- Selection: lowest `premature responses + missed endings`; ties favor fewer premature responses, fewer misses, then lower median delay
- Replay: saved causal timelines at 100 ms cadence; every score uses audio available at that timestamp
- Model weights, preprocessing, and audio examples: unchanged

Reproduce the sweep:

```bash
PYTHONPATH=src .venv/bin/python -m turn_detector.evaluation.optimize_threshold
```

## Baseline versus candidate

| Metric | Threshold 0.500 | Threshold 0.825 | Change |
|---|---:|---:|---:|
| Total causal event errors | 143 | 115 | -28 (-19.6%) |
| Premature responses | 114/474 (24.05%) | 59/474 (12.45%) | -55 (-11.60 pp) |
| Missed endings | 29/198 (14.65%) | 56/198 (28.28%) | +27 (+13.64 pp) |
| Detected endings | 169/198 | 142/198 | -27 |
| Premature share of emitted responses | 40.28% | 29.35% | -10.93 pp |
| Detected-ending delay, median | 0 ms | 0 ms | 0 ms |
| Detected-ending delay, p95 | 400 ms | 500 ms | +100 ms |
| Full-span accuracy | 76.93% | 78.72% | +1.79 pp |
| Full-span end precision | 58.57% | 67.30% | +8.73 pp |
| Full-span end recall | 74.24% | 54.04% | -20.20 pp |
| Full-span F1 | 65.48% | 59.94% | -5.54 pp |

The primary criterion passed because total causal errors decreased. The result
is not an unqualified quality improvement: fewer interruptions came at the cost
of substantially more missed endings and lower end recall.

## Pause-duration breakdown

| Pause slice | Baseline premature | Candidate premature | Ending misses |
|---|---:|---:|---:|
| Under 500 ms | 64/320 (20.00%) | 30/320 (9.38%) | Not applicable |
| 500 ms to 1 s | 35/113 (30.97%) | 19/113 (16.81%) | Not applicable |
| At least 1 s, continuing pause | 15/41 (36.59%) | 10/41 (24.39%) | Not applicable |
| At least 1 s, true ending | Not applicable | Not applicable | 29/198 to 56/198 |

All examples have the same public speech-source provenance, so a source-level
breakdown repeats the aggregate result and cannot establish generalization to
other recording conditions.

## Recommendation

Keep serving Baseline v0 at threshold `0.5` for now. Threshold `0.825` is useful
when avoiding interruptions is valued more strongly than responding promptly,
but the missed-ending regression is too large to call it a general replacement.
The next model experiment should target separation between continuation pauses
and true endings, rather than relying only on a stricter threshold. After a
candidate resolves that trade-off, evaluate it once on an untouched holdout.
