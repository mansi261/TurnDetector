# ENG Baseline v0 development evaluation

## Frozen protocol

- Evaluation partition: `ENG_evaluation_dev` only.
- Dataset revision: `ca9d98a9686b920a2d8c9eb984224ba9be74e4dd`.
- Turns, groups, and decisions: 198 turns, 46 inferred conversation groups, 672 pause decisions.
- Labels: 474 continuing-turn pauses (`hold`) and 198 final pauses (`end`).
- Checkpoint SHA-256: `ef2315dd6bad380ccdc123a8cc73c4638f7503c57cf3eb0258d608d159979b35`.
- Model threshold: `0.5`, unchanged from Baseline v0.
- Input: latest eight seconds of 16 kHz mono audio available at each timestamp.
- Replay: first score at silence onset, then every 100 ms until the annotated span ends.
- Dispatch: first threshold crossing within a pause; subsequent positive frames do not dispatch again.
- Deadline: end of the annotated silence span.
- No smoothing, debounce, or timeout was applied.
- Timing is simulated event time and excludes VAD, network, inference, and scheduling latency.

## Operational results

| Policy | Premature / 474 holds | Missed / 198 ends | Median end delay | P95 end delay |
| --- | ---: | ---: | ---: | ---: |
| Baseline v0, threshold 0.5 | 114 (24.05%) | 29 (14.65%) | 0 ms | 400 ms |
| Fixed silence 200 ms | 341 (71.94%) | 0 | 200 ms | 200 ms |
| Fixed silence 500 ms | 154 (32.49%) | 0 | 500 ms | 500 ms |
| Fixed silence 1,000 ms | 41 (8.65%) | 0 | 1,000 ms | 1,000 ms |

Baseline v0 emitted 283 simulated responses: 169 matched endings and 114 premature responses. Therefore 40.28% of emitted responses were premature. It is substantially faster than the one-second silence rule, but it makes more operational errors. It improves over the 200 ms and 500 ms rules on premature responses, while missing endings that all fixed rules eventually detect because every final span lasts 1.5 seconds.

## Pause-duration slices

| Pause duration | Hold events | Premature responses | Premature rate | End events | Missed endings |
| --- | ---: | ---: | ---: | ---: | ---: |
| Under 500 ms | 320 | 64 | 20.00% | 0 | 0 |
| 500 ms–1 s | 113 | 35 | 30.97% | 0 | 0 |
| At least 1 s | 41 | 15 | 36.59% | 198 | 29 |

The source has one recorded speech-source value, so a meaningful microphone, telephone, noise, or synthetic-source breakdown is unavailable.

## Full-span diagnostic

At the end of every annotated span, accuracy is 76.93%, end precision is 58.57%, and end recall is 74.24% (`TN=370`, `FP=104`, `FN=51`, `TP=147`). These differ from causal replay because model probabilities are not monotonic: some events cross 0.5 earlier and fall below it by the end. Operational scoring therefore uses the first causal crossing rather than only the final frame.

## Representative development errors

High-confidence premature examples include:

- `livekit:en__280_147__user_turn_003:pause_01` — 0.7 s hold, maximum end probability 0.9948.
- `livekit:en__287_49__user_turn_008:pause_05` — 0.7 s hold, maximum end probability 0.9896.
- `livekit:en__287_49__user_turn_008:pause_00` — 1.4 s hold, maximum end probability 0.9884.

Low-confidence missed endings include:

- `livekit:en__875_82__user_turn_000:pause_04` — maximum end probability 0.0023.
- `livekit:en__989_144__user_turn_020:pause_00` — maximum end probability 0.0045.
- `livekit:en__633_9__user_turn_002:pause_03` — maximum end probability 0.0074.

## Interpretation

Baseline v0 demonstrates useful semantic endpointing but is not yet a final production policy. Its fast detections come with a high premature-response rate, and 29 final endings never cross the unchanged threshold during the available 1.5-second silence. The next planned phase is serving-artifact verification and latency measurement before deciding whether to adjust the model or policy. Neither holdout has been scored.

Raw artifacts:

- `experiments/english_whisper_tiny_v1/baseline_v0/evaluation/ENG_evaluation_dev_report.json`
- `experiments/english_whisper_tiny_v1/baseline_v0/evaluation/ENG_evaluation_dev_report_events.json`
- `experiments/english_whisper_tiny_v1/baseline_v0/evaluation/ENG_evaluation_dev_report_full_span_predictions.npz`
