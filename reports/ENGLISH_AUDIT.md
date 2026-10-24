# English Turn Detector Audit

## Data

| Source | Role | Observed coverage |
|---|---|---|
| Pipecat | Train/validation | 59,280 / 6,522 English clips; near-balanced `hold`/`end` labels |
| LiveKit | Causal evaluation | 400 turns, 96 groups, 850 continuation pauses, 400 endings |

## Selected model

Baseline v0 uses `openai/whisper-tiny`, learned attention pooling, and a
`384 -> 256 -> 64 -> 1` classification head. It consumes the latest eight
seconds of 16 kHz mono audio and uses threshold `0.5`. The selected checkpoint
is epoch 1 with the encoder frozen; exact configuration and hashes are in
`checkpoint.md`.

## Quality

| Evaluation | Accuracy | End precision | End recall | Premature | Missed endings |
|---|---:|---:|---:|---:|---:|
| Pipecat validation | 93.24% | 91.24% | 95.15% | — | — |
| LiveKit causal development | 78.72% | 59.72% | 85.35% | 24.05% | 14.65% |
| LiveKit Holdout A | 81.79% | 67.63% | 92.16% | 23.81% | 7.84% |

The external result confirms a training-task mismatch: Pipecat supplies one
label per isolated clip, while production-like LiveKit replay makes repeated
decisions at pauses inside a conversation. Canonicalization aligns the schema
and audio format but cannot remove this difference.

A training variation was tried by unfreezing through epoch 3. It improved
Pipecat validation but left causal errors unchanged at 143 and increased
premature responses from 114 to 125, so Baseline v0 remained selected.

Audit evidence can be regenerated with
`python3 -m turn_detector.data_fetching.audit_split_duplicates` and
`python3 -m turn_detector.data_fetching.audit_livekit` using `PYTHONPATH=src`.
