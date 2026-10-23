# LiveKit ENG public sample audit

## Scope and identity

- Intended use: external English turn-detection development evaluation and untouched holdouts.
- Dataset: `livekit/eot-bench-data`, English config, source split `validation`.
- Pinned revision: `ca9d98a9686b920a2d8c9eb984224ba9be74e4dd`.
- License declared by the snapshot: CC BY 4.0.
- Record unit: one complete user turn. Each turn contains one or more silence spans.
- Audit artifact: `reports/livekit_ENG_public_400_audit.json`.
- Audit command: `PYTHONPATH=src .venv/bin/python -m turn_detector.data_fetching.audit_livekit`.

## Observed counts

| Measure | Result |
| --- | ---: |
| English turns scanned | 400 |
| Inferred conversation groups | 96 |
| Audio duration | 1.4003 hours |
| Readable 16 kHz mono WAVs | 400/400 |
| Within-turn hold pauses | 850 |
| Final end pauses | 400 |
| Missing required fields | 0 |
| Malformed or out-of-bounds spans | 0 |
| Duplicate turn IDs | 0 |
| Encoded-audio duplicate groups | 0 |
| Decoded-audio duplicate groups | 0 |
| Silent recordings | 0 |
| Recordings reaching absolute peak 0.999 | 59 |

The 59 peak-level recordings are candidates for clipping inspection; a peak threshold alone does not prove audible clipping.

## Pause coverage

Hold pauses range from 0.1 to 4.4 seconds, with a median of 0.3 seconds. Their bins are:

| Hold-pause duration | Count |
| --- | ---: |
| Under 0.2 s | 235 |
| 0.2–0.5 s | 332 |
| 0.5–1.0 s | 204 |
| 1.0–2.0 s | 61 |
| 2.0–5.0 s | 18 |

Every final end span is exactly 1.5 seconds. The dataset supports causal scoring at multiple points inside those spans, but it cannot measure natural variation in the total duration of final trailing silence. Pauses longer than five seconds were excluded upstream. Backchannels, interruptions, overlap, and truncation do not have explicit labels.

## Grouping, provenance, and leakage

Conversation groups are inferred from the ID prefix before `__user_turn_`; the source does not provide a separate conversation column or speaker IDs. Therefore conversation-disjoint partitions can be enforced using the inferred key, but speaker-independent evaluation cannot be verified.

The rows contain no real-versus-synthetic or recording-source metadata, so provenance is unknown for all 400 records from the files alone. Transcripts, word timestamps, and prior messages are retained in the immutable source Parquet but are not copied into model input fields.

The canonical partitions contain no conversation group overlap:

| Partition | Turns | Groups | Hold decisions | End decisions |
| --- | ---: | ---: | ---: | ---: |
| `ENG_evaluation_dev` | 198 | 46 | 474 | 198 |
| `ENG_holdout_a` | 102 | 25 | 189 | 102 |
| `ENG_holdout_b` | 100 | 25 | 187 | 100 |

All groups containing any of the previously inspected 50 turns are assigned to development. Exact encoded and decoded audio hashes are unique, so there is no exact audio duplication across these partitions.

## Verdict

Suitable with constraints for English external turn-detection evaluation. It has valid audio, useful short and long within-turn pauses, causal silence spans, and group-disjoint partitioning. The main limitations are the small public sample, inferred rather than explicit conversation IDs, missing speaker and provenance identifiers, and uniform 1.5-second final spans. Holdouts must remain unopened until a candidate and serving policy are frozen.
