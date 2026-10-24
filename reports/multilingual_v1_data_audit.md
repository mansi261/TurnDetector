# Multilingual v1 data audit

- Audit time: `2026-10-23T08:06:15.973055+00:00`
- Canonical index: `/Users/mansivyas/Desktop/TurnDetector/data/canonical_multilingual_v1/dataset.sqlite3`
- Languages: `ara, deu, eng, spa, fra, hin, ind, ita, jpn, kor, nld, por, tur, zho`
- Pipecat revision: `f8c95d284595defb6b88a715c356cbc5c31b7ad1`
- LiveKit revision: `ca9d98a9686b920a2d8c9eb984224ba9be74e4dd`
- Intended use: Pipecat training/validation and LiveKit causal development/final evaluation.

## Observed counts

| Language | Train | Validation | Evaluation decisions | Holdout decisions | Synthetic train |
|---|---:|---:|---:|---:|---:|
| ara | 6674 | 722 | 835 | 249 | 6674 |
| deu | 10861 | 1197 | 809 | 298 | 10861 |
| eng | 59270 | 6532 | 947 | 303 | 19951 |
| spa | 14386 | 1647 | 857 | 379 | 10797 |
| fra | 10338 | 1173 | 810 | 244 | 10338 |
| hin | 10814 | 1192 | 741 | 333 | 10814 |
| ind | 7242 | 847 | 991 | 302 | 7242 |
| ita | 6466 | 726 | 719 | 209 | 6466 |
| jpn | 6606 | 783 | 908 | 322 | 6606 |
| kor | 7388 | 776 | 616 | 225 | 7388 |
| nld | 11240 | 1278 | 709 | 253 | 11240 |
| por | 11464 | 1328 | 852 | 291 | 11464 |
| tur | 7377 | 836 | 631 | 203 | 7377 |
| zho | 7359 | 800 | 823 | 276 | 7359 |

## Integrity and audio

- Required-field missing counts: `{"audio_path": 0, "dataset_revision": 0, "decision_id": 0, "group_id": 0, "label": 0, "language": 0, "parquet_row_group": 0, "parquet_row_index": 0, "provenance": 0, "source": 0, "source_id": 0, "source_language": 0, "split": 0}`
- Readable source recordings: **202,776 / 202,776**; unreadable: **0**.
- Total source audio: **449.48 hours**.
- Exact encoded-audio duplicate groups: **0**; cross-split: **0**; cross-source: **0**.
- LiveKit conversation-group overlap: **0**; source-record overlap: **0**.
- Deterministic audio sample: **112** recordings; silent: **0**; clipped: **25**.
- Full-population audio checks cover decodability, sample rate, channels, duration, and exact encoded-byte duplicates. Silence and clipping are sampled.

## Coverage and limitations

- Pipecat supplies isolated hold/end clips. LiveKit supplies causal continuation and final-silence spans grouped by inferred conversations.
- Explicit speaker IDs are unavailable, so speaker-independent splitting cannot be verified.
- LiveKit real/synthetic provenance is not declared in the indexed schema and remains unknown.
- Only English and Spanish include non-synthetic Pipecat records in this scope; the other training languages are synthetic in the source metadata.
- Backchannels, overlap, interruptions, recording device, and acoustic environment are not explicitly labeled.
- Exact decoded-content duplicates across different codecs were not measured.

## Decision

**Suitable with stated constraints**

The indexed files are complete, readable, group-disjoint where conversation IDs exist, and contain both labels in every language and partition. The main constraints are synthetic-heavy training, unavailable speaker IDs, unknown LiveKit provenance, and incomplete difficult-case annotations.

The training run uses a deterministic full-coverage grouped shuffle. Temperature resampling is not used in this one-epoch run because it would either omit examples or repeat mostly synthetic minority-language examples; macro per-language validation prevents English volume from selecting the checkpoint alone.
