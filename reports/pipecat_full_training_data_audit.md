# Pipecat full training-data audit

## Verdict

Suitable for audio-only complete-versus-incomplete turn training, with important
constraints. The English subset is large and nearly label-balanced, all sampled
audio decoded consistently, and the download is pinned to an immutable revision.
The release has no transcripts, word timestamps, pause spans, speaker IDs, or
conversation IDs. A speaker- or conversation-independent split therefore cannot
be guaranteed from the supplied metadata.

## Audited snapshot

- Dataset: `pipecat-ai/smart-turn-data-v3.2-train`
- Revision: `f8c95d284595defb6b88a715c356cbc5c31b7ad1`
- License recorded by the fetcher: CC BY 4.0
- Storage: 83 Parquet shards, 41,400,926,042 bytes (38.56 GiB)
- Integrity: 83/83 shards present; zero size mismatches
- Records: 270,946 total; 270,946 unique IDs; zero duplicate IDs

## Observed counts

| Slice | Records |
|---|---:|
| Complete | 134,305 |
| Incomplete | 136,641 |
| Synthetic | 223,343 |
| Real | 47,603 |
| English | 65,802 |
| English complete | 32,443 |
| English incomplete | 33,359 |
| English synthetic | 22,185 |
| English real | 43,617 |

The full release contains 23 language codes. English is the largest slice.
Within English, source counts range from 679 to 30,922 records, so source-aware
reporting is needed to expose source imbalance.

## Schema and missing fields

Required label and provenance fields (`id`, `language`, `endpoint_bool`,
`synthetic`, and `dataset`) have zero missing values. `spoken_text` is absent in
all 270,946 records. `midfiller` and `endfiller` are each absent in 55,881
records. No speaker ID, conversation ID, word timestamps, or explicit pause
duration is supplied.

## Audio validation

One deterministic row from every shard was decoded (83 files). All 83 were
mono, 16 kHz, FLAC/PCM-16; none failed. Sampled durations ranged from 2.08 to
16.00 seconds, with a mean of 7.61 seconds. This is a shard-wide sample, not a
claim that every audio payload was decoded.

## Duplicate and leakage risk

Metadata IDs are unique. A full SHA-256 scan of all 65,802 English encoded audio
payloads found 65,802 unique hashes, zero duplicate groups, zero cross-split
duplicates, and zero conflicting duplicate labels. Perceptual similarity and
re-encoded waveform equality remain unmeasured. The lack of speaker,
conversation, and source-recording identifiers prevents group-independent split
verification. All 10 broad English provenance categories occur in both train and
validation. Keep the separate external source isolated and report results by
real/synthetic provenance and source.

## Reproducibility artifacts

The ignored local directory
`data/pipecat_smart_turn_v3_2_full/snapshot_manifest.json` records the immutable
revision and every expected shard and byte size.
`src/turn_detector/data_fetching/fetch_pipecat.py`
recreates and verifies the snapshot; `CanonicalPipecatDataset` uses the
canonical index to stream the selected split and performs training-time decoding
and fixed-window preprocessing.
`reports/english_split_duplicate_audit.json` records the full English exact-audio
duplicate scan.
