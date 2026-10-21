# Pipecat training data adoption audit

Audit date: 2026-10-21  
Intended use: initial English audio-only end-of-turn classifier  
Verdict: **Suitable for pipeline smoke tests and a toy baseline only; insufficient as the primary training dataset.**

## Identity and scope

- Source: `pipecat-ai/smart-turn-data-v3.2-train`, default configuration, train split.
- Local population audited: all 50 downloaded English recordings and all 50 metadata rows.
- Selection: deterministic convenience sample consisting of the first 25 English
  complete and first 25 English incomplete rows found within Hugging Face rows 0-299.
- License recorded locally: CC BY 4.0.
- Immutable upstream revision: **not recorded**. Local manifest SHA-256:
  `223579f7f771aef1ba5a2bb3a39516c2e626bfb005f6331bb9b8502d9329b74c`.
- Record unit: one isolated user speech sample with one clip-level
  complete/incomplete label. No conversation, speaker, or session unit is supplied.

## Observed counts

| Measure | Full local population result |
|---|---:|
| Metadata records | 50 |
| Present, non-empty, decodable audio files | 50/50 |
| Total audio | 421.397 s (0.117 h) |
| Duration, min / mean / max | 2.475 / 8.428 / 16.000 s |
| Complete | 25 records; 215.484 s |
| Incomplete | 25 records; 205.913 s |
| Explicitly real | 31 records; 290.159 s |
| Explicitly synthetic | 19 records; 131.238 s |
| Unknown provenance | 0 |
| English | 50/50 |
| Mono, 16 kHz, FLAC, 16-bit source | 50/50 |
| Exact duplicate IDs | 0 groups / 0 excess records |
| Exact duplicate audio SHA-256 | 0 groups / 0 excess records |
| Missing audio | 0 |
| Empty files | 0 |

All 50 stored SHA-256 checksums pass. Twenty-six clips are shorter than eight
seconds, 24 are at least eight seconds, and one is exactly 16 seconds.

## Schema and missingness

| Field | Missing/null | Finding |
|---|---:|---|
| `id` | 0/50 | Unique for every row |
| `language` | 0/50 | All `eng` |
| `endpoint_bool` / `label` | 0/50 | Mapping is consistent for every row |
| `synthetic` | 0/50 | 19 synthetic, 31 real |
| `source_dataset` | 0/50 | Eight source families |
| `spoken_text` | 50/50 | Transcript unavailable, not an empty transcript |
| `midfiller` | 41/50 | Null must remain unknown, not false |
| `endfiller` | 41/50 | Null must remain unknown, not false |
| speaker ID | 50/50 | Not supplied |
| conversation/session ID | 50/50 | Not supplied |
| silence spans / word timestamps | 50/50 | Not supplied |

Source coverage is uneven: `liva_1` contributes 20 records and 241.081 seconds;
`midcentury_1` contributes 10; `chirp3_1` contributes 9; the remaining five
sources contribute 1-4 records each. Four synthetic `rime_2` records occur only
in the incomplete class, and `orpheus_endfiller_1` occurs only in complete.
Source and label are therefore partially confounded in this small sample.

## Turn-detection coverage

The binary target is present and balanced, but the metadata cannot measure:

- internal continuation-pause counts or duration distribution;
- trailing-silence distribution by label;
- backchannels, overlap, interruptions, or truncation;
- speaker- or conversation-independent splitting;
- repeated speakers or source recordings across upstream splits;
- transcript/label agreement.

Those are unavailable measures, not zero-count phenomena. The files decode and
are non-empty, but speech activity and fully silent examples were not measurable
with the current metadata-only decoder. Clipping analysis was also unavailable
because the local CoreAudio clipping tool cannot analyze these FLAC payloads.

## Suitability

The sample is suitable for validating the downloader, loader, label mapping,
padding/truncation code, one optimizer update, and end-to-end experiment wiring.
It is not sufficient evidence for a trained production model: 0.117 hours is too
small, the sample is not representative, independent grouping cannot be
verified, and pause-level difficult cases are not annotated.

Do not create a random train/validation/test split from these 50 rows and report
it as a reliable generalization result. Use LiveKit only as held-out causal
evaluation; do not tune repeatedly on its 50-clip sample until it effectively
becomes training data.

## Highest-impact fixes

1. Fetch a substantially larger English Pipecat population and pin the exact
   Hugging Face revision/commit in the manifest.
2. Obtain or derive verified speaker/conversation/source-recording group IDs
   before defining independent train/validation splits.
3. Audit trailing silence and internal pauses with the exact production VAD;
   normalize the observation point so the label cannot be inferred from silence
   duration alone.
4. Deduplicate the expanded population by ID and decoded-audio content, then
   check duplicates and source lineage across every split.
5. Preserve source and synthetic metadata and report metrics separately for
   real/synthetic and each sufficiently populated source.
6. Keep null filler values as unknown. Do not impute them as `false`.

## Reproduction artifacts and commands

- Metadata: `data/pipecat_smart_turn_v3_2_sample/manifest.csv`
- Audio validation: `data/pipecat_smart_turn_v3_2_sample/audio_validation.tsv`
- Content hashes: `data/pipecat_smart_turn_v3_2_sample/SHA256SUMS`
- Indexed metadata: `data/pipecat_smart_turn_v3_2_sample/dataset.sqlite3`

Core verification commands:

```bash
python3 scripts/fetch_pipecat.py --index-only
shasum -a 256 -c data/pipecat_smart_turn_v3_2_sample/SHA256SUMS
sqlite3 data/pipecat_smart_turn_v3_2_sample/dataset.sqlite3 \
  'SELECT label, COUNT(*) FROM examples GROUP BY label;'
python3 -m unittest discover -s tests -v
```
