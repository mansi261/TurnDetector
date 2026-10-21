# Pipecat Smart Turn v3.2 - 50-example audit sample

This directory contains 50 English training examples from
[`pipecat-ai/smart-turn-data-v3.2-train`](https://huggingface.co/datasets/pipecat-ai/smart-turn-data-v3.2-train),
downloaded on 2026-10-20 for schema and data-quality evaluation.

## Why this sample

The project targets an English end-of-turn detector. The sample is intentionally
balanced by target label:

- 25 `complete` examples (`endpoint_bool=true`)
- 25 `incomplete` examples (`endpoint_bool=false`)
- 31 real examples and 19 synthetic examples
- 421.4 seconds of audio in total (7.02 minutes)

This is a deterministic convenience sample: the first 25 English rows per class found
in Hugging Face rows 0-299. It is useful for schema, loader, and qualitative audio
audits, but it is not a statistically representative evaluation split. Do not report
model accuracy from these 50 examples as a benchmark.

## Files

- `audio/complete/`: audio labelled as a finished turn
- `audio/incomplete/`: audio labelled as a turn that should continue
- `manifest.csv`: flat metadata for spreadsheet inspection
- `metadata.jsonl`: the same sample metadata in training-friendly JSON Lines format
- `audio_validation.tsv`: local decode and technical-format checks for every file

## Source schema

| Field | Meaning | Observation in this sample |
|---|---|---|
| `audio` | FLAC audio payload | Stored locally under `audio/` |
| `id` | UUID-like example identifier | Present for all 50 rows |
| `language` | ISO 639-3 language code | `eng` for all 50 rows |
| `endpoint_bool` | End-of-turn target | Balanced: 25 true / 25 false |
| `midfiller` | Filler appears within the utterance | Null for 41/50 rows |
| `endfiller` | Filler appears at the end | Null for 41/50 rows |
| `synthetic` | Whether audio was synthesized | 19 true / 31 false |
| `spoken_text` | Transcript | Null for all 50 rows |
| `dataset` | Upstream source/provenance | Eight sources represented |

For convenience, the local manifests add `sample_no`, `hf_row_idx`, `label`,
`relative_audio_path`, and rename `dataset` to `source_dataset` in JSONL.

## Technical validation

All 50 files decode successfully and are mono, 16 kHz FLAC from 16-bit sources.
Durations range from 2.475 to 16.000 seconds, with an 8.428-second mean. No empty or
nonstandard files were found.

## Initial quality findings

1. The target and provenance fields are clean enough to build an audio classifier.
2. `spoken_text` is entirely null, so this dataset cannot directly support a
   text-only baseline or transcript/label consistency review without running ASR.
3. `midfiller` and `endfiller` are sparsely populated and should not be interpreted as
   false when null. Preserve the three-state values or add an explicit `unknown` flag.
4. The sample mixes real and synthetic audio. Split by source/speaker lineage—not by
   random row—to reduce leakage and measure synthetic-to-real generalization.
5. Source representation is uneven (`liva_1` contributes 20/50 rows). Use grouped
   evaluation and per-source metrics before drawing quality conclusions.
6. The production model consumes at most the latest eight seconds. Several source
   clips exceed eight seconds, so preprocessing and evaluation must use the same
   left-padding/truncation rule as inference.
7. A listening audit is still required for label correctness, trailing-silence
   consistency, clipping, background noise, and speaker/source duplicates.

## License and attribution

The source dataset is published under CC BY 4.0. Retain attribution to Pipecat and the
upstream contributors listed on the dataset card when redistributing or presenting
these samples.
