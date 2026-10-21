# TurnDetector

## Prepare the Pipecat data

Fetch a balanced English sample and build its SQLite index:

```bash
python3 scripts/fetch_pipecat.py
```

The command is resumable. It refreshes Hugging Face's expiring asset URLs,
retries transient failures, downloads through temporary files, verifies FLAC
headers, and skips files that have already completed.

To rebuild only the database from the existing manifest and audio:

```bash
python3 scripts/fetch_pipecat.py --index-only
```

## Load data in model code

Run code with `src` on the Python path:

```python
from turn_detector import PipecatDataset

dataset = PipecatDataset("data/pipecat_smart_turn_v3_2_sample/dataset.sqlite3")
print(dataset.class_counts())

for example in dataset.iter_examples(label="complete"):
    audio_path = example.audio_path
    target = int(example.endpoint_bool)
```

Example:

```bash
PYTHONPATH=src python3 -c 'from turn_detector import PipecatDataset; d=PipecatDataset("data/pipecat_smart_turn_v3_2_sample/dataset.sqlite3"); print(len(d), d.class_counts())'
```

Run the loader tests with:

```bash
python3 scripts/fetch_pipecat.py --index-only
python3 -m unittest discover -s tests -v
```

## Build the shared training/evaluation schema

```bash
python3 scripts/build_canonical_dataset.py
```

This creates `data/canonical/train.jsonl`, `evaluation.jsonl`, `summary.json`,
and a combined `dataset.sqlite3`. Pipecat's complete/incomplete labels become
`end`/`hold`. Each LiveKit pause becomes a causal evaluation record scored 200
ms after silence begins. Transcripts and dialogue context are intentionally not
exposed in either split for the initial audio-only model.
