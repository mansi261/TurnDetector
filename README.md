# TurnDetector

## Clone and run

### Prerequisites

- Git
- Docker Desktop, or Docker Engine with Docker Compose v2
- At least 2 GB of free disk space for the image and build cache

The model runs locally and does not require an API key.

### Start the service

```bash
git clone https://github.com/mansi261/TurnDetector.git
cd TurnDetector
docker compose up --build -d
```

The first build downloads the Python base image and installs the pinned serving
dependencies. Later starts reuse Docker's build cache.

Wait until the model is ready:

```bash
curl --fail http://127.0.0.1:8000/health
curl --fail http://127.0.0.1:8000/ready
```

Then use:

- Demo and API dashboard: `http://127.0.0.1:8000`
- API metrics: `http://127.0.0.1:8000/metrics`

The demo model dropdown selects either **English** or **Multilingual — 14
languages** before the microphone session starts. The selected model remains
fixed for that session.

To call the model with a 16 kHz WAV or FLAC file:

```bash
curl --fail --request POST \
  --form "audio=@/absolute/path/to/audio.wav" \
  http://127.0.0.1:8000/predict
```

Add `?model=multilingual` to use the multilingual artifact. English is the
default.

The response contains `p_end`, the `hold` or `end` decision, selected model ID,
threshold, model version, and preprocessing/inference timing.

View container logs or stop the service with:

```bash
docker compose logs --follow
docker compose down
```

If port 8000 is already in use, select another host port:

```bash
TURN_DETECTOR_PORT=8001 docker compose up --build -d
```

In that case, open `http://127.0.0.1:8001` and use port 8001 for the API calls.

## Project layout

```text
src/turn_detector/
├── data_fetching/
│   ├── fetch_pipecat.py
│   ├── fetch_livekit.py
│   ├── audit_livekit.py
│   └── build_canonical_index.py
└── data_loader/
    └── data.py
```

Data acquisition and index construction are separate from model-facing data
loading. Downloaded data and generated indexes live under `data/` and are not
tracked by Git.

## Fetch the complete training dataset

Install the download and Parquet dependencies:

```bash
python3 -m pip install -r requirements-data.txt
```

Inspect the immutable revision, shard count, download size, and free disk space:

```bash
PYTHONPATH=src python3 -m turn_detector.data_fetching.fetch_pipecat --dry-run
```

Download and verify all 83 shards:

```bash
PYTHONPATH=src python3 -m turn_detector.data_fetching.fetch_pipecat
```

The complete snapshot defaults to `data/pipecat_smart_turn_v3_2_full`. It pins
revision `f8c95d284595defb6b88a715c356cbc5c31b7ad1`, resumes interrupted
downloads through the Hugging Face cache, verifies shard sizes, and records the
inventory in `snapshot_manifest.json`. Audio remains embedded in Parquet.

## Fetch the public English evaluation dataset

```bash
PYTHONPATH=src python3 -m turn_detector.data_fetching.fetch_livekit --dry-run
PYTHONPATH=src python3 -m turn_detector.data_fetching.fetch_livekit
```

The fetcher downloads all 400 publicly available English turns at revision
`ca9d98a9686b920a2d8c9eb984224ba9be74e4dd` into
`data/livekit_eot_bench_ENG_public_400`. It resumes through the local cache and
verifies both byte sizes and SHA-256 hashes. The upstream source contains other
languages, but this snapshot and experiment are explicitly `ENG` only.

## Build the canonical metadata index

```bash
PYTHONPATH=src python3 -m turn_detector.data_fetching.build_canonical_index
```

This creates `data/canonical/dataset.sqlite3` and `summary.json`. The SQLite
database contains canonical labels, deterministic train/validation assignments,
language and provenance metadata, and Parquet row locators. It does not contain
audio. The indexes support language lookup and the primary
`(split, language, label)` training query.

Training labels are normalized to `hold=0` and `end=1`. English evaluation
silence spans use the same labels and are divided by inferred conversation group
into `ENG_evaluation_dev`, `ENG_holdout_a`, and `ENG_holdout_b`. Previously
inspected examples are forced into development. No fixed silence duration or
decision timestamp is stored in the canonical representation; causal replay
chooses observation timestamps independently. Transcripts and dialogue context
are not exposed to either source.

The generated `ENG_evaluation_partitions.json` records the deterministic group
assignments. The dataset audit is reproduced with:

```bash
PYTHONPATH=src python3 -m turn_detector.data_fetching.audit_livekit
```

## Load complete data

Install the training dependencies:

```bash
python3 -m pip install -r requirements-train.txt
```

The canonical PyTorch iterable loader streams indexed rows from the original
Parquet snapshot:

```python
from torch.utils.data import DataLoader
from turn_detector.data_loader import CanonicalPipecatDataset

training_data = CanonicalPipecatDataset(
    "data/canonical/dataset.sqlite3",
    split="train",
    language="eng",
)
loader = DataLoader(training_data, batch_size=16, num_workers=4)

for batch in loader:
    # waveform: [batch, 128000], representing 8 seconds at 16 kHz
    # label: 0=hold, 1=end
    pass
```

The loader filters before decoding, assigns different shards to worker
processes, converts audio to mono, takes the most recent eight seconds, and
left-pads shorter clips. It never loads the complete snapshot into memory.

## Tests

```bash
python3 -m unittest discover -s tests -v
```

Tests requiring the ignored full snapshot or optional data packages skip when
those local resources are unavailable.

## Export and serve Baseline v0

Export a checkpoint and verify FP32 and INT8 parity:

```bash
PYTHONPATH=src python3 -m turn_detector.training.export \
  --checkpoint experiments/english_whisper_tiny_v1/full_interrupted_drop_last/checkpoints/best.pt \
  --output-dir experiments/english_whisper_tiny_v1/baseline_v0/export
```

Only artifacts whose parity check passes should be served. Baseline v0 uses the
verified FP32 artifact at threshold `0.5`:

```bash
TURN_DETECTOR_CHECKPOINT=experiments/english_whisper_tiny_v1/baseline_v0/export/model.fp32.onnx \
TURN_DETECTOR_METADATA=experiments/english_whisper_tiny_v1/baseline_v0/export/model_metadata.json \
TURN_DETECTOR_MODEL_VERSION=english-whisper-tiny-baseline-v0 \
TURN_DETECTOR_THRESHOLD=0.5 \
PYTHONPATH=src uvicorn turn_detector.serving.app:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000` for the microphone probability timeline. Select
English or multilingual before starting the microphone. The frontend sends a
rolling audio window; its chart cadence is not a silence rule.
Use the **API dashboard** tab to inspect live request, p50/p95 latency,
throughput, probability-distribution, and decision metrics. Error and p99
telemetry remain available from `GET /metrics` for diagnostics.

- `GET /health` is a process liveness check.
- `GET /ready` lists the loaded models and confirms their identity and threshold.
- `GET /metrics` returns bounded, process-local API and model-output telemetry.
- `POST /predict?model=english|multilingual` accepts a 16 kHz WAV or FLAC file
  in the `audio` multipart field.
- `POST /stream?model=english|multilingual` creates an isolated, model-pinned
  microphone session. Send new 16 kHz mono
  PCM16 samples to `POST /stream/{session_id}/predict`, then release the state
  with `DELETE /stream/{session_id}`. The server reuses unchanged Whisper feature
  frames while preserving the same eight-second input window.

The local telemetry retains at most 5,000 request summaries and resets when the
service restarts. It is intended to demonstrate the monitoring contract, not to
replace persistent deployment metrics and alerting.

Run the development-only threshold optimization without scoring either holdout:

```bash
PYTHONPATH=src .venv/bin/python -m turn_detector.evaluation.optimize_threshold
```

The observed trade-off is documented in
[`reports/ENG_threshold_optimization_v1.md`](reports/ENG_threshold_optimization_v1.md).

Build the unified classification, causal replay, and serving-policy comparison:

```bash
PYTHONPATH=src .venv/bin/python -m turn_detector.evaluation.build_unified_report
```

The comparison keeps threshold `0.5` active and treats `0.825` as an
unpromoted development candidate. Its report is
[`reports/ENG_unified_three_level_evaluation_v1.md`](reports/ENG_unified_three_level_evaluation_v1.md).

Compare Baseline v0 and the fully fine-tuned epoch-3 checkpoint on the complete
English validation split at the unchanged threshold `0.5`:

```bash
PYTHONPATH=src .venv/bin/python -m turn_detector.evaluation.compare_validation_checkpoints --device mps
```

The recorded comparison is
[`reports/ENG_epoch3_validation_comparison_v1.md`](reports/ENG_epoch3_validation_comparison_v1.md).

Compare the same checkpoints using causal development replay at threshold `0.5`:

```bash
PYTHONPATH=src .venv/bin/python -m turn_detector.evaluation.compare_causal_checkpoints
```

The event-level trade-off is documented in
[`reports/ENG_epoch3_causal_development_comparison_v1.md`](reports/ENG_epoch3_causal_development_comparison_v1.md).

Docker uses `model/serving_models.json` to load both selectable artifacts once
at startup. Single-model deployments can omit `TURN_DETECTOR_MODEL_MANIFEST`
and configure `TURN_DETECTOR_CHECKPOINT`, `TURN_DETECTOR_METADATA`,
`TURN_DETECTOR_MODEL_VERSION`, and `TURN_DETECTOR_THRESHOLD`. The service
rejects a checkpoint whose SHA-256 is not declared in its metadata.

Call the service with a file:

```bash
python3 -m pip install -r requirements-tools.txt
PYTHONPATH=src python3 -m turn_detector.serving.client --audio sample.wav
```

The client records the source and transmitted payload hashes, audio shape,
service/model identity, prediction, server-stage timing, and client-observed
HTTP timing. It also verifies that readiness and prediction used the same model
version and threshold.

Inspect a causal snapshot using only audio available through five seconds:

```bash
PYTHONPATH=src python3 -m turn_detector.serving.client \
  --audio sample.wav \
  --end-seconds 5.0 \
  --expect-model-version english-whisper-tiny-baseline-v0 \
  --expect-threshold 0.5 \
  --output experiments/manual_inference/sample_at_5s.json
```

Without `--end-seconds`, the complete file is sent and the service scores its
latest eight seconds. With `--end-seconds`, the client sends only the prefix
available at that timestamp; the service still applies its configured latest
eight-second window. The input must be 16 kHz WAV or FLAC.

Verify PyTorch, direct ONNX, and API parity on causal development inputs:

```bash
PYTHONPATH=src python3 -m turn_detector.serving.verify_parity \
  --training-checkpoint experiments/english_whisper_tiny_v1/full_interrupted_drop_last/checkpoints/best.pt \
  --serving-checkpoint experiments/english_whisper_tiny_v1/baseline_v0/export/model.fp32.onnx \
  --metadata experiments/english_whisper_tiny_v1/baseline_v0/export/model_metadata.json \
  --canonical-index data/canonical/dataset.sqlite3 \
  --output experiments/english_whisper_tiny_v1/baseline_v0/serving_parity.json
```

Compare repeated full-window requests with incremental feature reuse:

```bash
PYTHONPATH=src python3 -m turn_detector.serving.benchmark \
  --audio experiments/english_whisper_tiny_v1/baseline_v0/serving_benchmark_input.wav \
  --requests 20 \
  --concurrency 1 \
  --streaming-replays 2 \
  --output experiments/english_whisper_tiny_v1/baseline_v0/streaming_preprocessing_benchmark_v1.json
```

## Reproduce the 14-language model

The multilingual cycle is isolated from the English index and checkpoint. It
covers `ara`, `deu`, `eng`, `fra`, `hin`, `ind`, `ita`, `jpn`, `kor`, `nld`,
`por`, `spa`, `tur`, and `zho`.

The exact English initialization and selected multilingual PyTorch checkpoints
are packaged under `model/checkpoints/` for reproducibility and are excluded
from the runtime container. Their SHA-256 values are verified by the training
and evaluation commands.

Fetch LiveKit, build the canonical index, and run the audit:

```bash
PYTHONPATH=src python3 -m turn_detector.data_fetching.fetch_livekit \
  --output data/livekit_eot_bench_MULTI_public \
  --languages en,ar,de,es,fr,hi,id,it,ja,ko,nl,pt,tr,zh
PYTHONPATH=src python3 -m turn_detector.data_fetching.build_multilingual_index
PYTHONPATH=src python3 -m turn_detector.data_fetching.audit_multilingual
```

Smoke-test the actual training path, then run the complete frozen-encoder pass:

```bash
PYTHONPATH=src python3 -m turn_detector.training.train_multilingual \
  --smoke --device mps --output-name smoke_integrity_v1 \
  --initialization-checkpoint model/checkpoints/english_baseline_v0.pt
PYTHONPATH=src python3 -m turn_detector.training.train_multilingual \
  --device mps --output-name full_v1 \
  --initialization-checkpoint model/checkpoints/english_baseline_v0.pt
```

Use `--device cpu` on systems without Apple MPS. Interrupted full runs resume
from `full_v1/checkpoints/latest.pt` with `--resume-from` and the same
`--output-name`.

Compare Baseline v0 and the candidate on multilingual development data, then
export and verify the candidate:

```bash
PYTHONPATH=src python3 -m turn_detector.evaluation.evaluate_multilingual \
  --baseline-checkpoint model/checkpoints/english_baseline_v0.pt \
  --candidate-checkpoint model/checkpoints/multilingual_v1.pt \
  --output experiments/multilingual_whisper_tiny_v1/full_v1/evaluation/development_report.json \
  --device mps
PYTHONPATH=src python3 -m turn_detector.training.export_multilingual \
  --checkpoint model/checkpoints/multilingual_v1.pt
PYTHONPATH=src python3 -m turn_detector.serving.verify_multilingual_parity \
  --training-checkpoint model/checkpoints/multilingual_v1.pt \
  --serving-checkpoint model/multilingual_v1/model.fp32.onnx \
  --metadata model/multilingual_v1/model_metadata.json \
  --output experiments/multilingual_whisper_tiny_v1/full_v1/serving_parity.json
```

The holdout command requires a frozen manifest and is intended for one final
run only. Do not use its result to tune the same candidate.

```bash
PYTHONPATH=src python3 -m turn_detector.evaluation.freeze_multilingual_candidate \
  --baseline-checkpoint model/checkpoints/english_baseline_v0.pt \
  --candidate-checkpoint model/checkpoints/multilingual_v1.pt \
  --development-report experiments/multilingual_whisper_tiny_v1/full_v1/evaluation/development_report.json \
  --serving-metadata model/multilingual_v1/model_metadata.json \
  --output experiments/multilingual_whisper_tiny_v1/full_v1/frozen_candidate.json
PYTHONPATH=src python3 -m turn_detector.evaluation.evaluate_multilingual \
  --baseline-checkpoint model/checkpoints/english_baseline_v0.pt \
  --candidate-checkpoint model/checkpoints/multilingual_v1.pt \
  --partition MULTI_holdout --allow-holdout \
  --freeze-manifest experiments/multilingual_whisper_tiny_v1/full_v1/frozen_candidate.json \
  --output experiments/multilingual_whisper_tiny_v1/full_v1/evaluation/holdout_report.json \
  --device mps
```

The default Docker service loads both English and multilingual artifacts. Use
the model dropdown in the demo or the `model` API query parameter to choose one.

## Docker

The repository includes the verified FP32 serving artifact, its preprocessing
configuration, and its metadata under `model/`. From a fresh clone, build and
start the CPU service with:

```bash
docker compose up --build
```

Open `http://127.0.0.1:8000` for the demo and API dashboard. Verify the running
container with:

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/ready
```

Stop it with `docker compose down`.

If dependency installation fails, show the package-level error rather than only
Docker's final summary line:

```bash
docker compose --progress=plain build --no-cache
```

The image retries slow package downloads and uses a two-minute download timeout.

The Compose defaults serve Baseline v0 at threshold `0.5`. Override the host
port, threshold, version, or model directory without changing code:

```bash
TURN_DETECTOR_PORT=8080 \
TURN_DETECTOR_THRESHOLD=0.6 \
TURN_DETECTOR_MODEL_VERSION=english-whisper-tiny-candidate \
TURN_DETECTOR_MODEL_DIR=/absolute/path/to/export \
docker compose up --build
```

ONNX Runtime CPU settings are also configurable without rebuilding:

```bash
TURN_DETECTOR_ORT_INTRA_OP_THREADS=2 \
TURN_DETECTOR_ORT_INTER_OP_THREADS=1 \
TURN_DETECTOR_ORT_EXECUTION_MODE=sequential \
docker compose up -d --force-recreate
```

Use `0` for either thread count to retain ONNX Runtime's automatic default.
The Docker default is the measured two-intra-op/one-inter-op configuration. The
active settings and visible CPU count are returned by `/ready`.

The mounted model directory must contain the selected ONNX checkpoint, metadata,
and the metadata-referenced `preprocessor/` directory. If filenames differ, set
`TURN_DETECTOR_CHECKPOINT_FILE` and `TURN_DETECTOR_METADATA_FILE`. The container
runs as a non-root user, loads the model once during startup, and exposes the
health, readiness, file-prediction, streaming, and metrics routes on port 8000
inside the container.
