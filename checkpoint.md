# Implementation checkpoints

The run artifacts referenced here are written under the ignored `experiments/`
directory. This file records the reproducible path back to each result without
tracking model binaries.

## Experiment contract

- Primary scope: English audio-only turn detection.
- Labels: `hold=0`, `end=1`.
- Input: the latest 8 seconds of 16 kHz mono audio, left-padded with silence.
- Training/validation: deterministic assignments in the canonical SQLite index.
- External evaluation: independent audio, scored causally with no future samples.
- Backbone: generic `openai/whisper-tiny`; no task-specific checkpoint is used to initialize training.
- Selection: lowest validation loss; decision threshold tuned on validation only.
- `ENG_evaluation_dev` is reusable development data; Holdout A and Holdout B
  are reserved for frozen-candidate reporting.

## Phase status

- [x] Full source snapshot downloaded and revision pinned.
- [x] Canonical SQLite index created with language and split indexes.
- [x] English train, validation, and external record counts verified.
- [x] Complete public ENG evaluation sample fetched and SHA-256 verified.
- [x] ENG development and two conversation-disjoint holdouts frozen.
- [x] Dataset suitability audit recorded.
- [x] Exact English audio duplicate audit completed with no train/validation overlap.
- [x] Canonical PyTorch loaders verified.
- [x] End-to-end smoke training and checkpoint reload verified.
- [x] Frozen-encoder Baseline v0 selected on validation at threshold `0.5`.
- [x] Complete encoder fine-tuning run.
- [x] Causal ENG development replay compared with fixed-silence baselines.
- [x] Development-only threshold sweep completed; both holdouts remain unscored.
- [x] Unified three-level Baseline v0 and threshold `0.825` comparison recorded.
- [x] FP32 ONNX export verified against PyTorch.
- [ ] INT8 export corrected and revalidated.
- [x] FastAPI service verified with the FP32 artifact.
- [x] Live API metrics endpoint and monitoring dashboard implemented.
- [x] Standalone inference client records input/artifact identity, causal-prefix predictions, and client/server timing.
- [x] CPU-only container build and mounted-artifact inference verified.
- [x] Cold/warm/concurrent API benchmark recorded.
- [x] Serving preprocessing, artifact identity, causal parity, and holdout isolation reviewed.
- [x] Production monitoring, rollback, and improvement plan documented.

## Fixed data identity

- Dataset revision: `f8c95d284595defb6b88a715c356cbc5c31b7ad1`
- Canonical schema version: 3
- English training records: 59,280
- English validation records: 6,522
- ENG external source: `livekit/eot-bench-data` revision `ca9d98a9686b920a2d8c9eb984224ba9be74e4dd`
- ENG external turns: 400 across 96 inferred conversation groups
- ENG external decisions: 1,250 (`850 hold`, `400 end`)
- ENG development: 198 turns, 46 groups, 672 decisions
- ENG Holdout A: 102 turns, 25 groups, 291 decisions
- ENG Holdout B: 100 turns, 25 groups, 287 decisions
- External timing: complete silence spans are retained; no fixed decision time is stored.
- Known limitation: source data does not provide speaker identities, so the deterministic split cannot be speaker-disjoint.

## Run records

### Interrupted full run

- Preserved at: `experiments/english_whisper_tiny_v1/full_interrupted_drop_last`
- Stopped after epoch 1 because `drop_last=True` processed 59,264 of 59,280 training rows.
- Diagnostic validation: loss 0.1742, AUROC 0.9800.
- Corrective action: retain final partial worker batches and restart from generic initialization.
- Current decision: use its best checkpoint as Baseline v0 at threshold `0.5`; defer retraining.
- Optimization and leakage backlog: `optimization_tracker.md`.

### Three-epoch v1 run — completed

- Run directory: `experiments/english_whisper_tiny_v1/full_three_epoch_v1`.
- Epoch 1 completed on all 59,280 records and its validation/checkpoints are intact.
- Epoch 1 validation loss: `0.1741907`; accuracy: `93.24%`; AUROC: `0.9800`.
- Training was stopped during epoch 2 at step 1,000 because the local machine
  needed compute and memory released.
- The partial epoch 2 update is intentionally discarded because no mid-epoch
  checkpoint was written.
- Resume support was smoke-tested and the run restarted from
  `checkpoints/latest.pt` at the beginning of epoch 2.
- Epoch 2 uses the unfrozen encoder optimizer and the two-epoch cosine schedule.
- The original checkpoint did not store RNG state, so this is a controlled
  epoch-boundary continuation but not bit-for-bit identical to an uninterrupted run.
- `resume_events.jsonl` records the checkpoint hash, command, code identity, and
  environment. The full run completed checkpoint reload verification.
- Epoch 3 validation at threshold `0.5`: 94.99% accuracy, 93.61% end precision,
  96.19% end recall, 94.89% F1, 98.79% AUROC, and 6.14% hold false-positive rate.
- Epoch 1 remains `best.pt` under the configured validation-loss selection rule;
  epoch 3 is preserved as `latest.pt` for explicit candidate evaluation.
- Results: `experiments/english_whisper_tiny_v1/full_three_epoch_v1/results.json`.

### Normalized preprocessing v2 — implementation ready

- Config: `configs/english_whisper_tiny_v2_normalized.json`.
- Change: zero-mean/unit-variance normalization on the available latest audio
  window before left-zero padding.
- Scope: training, validation, causal evaluation, export metadata, parity checks,
  and API inference share the same versioned setting.
- Compatibility: v1 and legacy serving artifacts continue to use no normalization.
- Smoke run: 64 train and 64 validation records completed frozen and unfrozen
  epochs with finite losses and checkpoint reload parity.
- Smoke artifact: `experiments/english_whisper_tiny_v2_normalized/normalized_preprocessing_smoke`.
- Status: no full v2 run or LiveKit development evaluation has been performed.

### Epoch-3 fixed-threshold validation comparison

- Both Baseline v0 epoch 1 and the fully fine-tuned epoch 3 checkpoint were
  rescored on the same 6,522 Pipecat English validation records at threshold `0.5`.
- Epoch 3 improved accuracy by 1.75 percentage points, end precision by 2.37,
  end recall by 1.05, F1 by 1.73, and AUROC by 0.79.
- Hold false-positive rate decreased from 8.55% to 6.14%.
- Probability quality is mixed: Brier score improved, while BCE loss and 10-bin
  expected calibration error worsened.
- Epoch 3 remains a development candidate; neither LiveKit development nor the
  two holdouts was accessed in this comparison.
- Report: `reports/ENG_epoch3_validation_comparison_v1.md`.
- Raw artifacts: `experiments/english_whisper_tiny_v1/validation_comparison_v1/`.

### Epoch-3 causal development comparison

- Baseline v0 and epoch 3 were replayed on the identical 672
  `ENG_evaluation_dev` events at threshold `0.5` and 100 ms cadence.
- Premature responses increased from 114/474 (24.05%) to 125/474 (26.37%).
- Missed endings decreased from 29/198 (14.65%) to 18/198 (9.09%).
- Total causal event errors remained 143; median and p95 detected-end delay
  remained 0 ms and 400 ms in simulated event time.
- Epoch 3 fixed 17 premature events but introduced 28 new premature events; it
  recovered 14 endings but introduced 3 new misses.
- Decision: do not promote epoch 3. Baseline v0 remains the serving policy.
- Both holdouts remain unscored.
- Report: `reports/ENG_epoch3_causal_development_comparison_v1.md`.
- Raw artifacts: `experiments/english_whisper_tiny_v1/causal_comparison_v1/`.

### Baseline v0 serving

- FP32 artifact: `experiments/english_whisper_tiny_v1/baseline_v0/export/model.fp32.onnx`
- PyTorch/ONNX maximum probability difference: 0.0000023; zero decision changes.
- Direct ONNX/API maximum probability difference: 0.0 across causal hold/end development inputs and a short boundary input.
- INT8 diagnostic: not serving-eligible due to 0.487 maximum difference and 2/16 decision changes.
- Serving threshold: 0.5.
- Model startup: 1.91 seconds on CPU with filesystem cache warm.
- Cold HTTP inference: 178.0 ms.
- Warm concurrency 1: p50 42.1 ms, p95 96.3 ms, 18.9 requests/second.
- Warm concurrency 4: p50 113.3 ms, p95 144.2 ms, 34.5 requests/second.
- Warm concurrency 8: p50 213.7 ms, p95 313.8 ms, 34.0 requests/second.
- Benchmark errors: 0/150 valid requests; malformed inputs returned HTTP 400 and oversized input returned HTTP 413.
- Container check: the clone-ready CPU image built and returned the same decision
  with a 0.00000018 probability difference from the recorded local runtime.
- Verified container image: `sha256:e80bc2664942a86bd9cabcddad089e91e295e33a612e4c30b4951da35f32e2ae`
  (533,451,298 bytes, Linux ARM64).
- The image contains the verified FP32 model package and starts without a host
  model mount. Compose also mounts `model/` read-only so a different export can
  be selected without rebuilding the application code.
- Default Compose startup reached `/ready`, accepted an audio prediction, and
  passed the Docker health check. A self-contained `docker run` check confirmed
  that threshold `0.6` and model version `container-override-check` were applied
  through environment variables.
- The container runs as non-root UID/GID `10001`.
- Autogenerated `/docs`, `/redoc`, and `/openapi.json` routes are disabled; the
  required service routes remain available.
- Serving preprocessing uses NumPy and ONNX Runtime; PyTorch remains a training/parity dependency and is not included in the image.
- Benchmark conditions: local closed-loop HTTP, one Uvicorn worker, macOS 15.5 ARM CPU, 8 logical CPUs, 8 GiB RAM, 352,044-byte 16 kHz mono WAV.
- Raw parity: `experiments/english_whisper_tiny_v1/baseline_v0/serving_parity.json`.
- Raw benchmark: `experiments/english_whisper_tiny_v1/baseline_v0/api_benchmark_serving_v1.json`.

Serving limitations: the latency run excludes a real network hop and production
monitoring overhead. The browser layout and ready state were verified, while
microphone capture still requires the user to grant browser permission. The
demo polls rolling windows and is not a streaming transport implementation.

### Baseline v0 ENG development evaluation

- Partition: `ENG_evaluation_dev` only; Holdout A and Holdout B remain unscored.
- Decisions: 672 (`474 hold`, `198 end`) across 198 turns and 46 groups.
- Threshold: 0.5.
- Premature responses: 114/474 (24.05%).
- Missed endings: 29/198 (14.65%).
- Detected-ending delay: median 0 ms, p95 400 ms in simulated event time.
- Full-span diagnostic: 76.93% accuracy, 58.57% end precision, 74.24% end recall.
- Report: `reports/ENG_baseline_v0_development_evaluation.md`.
- Raw artifact: `experiments/english_whisper_tiny_v1/baseline_v0/evaluation/ENG_evaluation_dev_report.json`.

### Threshold optimization v1

- Fixed search: 19 thresholds from `0.500` to `0.950` in `0.025` steps.
- Selection data: `ENG_evaluation_dev` only; no model weights or preprocessing changed.
- Selected development candidate: `0.825`, based on minimum total causal event errors.
- Total causal errors: 143 to 115 (-28).
- Premature responses: 114/474 (24.05%) to 59/474 (12.45%).
- Missed endings: 29/198 (14.65%) to 56/198 (28.28%).
- Delay: median unchanged at 0 ms; p95 increased from 400 ms to 500 ms simulated event time.
- Decision: do not promote yet because the missed-ending regression is substantial.
- Holdout A and Holdout B remain unscored.
- Report: `reports/ENG_threshold_optimization_v1.md`.
- Raw artifacts: `experiments/english_whisper_tiny_v1/threshold_optimization_v1/`.

### Unified three-level development evaluation

- Level 1: full-span classification, probability distribution, and calibration.
- Level 2: causal premature responses, missed endings, delay, policy transitions,
  pause slices, and conversation-group bootstrap intervals.
- Level 3: serving parity, cold/warm latency, concurrency, throughput, and policy status.
- Threshold `0.5` remains active; `0.825` remains development-only.
- Report: `reports/ENG_unified_three_level_evaluation_v1.md`.
- Raw artifact: `experiments/english_whisper_tiny_v1/unified_evaluation_v1/report.json`.

### English smoke runs

- Run directory: `experiments/english_whisper_tiny_v1/smoke`
- Fine-tuning gate: `experiments/english_whisper_tiny_v1/smoke_unfrozen`
- Device: Apple MPS; batch size 16 retained after a batch-size 32 capacity check was slower.
- Training/validation records: 64/64
- Best validation loss: 0.6922
- Validation AUROC: 0.5896
- Selected validation threshold: 0.5121
- Best checkpoint: `checkpoints/best.pt`
- Reload parity: passed
- Run manifest: `run_manifest.json`
