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
- `ENG_evaluation_dev` is reusable development data. Holdout A has been used
  once for frozen Baseline v0 reporting; Holdout B remains untouched.
- The separate multilingual v1 cycle covers 14 languages with its own index,
  development partition, one final holdout, checkpoint, and serving package.
  Its holdout has been scored once; it cannot be used to tune that candidate.

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
- [x] Development-only threshold sweep completed without using either holdout.
- [x] Unified three-level Baseline v0 and threshold `0.825` comparison recorded.
- [x] Frozen Baseline v0 evaluated once on ENG Holdout A at threshold `0.5`.
- [x] FP32 ONNX export verified against PyTorch.
- [ ] INT8 export corrected and revalidated.
- [x] FastAPI service verified with the FP32 artifact.
- [x] Live API metrics endpoint and monitoring dashboard implemented.
- [x] Standalone inference client records input/artifact identity, causal-prefix predictions, and client/server timing.
- [x] CPU-only container build and mounted-artifact inference verified.
- [x] Cold/warm/concurrent API benchmark recorded.
- [x] Serving preprocessing, artifact identity, causal parity, and holdout isolation reviewed.
- [x] Production monitoring, rollback, and improvement plan documented.
- [x] Architecture decisions documented from data through serving.
- [x] Dashboard simplified to display p50 and p95 latency only.
- [x] Fourteen-language source snapshot fetched, pinned, and audited.
- [x] Isolated multilingual canonical index, splits, and loaders verified.
- [x] Multilingual training smoke and exact-resume checks passed.
- [x] One complete frozen-encoder multilingual epoch trained and validated.
- [x] Multilingual causal development comparison completed at threshold `0.5`.
- [x] Multilingual FP32 ONNX, API parity, Docker, and latency verified.
- [x] Multilingual candidate and evaluation protocol frozen by content hash.
- [x] Multilingual holdout scored once with no post-holdout tuning.

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

### Multilingual v1 data identity

- Languages: `ara`, `deu`, `eng`, `fra`, `hin`, `ind`, `ita`, `jpn`, `kor`,
  `nld`, `por`, `spa`, `tur`, and `zho`.
- Pipecat revision: `f8c95d284595defb6b88a715c356cbc5c31b7ad1`.
- LiveKit revision: `ca9d98a9686b920a2d8c9eb984224ba9be74e4dd`.
- Canonical schema version: 4; index SHA-256:
  `e1aa5002157364f378dd23f7e086e70a27a8844aed8b0861900763bb77d018ba`.
- Training: 177,485 decisions (`89,430 hold`, `88,055 end`).
- Validation: 19,837 decisions (`10,167 hold`, `9,670 end`).
- Development evaluation: 11,248 decisions (`7,157 hold`, `4,091 end`).
- Final holdout: 3,887 decisions (`2,524 hold`, `1,363 end`).
- All 14 languages and both labels occur in every partition.
- Full audit: 202,776 readable source recordings, 449.48 audio hours, zero
  exact encoded-audio duplicate groups, and zero known group/source overlap
  across external partitions.
- Limitations: speaker IDs are unavailable, LiveKit provenance is unknown, and
  12 of 14 Pipecat languages are entirely synthetic in source metadata.

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
- At the time of this comparison, both holdouts were unscored.
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
demo uses process-local streaming state and is not a distributed streaming
transport implementation.

### Baseline v0 ENG development evaluation

- Partition: `ENG_evaluation_dev` only; neither holdout was used in this run.
- Decisions: 672 (`474 hold`, `198 end`) across 198 turns and 46 groups.
- Threshold: 0.5.
- Premature responses: 114/474 (24.05%).
- Missed endings: 29/198 (14.65%).
- Detected-ending delay: median 0 ms, p95 400 ms in simulated event time.
- Full-span diagnostic: 76.93% accuracy, 58.57% end precision, 74.24% end recall.
- Report: `reports/ENG_baseline_v0_development_evaluation.md`.
- Raw artifact: `experiments/english_whisper_tiny_v1/baseline_v0/evaluation/ENG_evaluation_dev_report.json`.

### Baseline v0 ENG Holdout A evaluation

- Completed once with the frozen Baseline v0 checkpoint and threshold `0.5`.
- Decisions: 291 (`189 hold`, `102 end`) across 102 turns and 25 groups.
- Causal event accuracy: 81.79%; end precision: 67.63%; end recall: 92.16%.
- Premature responses: 45/189 (23.81%).
- Missed endings: 8/102 (7.84%).
- Detected-ending delay: median 0 ms, p95 300 ms in simulated event time.
- Holdout A will not be used for tuning; Holdout B remains untouched.
- Report: `reports/ENG_baseline_v0_holdout_a_evaluation.md`.
- Raw artifacts: `experiments/english_whisper_tiny_v1/baseline_v0/evaluation/ENG_holdout_a_report*`.

### Threshold optimization v1

- Fixed search: 19 thresholds from `0.500` to `0.950` in `0.025` steps.
- Selection data: `ENG_evaluation_dev` only; no model weights or preprocessing changed.
- Selected development candidate: `0.825`, based on minimum total causal event errors.
- Total causal errors: 143 to 115 (-28).
- Premature responses: 114/474 (24.05%) to 59/474 (12.45%).
- Missed endings: 29/198 (14.65%) to 56/198 (28.28%).
- Delay: median unchanged at 0 ms; p95 increased from 400 ms to 500 ms simulated event time.
- Decision: do not promote yet because the missed-ending regression is substantial.
- Neither holdout was used for this threshold search.
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

### Docker latency recheck after code cleanup

- Serving and training preprocessing parity passed on two development records and
  a short-silence boundary input: zero decision mismatches, direct ONNX/API
  probability difference `0.0`, and maximum PyTorch/ONNX difference `0.0000024`.
- The rebuilt Linux ARM64 container loaded the FP32 model once, reported the
  expected 16 kHz mono/latest-8-second preprocessing, and passed its health check.
- The runtime image excludes the benchmark-only `httpx` and `psutil` packages.
- First inference after the container restart, using an 11-second/352,044-byte
  WAV, took `475.4 ms` client-observed and `431.4 ms` inside preprocessing plus
  model inference. Filesystem and Docker caches may already have been warm.
- Warm Docker requests using the same input at concurrency 1 measured `106.1 ms`
  p50 and `138.3 ms` p95. At concurrency 4 they measured `321.2/679.6 ms`, and at
  concurrency 8 they measured `561.2/866.9 ms` p50/p95.
- A second warm run with an exact 8-second/256,044-byte WAV measured `108.7 ms`
  p50 and `153.4 ms` p95 at concurrency 1, `312.4/480.5 ms` at concurrency 4,
  and `466.0/803.4 ms` at concurrency 8.
- Both runs completed all 150 valid requests with zero errors. Empty, malformed,
  wrong-rate, and oversized inputs returned the expected `400` or `413` status.
- Conclusion: the Docker path is functional, but it does not currently meet the
  `<100 ms` end-to-end target consistently and degrades under concurrent load.
- Raw results:
  `experiments/english_whisper_tiny_v1/baseline_v0/api_benchmark_docker_v1.json`
  and
  `experiments/english_whisper_tiny_v1/baseline_v0/api_benchmark_docker_8s_v1.json`.

### Incremental preprocessing v1

- The file-oriented `/predict` contract remains unchanged.
- The microphone demo creates an isolated stream session and sends only new 16
  kHz mono PCM16 samples. Sessions expire after five minutes and are capped at
  32 per process.
- The server retains the latest eight-second waveform and raw log-mel window. A
  typical 350 ms update recomputed 38/800 feature frames and reused 762/800.
- Incremental features matched full NumPy Whisper preprocessing exactly in the
  deterministic check. PyTorch/direct ONNX/stateless API/streaming API parity
  passed with zero decision changes and maximum streaming probability difference
  `0.00000083`.
- Docker chronological replay used the same 11-second PCM16 audio twice in
  alternating order. The final reproducible run measured `8.30/31.56 ms`
  stateless preprocessing and `4.63/12.15 ms` incremental preprocessing at
  p50/p95.
- In that run, end-to-end latency improved from `123.13/176.23 ms` to
  `78.55/118.40 ms` p50/p95. Median latency is below 100 ms, but p95 is not;
  inference remains the dominant stage.
- Total uploaded audio fell from 10,790,016 bytes to 704,000 bytes across 64
  requests. Paired predictions were identical with zero decision changes.
- Raw result:
  `experiments/english_whisper_tiny_v1/baseline_v0/streaming_preprocessing_benchmark_v1.json`.

### ONNX Runtime CPU tuning v1

- The baseline already used sequential execution and full graph optimization;
  only CPU thread allocation required tuning.
- Automatic, 1/1, 2/1, 3/1, 4/1, and parallel 4/2 thread configurations were
  screened on the same four visible Docker CPUs.
- The automatic baseline and the two strongest candidates were each rerun three
  times with a fresh container and 96 warm streaming updates per run.
- Sequential 2/1 was selected. The median-of-runs client p50/p95/p99 changed
  from `93.05/126.07/162.37 ms` to `60.61/99.40/116.06 ms`, improvements of
  `34.9%`, `21.2%`, and `28.5%` respectively.
- One of the three selected-candidate runs measured `111.52 ms` p95, so latency
  is improved but not reliably below 100 ms in every run.
- Parity passed with zero decision changes and maximum streaming probability
  difference `0.00000048`.
- The selected runtime is configurable through environment variables and is
  reported by `/ready`.
- Report: `reports/ENG_onnx_runtime_cpu_tuning_v1.md`.
- Raw results:
  `experiments/english_whisper_tiny_v1/baseline_v0/onnx_tuning/`.

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

### Multilingual v1 — completed, experimental

- Config: `configs/multilingual_whisper_tiny_v1.json`.
- Initialization: complete English Baseline v0 state at SHA-256
  `ef2315dd6bad380ccdc123a8cc73c4638f7503c57cf3eb0258d608d159979b35`;
  optimizer, scheduler, gradients, and run state were reset.
- Training: one frozen-encoder epoch, seed 1337, batch size 16, AdamW head
  learning rate `1e-4`, weight decay 0.01, 10% warm-up/cosine decay, gradient
  clipping 1.0, and dropout 0.1 during training only.
- Sampling: deterministic full-coverage grouped shuffle. Every one of the
  177,485 training records was processed exactly once in 11,093 updates.
- Integrity smoke: one hold and one end record per language in both training
  and validation; finite update, evaluation mode, disabled validation gradients,
  checkpoint reload parity, and model/optimizer/scheduler/RNG/resume restoration
  all passed.
- Full run: `experiments/multilingual_whisper_tiny_v1/full_v1`; MPS runtime
  3,121.85 seconds; mean training loss 0.2071.
- The run manifest records source SHA-256
  `416a55c80a9ce8d43c507edb947fb72d9693362750600fdba9515ddd73307236`,
  repository head, dirty-diff identity, dependencies, hardware, invocation, and
  input hashes. The current source hash differs because evaluation, serving,
  caching, and reproducibility packaging were completed after training; the
  training configuration and selected checkpoint are preserved by content hash.
- Selected checkpoint SHA-256:
  `c813264ec85e0013a7e96e5391536e2d6d6a7df2322c8020751fd3ae684367d4`.
- Reproducible checkpoint copies: `model/checkpoints/english_baseline_v0.pt`
  and `model/checkpoints/multilingual_v1.pt`. They are excluded from the Docker
  runtime image; the trainer accepts the packaged initialization path explicitly.
- Validation over 19,837 records: 92.58% micro accuracy, 90.51% end precision,
  94.71% end recall, 92.56% F1, 97.61% AUROC, 9.44% hold false-positive rate,
  0.1904 loss, and 0.0081 ten-bin calibration error. Macro-language accuracy
  was 92.04% and macro-language loss was 0.2033.
- Development causal replay compared both checkpoints on the same 11,248 events
  at threshold 0.5. Candidate micro errors fell from 3,380 to 3,070: premature
  responses fell from 3,104/7,157 to 2,694/7,157, while missed endings rose from
  276/4,091 to 376/4,091. Macro error improved by 2.75 points, but the 2.41-point
  macro missed-ending increase failed the frozen 2-point guardrail.
- FP32 ONNX SHA-256:
  `c6a32108a4428132bc2eb4033451b85bc874920ae43bbabfa26d842baa352703`.
  Export parity used 28 validation records and had zero decision mismatches with
  a maximum probability difference of 0.00000155. Raw-audio PyTorch/ONNX/API
  parity across all 14 languages also had zero decision mismatches; direct
  ONNX/API difference was 0.0.
- Verified Docker image:
  `sha256:7a04e3fed28d00b2aa38a3feeb4083ea701a2ad688e1f949ac1aeef06724a9c6`
  (595,432,558 bytes, Linux ARM64, non-root). Readiness reported the exact model
  hash, all 14 languages, CPU provider, 0.5 threshold, and preprocessing. The
  packaged PyTorch checkpoints are absent from the runtime image.
- Docker streaming at concurrency one measured 63.76 ms p50 and 110.43 ms p95
  client latency; warm file requests measured 119.58/143.54 ms p50/p95. The
  100 ms p95 objective was not met. All 150 valid load requests succeeded and
  malformed, empty, wrong-rate, and oversized inputs returned 400/413.
- Frozen candidate manifest:
  `experiments/multilingual_whisper_tiny_v1/full_v1/frozen_candidate.json`.
  It locks checkpoint, config, scoring-code, threshold, cadence, and dispatch
  hashes before holdout access.
- The one-time holdout run scored 3,887 events. Candidate micro errors fell from
  1,178 to 1,049; premature responses fell from 1,090/2,524 to 922/2,524;
  missed endings rose from 88/1,363 to 127/1,363. Accuracy rose from 69.69% to
  73.01% and precision from 53.91% to 57.28%, while recall fell from 93.54% to
  90.68% and p95 event-time delay rose from 300 to 400 ms. Eleven languages
  improved, English and Japanese regressed, and Chinese was unchanged.
- Holdout report SHA-256:
  `68c9428a1f0b4614aa8718f6c4256074869476aa625eb9456bf6625b22ef544d`.
  Raw report:
  `experiments/multilingual_whisper_tiny_v1/full_v1/evaluation/holdout_report.json`.
- Decision: keep multilingual v1 experimental. It improves aggregate event
  error and interruption behavior, but fails the missed-ending and p95 latency
  guardrails. Holdout results will not be used to modify this candidate. The
  English Baseline v0 and default model package remain unchanged.
