# English Baseline v0

This file identifies the selected English model and the minimum information
needed to reproduce or verify it. Detailed analysis remains in `reports/`.

## Selected artifacts

- Model version: `english-whisper-tiny-baseline-v0`
- Training config: `configs/english_whisper_tiny_v1.json`
- PyTorch checkpoint: `model/checkpoints/english_baseline_v0.pt`
- Checkpoint SHA-256: `ef2315dd6bad380ccdc123a8cc73c4638f7503c57cf3eb0258d608d159979b35`
- Serving artifact: `model/model.fp32.onnx`
- ONNX SHA-256: `b79ad05c253996a73969454496583dbce5dcf85bfec9a6b2fdf16e7d9858d9bf`
- Decision threshold: `0.5`

## Data and preprocessing

- Training source: Pipecat revision `f8c95d284595defb6b88a715c356cbc5c31b7ad1`
- English records: 59,280 training and 6,522 validation
- External evaluation source: LiveKit revision `ca9d98a9686b920a2d8c9eb984224ba9be74e4dd`
- Input: latest eight seconds of 16 kHz mono audio
- Short inputs are left-padded with silence; older audio is truncated
- Labels: `hold=0`, `end=1`

## Training recipe

- Backbone: `openai/whisper-tiny` revision `169d4a4341b33bc18d8881c4b69c2e104e1cc0af`
- Selected checkpoint: epoch 1 with the encoder frozen
- Seed: 1337; batch size: 16
- Optimizer: AdamW; head learning rate: `1e-3`; weight decay: `0.01`
- Warm-up: 10%; gradient clipping: `1.0`; dropout: `0.1` during training
- Selection rule: lowest validation loss

```bash
PYTHONPATH=src python3 -m turn_detector.training.train \
  --config configs/english_whisper_tiny_v1.json
```

## Recorded quality

- Pipecat validation: 93.24% accuracy, 91.24% end precision, 95.15% end recall,
  98.00% AUROC, and 8.55% hold false-positive rate
- LiveKit causal development: 24.05% premature responses, 14.65% missed
  endings, and 400 ms p95 simulated detection delay
- LiveKit Holdout A: 81.79% event accuracy, 67.63% end precision, 92.16% end
  recall, 23.81% premature responses, and 7.84% missed endings

## Serving verification

- FP32 ONNX export changed no decisions in the parity test; maximum probability
  difference from PyTorch was `0.0000023`
- The selected CPU configuration achieved a median-of-runs warm p95 of 99.40 ms,
  but one run reached 111.52 ms; sub-100 ms p95 is therefore not guaranteed

## Detailed reports

- `reports/ENG_baseline_v0_development_evaluation.md`
- `reports/ENG_baseline_v0_holdout_a_evaluation.md`
- `reports/ENG_unified_three_level_evaluation_v1.md`
- `reports/ENG_local_api_benchmark.md`
