# ENG epoch-3 validation comparison v1

## Decision

This is a fixed-threshold internal validation comparison. Both checkpoints were
scored at `0.5`; the automatically calibrated threshold was ignored. Epoch 3
is a development candidate and is not promoted by this report.

## Frozen protocol

- Dataset: Pipecat English `validation` split at revision `f8c95d284595defb6b88a715c356cbc5c31b7ad1`.
- Records: 6522 (`3369 hold`, `3153 end`).
- Decision-ID hash: `b71ffef72cc49cdef78132ceda5fbde4e608a400ddbecc059291b8ce2416c241`.
- Input: latest 8 seconds of 16 kHz mono audio, left-zero-padded when shorter.
- Evaluation: `model.eval()` and `torch.inference_mode()`; dropout and gradients disabled.
- Threshold: `0.5` for both checkpoints.
- No LiveKit development or holdout examples were accessed.

## Checkpoints

| Role | Epoch | SHA-256 | Stored threshold | Evaluation threshold |
|---|---:|---|---:|---:|
| Baseline v0 | 1 | `ef2315dd6bad380ccdc123a8cc73c4638f7503c57cf3eb0258d608d159979b35` | 0.5000 | 0.5 |
| Fully fine-tuned candidate | 3 | `4a16afdd57b58611917cdca82f7ff2978f60e9b0b265f16165e4388991472e4b` | 0.5000 | 0.5 |

The candidate is `latest.pt` from epoch 3. It is deliberately not described as
the lowest-loss checkpoint: epoch 1 still has the lowest validation loss.

## Results at threshold 0.5

| Metric | Baseline epoch 1 | Candidate epoch 3 | Candidate change |
|---|---:|---:|---:|
| BCE loss | 0.17419 | 0.21384 | +0.03965 |
| Accuracy | 93.24% | 94.99% | +1.75 pp |
| End precision | 91.24% | 93.61% | +2.37 pp |
| End recall | 95.15% | 96.19% | +1.05 pp |
| F1 | 93.15% | 94.89% | +1.73 pp |
| AUROC | 98.00% | 98.79% | +0.79 pp |
| Hold false-positive rate | 8.55% | 6.14% | -2.40 pp |
| Brier score | 0.05071 | 0.04402 | -0.00668 |
| Calibration error, 10 bins | 1.24% | 3.87% | +2.63 pp |

| Actual / predicted | Baseline epoch 1 | Candidate epoch 3 |
|---|---:|---:|
| Hold predicted hold | 3081 | 3162 |
| Hold predicted end | 288 | 207 |
| End predicted hold | 153 | 120 |
| End predicted end | 3000 | 3033 |

The candidate corrected 104 baseline
hold errors and 70 baseline end
errors, while introducing 23 new
hold errors and 37 new end errors.

## Interpretation

Epoch 3 improves every threshold-based classification metric at `0.5`, including
accuracy, precision, recall, F1, AUROC, and the hold false-positive rate. Its
probability-quality results are mixed: Brier score improves, while BCE loss and
10-bin calibration error worsen. The candidate reduces average squared error
but has more calibration mismatch and a small number of more confidently wrong
predictions. `p_end` should continue to be treated as a model score rather than
a literal calibrated probability.

This validation split was observed during model development and cannot establish
external turn-taking quality. The candidate must next be compared with Baseline
v0 on the fixed causal `ENG_evaluation_dev` protocol at threshold `0.5`, measuring
premature responses, missed endings, and detection delay. Neither holdout should
be used yet.

## Reproduce

```bash
PYTHONPATH=src .venv/bin/python -m turn_detector.evaluation.compare_validation_checkpoints --device mps
```

Raw artifacts:
`experiments/english_whisper_tiny_v1/validation_comparison_v1/`.
