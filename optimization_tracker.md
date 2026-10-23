# Model optimization tracker

## Current decision

Use the epoch-1 frozen-encoder checkpoint as Baseline v0 and keep the decision
threshold at `0.5`. Complete fine-tuning and threshold calibration remain
optional follow-up experiments rather than blockers for initial serving.

Checkpoint:
`experiments/english_whisper_tiny_v1/full_interrupted_drop_last/checkpoints/best.pt`

## Baseline validation

The checkpoint was evaluated on all 6,522 English validation records with
dropout disabled and gradient tracking off.

| Metric | Result |
|---|---:|
| Accuracy | 93.24% |
| End precision | 91.24% |
| End recall | 95.15% |
| F1 | 93.15% |
| AUROC | 98.00% |
| Hold false-positive rate | 8.55% |

Validation calls `model.eval()`, which disables both dropout layers, and runs
inside `torch.inference_mode()`, which prevents gradient graph construction.
Dropout remains `0.1` during training.

## Leakage and independence checks

| Check | Status | Evidence or limitation |
|---|---|---|
| Record-ID overlap | Clear | Zero IDs overlap between English train and validation. |
| Encoded-audio duplicates | Clear | 65,802/65,802 English audio hashes are unique; zero cross-split groups. |
| Label conflicts among exact duplicates | Clear | Zero groups found. |
| External-source separation | Clear | External decisions use a distinct dataset provenance. |
| Broad source-family overlap | Present | The same 10 provenance categories appear in train and validation. |
| Speaker independence | Unknown | The training source provides no speaker IDs. |
| Conversation independence | Unknown | The training source provides no conversation IDs. |
| Original-recording independence | Unknown | The training source provides no source-recording IDs. |
| Re-encoded waveform duplicates | Pending | Compare hashes of decoded PCM, not only encoded FLAC bytes. |
| Perceptually similar recordings | Pending | Run a scalable audio-fingerprint or embedding-neighbor audit. |

The internal validation split is suitable for baseline iteration, but it cannot
support a claim of speaker- or conversation-independent generalization.

## Optimization backlog

### Controlled preprocessing and unfreezing comparisons

1. Train normalized v2 with the existing freeze-one-epoch schedule to isolate
   waveform normalization from every other setting.
2. Compare it with a normalized gradual-unfreeze candidate that keeps positional
   embeddings frozen and initially unfreezes only the top encoder blocks.
3. Keep the split, seed, threshold `0.5`, total epochs, and causal development
   protocol fixed. Repeat the selected comparison with another seed before
   attributing a small difference to the schedule.
4. Preserve every epoch checkpoint and do not score either holdout during these
   comparisons.

### Next controlled retraining experiment

1. Complete the configured three-epoch run from generic Whisper Tiny
   initialization: freeze the encoder for epoch 1, then fine-tune it for epochs
   2 and 3. This is the first priority because Baseline v0 stopped after the
   frozen-encoder epoch.
2. Keep the Pipecat train/validation split, preprocessing, seed, batch size,
   threshold `0.5`, and unified development protocol fixed for the first
   comparison. This isolates the effect of encoder fine-tuning.
3. If the complete run improves causal development behavior, compare one
   regularization candidate with dropout `0.2` against the existing `0.1`.
   Avoid a broad sweep on the 46 development groups.
4. Select a model using Pipecat validation loss plus the LiveKit development
   guardrails: fewer premature responses without materially increasing missed
   endings or p95 detection delay.
5. Freeze the checkpoint, preprocessing, threshold, and serving artifact before
   scoring Holdout A. Keep Holdout B untouched for a later final candidate.

Running the inference client does not train the model. Data becomes unsuitable
for an unbiased final claim when it contributes gradients, checkpoint or
threshold selection, or repeated human choices. `ENG_evaluation_dev` has
already informed threshold selection and is therefore development data. The
two holdouts remain clean.

### Model and decision policy

- Test dropout values such as `0.0`, `0.1`, and `0.2` during training.
- Improve the model's separation of continuation pauses and true endings; the
  development-only threshold sweep reduced premature responses at `0.825` but
  increased missed endings enough that the candidate was not promoted.
- Fine-tune the Whisper encoder for the remaining epochs.
- Compare multiple seeds before attributing small metric differences to a change.
- Consider source-balanced sampling because English provenance is imbalanced.

### Data and evaluation

- Complete decoded-PCM and perceptual near-duplicate audits.
- Adopt a speaker/conversation-grouped split if identifying metadata becomes available.
- Report validation metrics by provenance and real/synthetic source.
- Keep causal replay fixed across candidates and report classification,
  event-level, and serving-policy results separately.

### Serving

- Rework INT8 quantization; the first static export changed 2/16 parity decisions
  and had a maximum probability difference of 0.487, so it is not serving-eligible.
- Compare FP32 and corrected INT8 artifact quality and latency.
