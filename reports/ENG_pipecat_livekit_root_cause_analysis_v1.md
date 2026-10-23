# ENG Pipecat-to-LiveKit root-cause analysis v1

## Status

- Purpose: explain why the English model performs strongly on Pipecat validation
  but produces substantially more errors under causal LiveKit development replay.
- Scope: Pipecat English train/validation data and `ENG_evaluation_dev` only.
- Holdout A and Holdout B were not scored or inspected for model behavior.
- This is a diagnosis and design note. It does not promote a model or authorize a
  training change.

## Bottom line

The evidence does not support a single explanation such as "the model needs more
records." Pipecat already supplies 59,280 English training clips. The central
problem is that the model was trained and selected on a different decision task
from the one it faces in LiveKit causal replay.

Pipecat provides one complete/incomplete label for each isolated clip. LiveKit
provides a complete natural turn containing several silences, and the model must
make repeated causal decisions at every silence without seeing future speech.
Canonicalization makes field names, labels, audio format, and access paths
consistent; it cannot make these two label-generation processes or audio
distributions equivalent.

Insufficient effective heterogeneity is a contributing risk, not the complete
root cause. The missing variety is specifically variety in natural continuation
pauses, speakers, recording environments, conversational contexts, disfluencies,
and repeated online decision trajectories.

## Observed dataset differences

| Dimension | Pipecat English training | LiveKit English evaluation | Consequence |
|---|---|---|---|
| Learning unit | One independently labeled clip | One complete user turn with multiple silence spans | The training row does not represent the production decision sequence. |
| Label construction | `endpoint_bool` maps once to `hold` or `end` | Every non-final silence is `hold`; the final silence is `end` | LiveKit directly tests continuation pauses inside an ongoing turn. |
| Decisions per item | One decision per clip | Multiple pause events per turn and multiple score times per event | Small false-positive tendencies accumulate into user-visible interruptions. |
| English size | 59,280 train and 6,522 validation clips | 400 turns and 1,250 pause decisions in the public sample | Pipecat is larger, but record count does not supply the missing temporal structure. |
| Label prior | Train: 50.59% hold, 49.41% end | Full sample: 68.0% hold, 32.0% end; development: 70.54% hold | A classifier trained near 50/50 sees many more continuation opportunities in evaluation. |
| Real/synthetic | Train: 39,355 real and 19,925 synthetic according to source metadata | Source describes real human-to-agent turns; local rows do not encode provenance | Synthetic and source-specific cues may transfer imperfectly. |
| Source balance | Ten source categories; top three contain 79.82% of English records | One external source with unknown internal microphone/speaker breakdown | The effective training diversity is smaller than the row count suggests. |
| Conversation identity | No speaker or conversation IDs | 96 inferred conversation groups across the public sample | Training/validation speaker and conversation independence cannot be established. |
| Temporal metadata | No pause spans or word timestamps | Every silence span is timestamped | Pipecat cannot directly supervise score behavior throughout a pause. |
| Turn length | An 83-file audit sample ranged from 2.08 to 16.0 seconds, mean 7.61 seconds | Turns range from 2.2 to 85.1 seconds, mean about 12.6 seconds | The fixed latest-eight-second window may omit useful earlier context in long turns. |
| Context exposed to model | Latest eight seconds of audio only | Latest eight seconds available at each causal timestamp | Input modality is fair, but long-range semantic/dialogue context is unavailable to both. |
| Normalization | Baseline v0 does not normalize waveforms | Same baseline preprocessing is applied | Training/serving parity is preserved, but source-level gain mismatch is not corrected. |

The code-level difference is explicit. Pipecat maps one `endpoint_bool` to one
label in
[`build_canonical_index.py`](../src/turn_detector/data_fetching/build_canonical_index.py#L176-L252).
LiveKit expands every turn into all of its silence decisions and labels only the
last silence as `end` in
[`build_canonical_index.py`](../src/turn_detector/data_fetching/build_canonical_index.py#L340-L405).
Both sources then use the same mono conversion, latest-eight-second truncation,
optional normalization, and left padding in
[`data.py`](../src/turn_detector/data_loader/data.py#L198-L224).

## Why the dataset roles were not reversed

Pipecat remains the primary training source because it supplies 59,280 English
training clips with a nearly balanced hold/end target. LiveKit supplies only 400
original turns. Its 1,250 pause decisions are valuable, but they are correlated
observations generated from those 400 turns rather than 1,250 independent
conversations.

LiveKit is more informative as an external causal evaluation source because one
turn can produce several timestamped decisions: internal pauses are labeled
`hold` and the final pause is labeled `end`. That structure measures premature
responses, missed endings, and detection delay. Pipecat has no pause timestamps
or within-conversation trajectories, so reversing the roles would remove this
independent production-like measurement without providing an equivalent causal
evaluation from Pipecat.

Inference itself is not a dataset split: the deployed model receives previously
unseen microphone audio. Pipecat and LiveKit are used to train or evaluate that
behavior. A future domain-adaptation experiment may mix a separate, train-only
set of conversation-grouped causal examples with Pipecat, while retaining
untouched external groups. The current LiveKit development and holdout groups
should not be converted into training data after their results have influenced
model decisions.

## Evidence from model behavior

Baseline v0 was evaluated on 198 development turns containing 474 continuation
pauses and 198 true endings:

- Premature responses: 114/474, or 24.05%.
- Missed endings: 29/198, or 14.65%.
- Premature responses at silence onset: 62/114.
- Premature responses by 100 ms of silence: 89/114.
- Continuation events whose maximum `p_end` reached at least 0.9: 41/474.
- True endings whose maximum `p_end` never exceeded 0.1: 7/198.
- Causal replay made 5,810 model calls over 672 pause events. A hold event was
  scored 5.57 times on average; every final 1.5-second span was scored 16 times.

These are not merely borderline errors around threshold 0.5. Some incorrect
decisions are highly confident in both directions. Raising the threshold can
move the error trade-off, but it cannot teach the model the missing distinction.

The premature-rate pattern also does not behave like a simple fixed-silence
problem. Baseline premature rates increase from 20.00% below 500 ms to 30.97%
between 500 ms and 1 second and 36.59% at one second or longer. Long natural
thinking pauses remain difficult even though more silence is available.

Fully fine-tuning the encoder through epoch 3 improved in-domain Pipecat
validation, but on the identical LiveKit development replay it changed total
causal errors from 143 to 143 and increased premature responses from 114 to 125.
This rules out "the encoder was frozen" as the leading explanation for the
external failure.

Evidence artifacts:

- [`ENG_baseline_v0_development_evaluation.md`](ENG_baseline_v0_development_evaluation.md)
- [`ENG_epoch3_causal_development_comparison_v1.md`](ENG_epoch3_causal_development_comparison_v1.md)
- [`ENG_epoch3_validation_comparison_v1.md`](ENG_epoch3_validation_comparison_v1.md)
- [`pipecat_full_training_data_audit.md`](pipecat_full_training_data_audit.md)
- [`livekit_ENG_public_400_audit.md`](livekit_ENG_public_400_audit.md)

## Ranked causes

### 1. Training-target and production-decision mismatch — high confidence

The model learns isolated clip classification, while production requires a
sequence of causal pause decisions. Training does not explicitly penalize an
early score spike inside a continuing turn. Production latches the first spike
as a response, so later score recovery is irrelevant.

This is the best-supported root cause. It is visible in our data construction,
evaluation behavior, and published work. LiveTurn reports that moving from
isolated segments to whole-call training improved pause handling, and explicit
mid-turn-pause weighting improved it substantially further. Google turn-taking
work likewise creates separate supervision for hesitation pauses and genuine
endings.

### 2. Repeated-opportunity amplification — high confidence

A Pipecat validation clip creates one opportunity for a false end. A real turn
can contain several pauses, and each pause can be evaluated repeatedly. Even a
moderate per-decision false-positive tendency therefore becomes a much larger
per-turn interruption probability. Ordinary classification accuracy does not
measure this accumulation.

### 3. Source and class-prior shift — medium-high confidence

Pipecat English is almost label-balanced. LiveKit development is 70.54% hold.
The audio also comes from a completely separate collection. The 41 highly
confident premature events and seven highly confident missed endings indicate
representation/domain failures, not only a poorly chosen threshold.

The exact contribution from microphones, codecs, noise, accents, or gain cannot
currently be isolated because LiveKit does not expose those fields and Pipecat
does not provide speaker or conversation IDs.

### 4. Limited effective heterogeneity — medium confidence

The training set is large but concentrated: `liva_1` contains 46.99% of English
records, `chirp3_1` 19.93%, and `midcentury_1` 12.90%. Approximately 33.61% of
English training rows are synthetic. Ten source labels are better than one, but
row count is not the same as independent speaker, microphone, environment, and
conversation diversity.

This likely contributes to external generalization error, but the current
metadata is insufficient to calculate unique speakers, conversations, accents,
or recording devices. It should not be presented as a proven sole cause.

### 5. Missing temporal and conversational context — medium confidence

The model receives at most the latest eight seconds and no previous dialogue
messages or transcript. This keeps evaluation fair and avoids ASR leakage, but
some long lists, self-corrections, enumerations, or responses dependent on the
agent's previous question may require earlier context. LiveKit turns reach 85.1
seconds, so the current window necessarily discards most of some turns.

### 6. Waveform gain/preprocessing mismatch — plausible but unproven

Official Smart Turn currently enables zero-mean/unit-variance waveform
normalization, while baseline v0 does not. This supplies implementation
precedent, not an ablation demonstrating that normalization reduces premature
responses. Existing error inspection did not show a monotonic relationship
between amplitude and mistakes. Normalization should therefore remain a
conditional experiment rather than the stated root cause.

## What "more heterogeneous data" should mean

Simply appending more isolated clips from the same generators may increase the
row count without fixing the observed problem. Useful heterogeneity must cover
the decisions the service will actually encounter:

- Natural complete turns with every internal silence labeled.
- Thinking pauses, fillers, restarts, repetitions, self-corrections, lists,
  numbers, addresses, and unfinished clauses.
- Different speakers, accents, speaking rates, pitch ranges, and pause habits.
- Microphones, rooms, telephony codecs, compression, gain, clipping, reverb,
  background speech, and environmental noise.
- Short and long turns, with more than one continuation pause in the same turn.
- Conversation-level and speaker-level group identifiers so splits measure
  genuine generalization.
- Multiple causal training timestamps within a pause, including hard negative
  continuation frames that currently receive high `p_end` scores.

## Solution direction

### Diagnose before changing weights

1. On development data only, categorize the 114 premature responses and 29
   missed endings by filler, incomplete syntax, list/number sequence, pause
   duration, turn position, preceding prosody, noise, clipping, and turn length.
2. Run controlled gain and noise interventions. A large prediction change under
   signal scaling would justify matched train-and-serve normalization or gain
   augmentation.
3. Evaluate the existing score sequence as a false-cutoff-versus-latency curve,
   including a VAD-only baseline. This measures whether the present checkpoint
   already contains useful ranking information even if threshold 0.5 is not the
   best operating policy.

### Data and objective before a larger backbone

1. Obtain a separate train-only conversational corpus; do not train on
   `ENG_evaluation_dev`.
2. Generate causal prefix examples at every pause and several timestamps within
   each pause.
3. Keep conversation and speaker groups disjoint across training, validation,
   and external evaluation.
4. Mix causal conversational data with Pipecat rather than discarding Pipecat.
   Pipecat supplies broad endpoint semantics; causal data supplies realistic
   pause trajectories.
5. Balance by source and pause type, and emphasize high-confidence continuation
   errors as hard negatives.
6. Optimize an event-level objective that penalizes any premature first crossing
   and also accounts for missed endings and response delay.
7. Add random gain, noise, reverb, and codec augmentation while retaining
   natural prosodic contours.

### Architecture only after the data comparison

Start with the frozen Whisper encoder so that a data/objective comparison is
isolated and inexpensive. If pause-aware training still fails, compare explicit
prosodic features, a silence-duration feature, a longer causal context window,
or dialogue/streaming-ASR context. Full encoder unfreezing is not currently
supported by our causal results.

## Research support

- [LiveKit EoT benchmark](https://github.com/livekit/eot-bench): evaluates every
  real pause and compares false-cutoff/latency policies rather than ranking by
  accuracy alone.
- [Turn-Taking Prediction for Natural Conversational Speech](https://arxiv.org/abs/2208.13321):
  separately labels thinking pauses, fillers, and genuine endings.
- [Voice Activity Projection](https://github.com/ErikEkstedt/VoiceActivityProjection):
  trains continuously on dialogue dynamics and future voice activity.
- [How Much Does Prosody Help Turn-taking?](https://aclanthology.org/2022.sigdial-1.51/):
  demonstrates that pitch, intensity, and duration cues affect turn decisions.
- [Multilingual Turn-taking Prediction Using VAP](https://aclanthology.org/2024.lrec-main.1036/):
  reports a slight accuracy decrease when pretrained audio encoders were
  unfrozen.
- [Streaming Endpointer with Label-Delayed Training](https://arxiv.org/abs/2506.07081):
  reduces cutoff error through task-specific temporal supervision.
- [LiveTurn preprint](https://openreview.net/pdf?id=JIaOGuEMET): active work whose
  ablations favor whole-call training, explicit mid-turn negatives, and acoustic
  features over isolated segment training.

## Decision to preserve

Do not describe the problem as merely "not enough data." The most defensible
statement is:

> Baseline v0 has adequate in-domain clip-classification capacity, but its
> training distribution and objective underrepresent repeated causal decisions
> on natural continuation pauses. Source concentration and acoustic diversity
> likely add domain shift. Future work should prioritize pause-aware,
> conversation-grouped training data and event-level evaluation before increasing
> model capacity.
