# Multilingual Turn Detector Audit

## Verdict

The 14-language pipeline is reproducible, but the candidate remains
experimental. It reduces premature responses yet increases missed endings
beyond the frozen guardrail.

## Data

Languages: Arabic, German, English, Spanish, French, Hindi, Indonesian, Italian,
Japanese, Korean, Dutch, Portuguese, Turkish, and Chinese.

| Partition | Decisions |
|---|---:|
| Pipecat training | 177,485 |
| Pipecat validation | 19,837 |
| LiveKit development | 11,248 (`7,157 hold`, `4,091 end`) |
| LiveKit holdout | 3,887 (`2,524 hold`, `1,363 end`) |

All 202,776 source recordings were readable, totaling 449.48 hours. Required
fields were complete, exact encoded-audio duplicate groups were zero, and
LiveKit partitions had no conversation-group overlap. Speaker IDs are absent,
LiveKit provenance is unknown, and 12 of 14 Pipecat languages are entirely
synthetic in source metadata. The data is suitable with these constraints.

## Model and validation

The candidate starts from English Baseline v0, resets training state, and trains
the classification head for one frozen-encoder epoch at threshold `0.5`.
Validation reached 92.58% micro accuracy, 90.51% end precision, 94.71% end
recall, and 97.61% AUROC; macro-language accuracy was 92.04%.

## Causal evaluation

| Partition/model | Accuracy | End precision | End recall | Premature | Missed | Delay p95 |
|---|---:|---:|---:|---:|---:|---:|
| Development: English v0 zero-shot | 69.95% | 55.14% | 93.25% | 43.37% | 6.75% | 300 ms |
| Development: multilingual v1 | 72.71% | 57.97% | 90.81% | 37.64% | 9.19% | 400 ms |
| Holdout: multilingual v1 | 73.01% | 57.28% | 90.68% | 36.53% | 9.32% | 400 ms |

On holdout, multilingual v1 reduced total errors from 1,178 to 1,049 and
premature responses from 1,090 to 922, but missed endings increased from 88 to
127. Eleven languages improved; English and Japanese regressed, and Chinese was
unchanged. This failed the missed-ending guardrail, so the model was not
promoted.

Audit evidence can be regenerated with
`PYTHONPATH=src python3 -m turn_detector.data_fetching.audit_multilingual`.
