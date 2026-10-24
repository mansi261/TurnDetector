# Local API Benchmark and Production Plan

## English benchmark

The benchmark used English Baseline v0, FP32 ONNX Runtime on CPU, threshold
`0.5`, one API worker, and four container CPUs.

| CPU configuration | Client p50 | Client p95 | Inference p50 | Inference p95 |
|---|---:|---:|---:|---:|
| Automatic threads | 93.05 ms | 126.07 ms | 82.86 ms | 116.33 ms |
| Sequential, 2 intra-op / 1 inter-op | 60.61 ms | 99.40 ms | 56.13 ms | 91.74 ms |

The selected 2/1 CPU threading configuration improved streaming client p50 by
34.9% and p95 by 21.2%.

| Concurrency | Requests | Client p50 | Client p95 | Throughput | Errors |
|---:|---:|---:|---:|---:|---:|
| 1 | 50 | 42.1 ms | 96.3 ms | 18.9 requests/s | 0 |
| 4 | 50 | 113.3 ms | 144.2 ms | 34.5 requests/s | 0 |
| 8 | 50 | 213.7 ms | 313.8 ms | 34.0 requests/s | 0 |

Throughput saturated near 34 requests/s. Capacity planning should use peak
active streams and prediction requests per second because every call produces
multiple streaming requests.

## Multilingual benchmark

| Request path | Client p50 | Client p95 |
|---|---:|---:|
| Streaming | 63.76 ms | 110.43 ms |
| Warm file inference | 119.58 ms | 143.54 ms |

The multilingual candidate did not meet the 100 ms p95 objective. The English
model remains the default serving artifact.

## Production monitoring

Monitor:

- API, preprocessing, and inference p50/p95 latency.
- Throughput, concurrency, queue time, and error rate.
- CPU and memory utilization, pod restarts, and readiness failures.
- `p_end` distribution, hold/end rate, and active model threshold.
- Premature responses, missed endings, and detection delay on labeled samples.
- Audio drift across noise, duration, loudness, clipping, and decode failures.

Tag metrics with the model version, artifact hash, threshold, and service
version. Alert on sustained latency or error regressions, resource saturation,
unexpected restarts, and material quality changes.

## Improvement approach

At production scale, sample likely failures, uncertain scores, noisy audio,
accents, devices, and conversation lengths for causal labeling. Deduplicate and
split by conversation, retrain on realistic continuation pauses and endings,
then require causal evaluation, an untouched holdout, an API benchmark, and a
small canary release before promotion.

For multilingual improvement, add balanced real conversational data per
language, preserve conversation-level splits, report per-language causal
errors, and tune the objective for continuation pauses rather than validation
accuracy alone.
