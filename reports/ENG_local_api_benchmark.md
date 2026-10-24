# English Local API Benchmark and Production Plan

## Local benchmark

The benchmark used English Baseline v0, FP32 ONNX Runtime on CPU, threshold
`0.5`, one API worker, and four container CPUs. Streaming tests sent 350 ms
updates from a 16 kHz mono recording. Each selected configuration was measured
in three fresh-container runs with 96 warm requests per run.

| CPU configuration | Client p50 | Client p95 | Inference p50 | Inference p95 |
|---|---:|---:|---:|---:|
| Automatic threads | 93.05 ms | 126.07 ms | 82.86 ms | 116.33 ms |
| Sequential, 2 intra-op / 1 inter-op | 60.61 ms | 99.40 ms | 56.13 ms | 91.74 ms |

The 2/1 configuration reduced client p95 by 21.2% and was selected. Its three
run-level p95 values were 99.40, 111.52, and 97.28 ms, so the median meets the
100 ms objective but performance is not consistently below 100 ms. Prediction
parity passed with no decision changes or request errors.

A separate local full-file load test measured capacity under concurrency:

| Concurrency | Requests | Client p50 | Client p95 | Throughput | Errors |
|---:|---:|---:|---:|---:|---:|
| 1 | 50 | 42.1 ms | 96.3 ms | 18.9 requests/s | 0 |
| 4 | 50 | 113.3 ms | 144.2 ms | 34.5 requests/s | 0 |
| 8 | 50 | 213.7 ms | 313.8 ms | 34.0 requests/s | 0 |

Throughput saturated near 34 requests/s while latency increased with
concurrency. This single CPU container is suitable for the demo, but sustained
production load would require admission control and multiple replicas. Monthly
call count alone is not a capacity measure because each call produces many
streaming prediction requests; peak active streams and requests per second are
the important inputs.

## Production monitoring

Every metric should be tagged with model version, artifact hash, threshold, and
service version.

| Area | What to measure |
|---|---|
| Reliability | Readiness, HTTP failures, restarts, and request rejection |
| Performance | API, preprocessing, and inference p50/p95; throughput, concurrency, and queue time |
| Model behavior | `p_end` distribution, hold/end rate, and scores near the threshold |
| Input drift | Audio duration, loudness, clipping, noise level, and decode failures |
| Quality | Premature responses, missed endings, detection delay, precision, and recall on labeled samples |

Useful online warning signals include the user speaking after the agent starts,
which may indicate a premature response, and long silence or a timeout without
an end decision, which may indicate a missed ending. These signals are proxies;
quality metrics should be confirmed on reviewed, causally labeled examples.

Initial alerts should cover readiness failures, server errors above 1%, warm
API p95 above 100 ms for a sustained window, abrupt output-distribution drift,
and a material regression in labeled premature or missed-ending rates.

## Improvement loop at production scale

At roughly one million calls per month, I would not label traffic randomly.
I would sample likely failures, uncertain scores near the threshold, different
noise conditions, devices, accents, speakers, and conversation lengths. A
reviewed subset would receive causal `hold`/`end` labels and conversation IDs.

The improvement cycle would be:

1. Identify weak slices from monitoring and user-interruption signals.
2. Review and label representative production conversations.
3. Canonicalize, deduplicate, and split by conversation before training.
4. Train a candidate and compare it with Baseline v0 using the fixed causal
   evaluation, untouched holdout, parity, and API benchmark checks.
5. Release to a small canary, compare quality and latency, then promote or roll
   back using the recorded model version.

The model is not permanently limited by Pipecat and LiveKit. Pipecat made a
practical first baseline, but its isolated clips do not cover continuous
conversation, varied noise, or production speaker behavior well. Reviewed
production data can close those gaps. The highest-value additions are natural
continuation pauses, clear endings, noisy and overlapping speech, and difficult
cases where the current model interrupts or waits too long.

Model improvements can then include training-recipe and normalization tuning,
classification-head comparisons, threshold calibration, and a smaller or
cached encoder for more stable CPU latency. Each change should be judged on
premature responses, missed endings, detection delay, and p95 API latency rather
than validation accuracy alone.
