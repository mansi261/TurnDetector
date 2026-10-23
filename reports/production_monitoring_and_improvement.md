# Production monitoring and improvement plan

## Scope

This plan covers the English turn detector, its inference API, and the feedback
loop used to improve later model versions. Operational telemetry must identify
the model version, artifact SHA-256, threshold, and service version so that a
change in behavior can be traced to an exact deployment.

The current API exposes bounded process-local metrics through `GET /metrics`
and the API dashboard tab. It retains the latest 5,000 request summaries and
resets on restart. Persistent metric storage, alert delivery, and cross-instance
aggregation remain deployment requirements.

## Service and model metrics

The service dashboard should report these metrics by model version and deployment:

| Signal | Measurement |
|---|---|
| Availability | `/ready` success and HTTP response codes |
| Latency | Preprocessing, inference, and server-total p50/p95/p99 |
| End-to-end latency | Client-observed request p50/p95/p99, including network time |
| Load | Requests per second, concurrent requests, and queue time |
| Errors | Overall and per-status error rate, separated into invalid input and server failure |
| Input health | Audio byte size, duration, sample rate, and decode failures |
| Output behavior | Aggregated `p_end` histogram, hold/end rate, and active threshold |
| Resources | CPU, memory, restarts, and model startup time |

## Quality signals

Ground-truth quality cannot be calculated reliably from ordinary API traffic.
Premature responses, missed endings, precision, recall, and detection delay must
be measured on labeled samples using the causal replay protocol.

Online signals are useful as warnings, but are not ground-truth labels:

- A user speaking again after output begins is a possible premature response.
  Report the distribution of interruption offsets instead of treating one fixed
  duration as proof of an error.
- A client timeout or explicit "too slow" report after no end decision is a
  possible missed ending.
- Explicit feedback should distinguish `responded too early`, `responded too
  late`, and `correct`.
- Report signals by model version, audio quality, device class, and language
  when those fields are available.

The approved development result is the comparison baseline, not an acceptable
production target. Baseline v0 currently has a 24.05% premature-decision rate
and a 14.65% missed-ending rate on the English development replay.

## Drift detection

Compare rolling production summaries with the approved reference distribution:

- Input drift: audio duration, loudness, clipping, decode failures, sample rate,
  and noise estimates.
- Output drift: `p_end` histogram, hold/end ratio, and confidence near the
  decision threshold.
- Quality drift: labeled premature responses, missed endings, and detection
  delay by speech source and pause-duration slice.
- Traffic drift: concurrency, request size, and client/device mix.

Use a seven-day rolling reference after a representative pilot. Until that
reference exists, compare against the versioned offline evaluation artifact.
Run drift checks only after a minimum sample count to avoid alerts caused by
small batches.

## Initial alerts and dashboards

These are starting thresholds and should be adjusted from observed traffic:

| Condition | Initial action |
|---|---|
| Readiness fails twice consecutively | Page and stop routing new traffic to the instance |
| Server error rate exceeds 1% for 5 minutes with at least 100 requests | Page and inspect recent deployment changes |
| Warm server-total p95 exceeds 100 ms for 15 minutes | Warn and inspect load, queueing, and preprocessing |
| Warm server-total p95 exceeds 150 ms for 10 minutes | Page and reduce traffic or roll back |
| Output-distribution PSI exceeds 0.20 for three windows | Warn and review input and population drift |
| Labeled premature or missed-ending rate worsens by more than 5 percentage points | Block promotion or roll back the candidate |
| Artifact hash or preprocessing contract differs from the deployment manifest | Refuse startup |

Use three dashboard views:

1. Service health: availability, latency, throughput, errors, resources, and queueing.
2. Model behavior: `p_end`, decisions, threshold, drift, and input slices.
3. Quality and releases: labeled quality metrics, model versions, deployment events, and rollback status.

## Model versioning and rollback

- Store an immutable deployment manifest containing model version, model hash,
  metadata hash, preprocessing contract, threshold, source revision, and build ID.
- Return the model version and threshold from every prediction and attach the
  deployment identity to metrics.
- Keep the previous verified artifact available as the last-known-good version.
- Release through a small canary before increasing traffic.
- Roll back when parity fails, server errors or latency cross critical limits,
  or labeled quality regresses beyond the release guardrail.
- Treat a threshold change as a versioned model-policy release even when the
  ONNX file is unchanged.

## Feedback and retraining

1. Collect explicit feedback and operational events.
2. Review diagnostic samples and assign causal hold/end labels.
3. Canonicalize, deduplicate, and group samples before splitting them.
4. Add new data to training or development sets without modifying frozen final holdouts.
5. Record the data revision, preprocessing, seed, configuration, hardware, and checkpoint.
6. Run a smoke test, full training, causal development evaluation, export parity,
   API benchmark, and only then an untouched holdout.
7. Compare the candidate with the deployed baseline and record the promotion decision.
