# ONNX Runtime CPU tuning v1

## Scope

- Model: `english-whisper-tiny-baseline-v0`
- Model SHA-256: `b79ad05c253996a73969454496583dbce5dcf85bfec9a6b2fdf16e7d9858d9bf`
- Precision/provider: FP32, ONNX Runtime CPUExecutionProvider
- Threshold: `0.5`
- Container CPUs visible: `4`
- Input: one 11-second, 16 kHz mono PCM16 recording replayed chronologically
- Streaming update: `350 ms`
- Training data and evaluation holdouts were not changed or used for tuning.

The existing defaults already used sequential execution and full graph
optimization. The tuning variable was therefore CPU thread allocation.

## Candidate screen

Each candidate used one 32-update chronological replay. These small results were
used only to select candidates for the repeated comparison.

| Execution | Intra/inter threads | Client p50 | Client p95 | Inference p50 | Inference p95 |
|---|---:|---:|---:|---:|---:|
| Sequential | automatic/automatic | 95.97 ms | 111.92 ms | 85.92 ms | 102.72 ms |
| Sequential | 1/1 | 106.32 ms | 121.59 ms | 99.42 ms | 115.99 ms |
| Sequential | 2/1 | 59.42 ms | 68.74 ms | 54.27 ms | 60.48 ms |
| Sequential | 3/1 | 49.27 ms | 96.12 ms | 42.89 ms | 88.04 ms |
| Sequential | 4/1 | 92.97 ms | 122.96 ms | 84.70 ms | 113.30 ms |
| Parallel | 4/2 | 114.87 ms | 176.65 ms | 107.12 ms | 167.66 ms |

## Repeated comparison

The baseline and two strongest candidates were each measured in three runs. A
fresh container was created for every run, followed by warmup and 96 streaming
updates. The table reports the median of the three run-level percentiles.

| Configuration | Client p50 | Client p95 | Client p99 | Inference p50 | Inference p95 | Inference p99 |
|---|---:|---:|---:|---:|---:|---:|
| Automatic threads | 93.05 ms | 126.07 ms | 162.37 ms | 82.86 ms | 116.33 ms | 150.41 ms |
| Sequential 2/1 | 60.61 ms | 99.40 ms | 116.06 ms | 56.13 ms | 91.74 ms | 105.77 ms |
| Sequential 3/1 | 49.44 ms | 100.35 ms | 126.33 ms | 43.76 ms | 83.47 ms | 108.73 ms |

Sequential 2/1 reduced client p50 by `34.9%`, p95 by `21.2%`, and p99 by
`28.5%` relative to automatic threading. Its three run-level p95 measurements
were `99.40`, `111.52`, and `97.28 ms`; therefore the median meets 100 ms, but
the result is not reliably below 100 ms in every run.

Sequential 2/1 was selected over 3/1 because it had the lower and more stable
p95, which is the serving objective. All measured requests completed without an
error or decision mismatch.

## Correctness and stress checks

- PyTorch/direct ONNX/stateless API/streaming API parity passed.
- Maximum streaming probability difference: `0.00000048`.
- Streaming decision mismatches: `0`.
- Full-file `/predict` stress completed 150 requests without API errors, but
  latency degraded sharply at concurrency four. The four-CPU single-container
  service should not be presented as high-concurrency ready without admission
  control, replicas, or additional CPU capacity.

Raw results are under
`experiments/english_whisper_tiny_v1/baseline_v0/onnx_tuning/`.
