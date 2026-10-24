"""Shared classification metrics and report formatting."""

from __future__ import annotations

from typing import Any

import numpy as np


def calibration(
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    log_loss_key: str = "log_loss",
) -> dict[str, Any]:
    clipped = np.clip(probabilities.astype(np.float64), 1e-7, 1.0 - 1e-7)
    labels_float = labels.astype(np.float64)
    bins = []
    ece = 0.0
    for index in range(10):
        lower = index / 10
        upper = (index + 1) / 10
        mask = (probabilities >= lower) & (
            probabilities <= upper if index == 9 else probabilities < upper
        )
        count = int(mask.sum())
        if count:
            mean_score = float(probabilities[mask].mean())
            observed_rate = float(labels_float[mask].mean())
            gap = abs(mean_score - observed_rate)
            ece += count / len(labels) * gap
        else:
            mean_score = observed_rate = gap = None
        bins.append(
            {
                "range": [lower, upper],
                "count": count,
                "mean_p_end": mean_score,
                "observed_end_rate": observed_rate,
                "absolute_gap": gap,
            }
        )
    result = {
        "brier_score": float(np.mean((probabilities - labels_float) ** 2)),
        "expected_calibration_error_10_bins": float(ece),
        "bins": bins,
    }
    result[log_loss_key] = float(
        -np.mean(
            labels_float * np.log(clipped)
            + (1 - labels_float) * np.log(1 - clipped)
        )
    )
    return result


def score_distribution(
    labels: np.ndarray, probabilities: np.ndarray
) -> dict[str, Any]:
    def summarize(values: np.ndarray) -> dict[str, float]:
        return {
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "p10": float(np.percentile(values, 10)),
            "p90": float(np.percentile(values, 90)),
        }

    return {
        "hold": summarize(probabilities[labels == 0]),
        "end": summarize(probabilities[labels == 1]),
    }


def format_percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def format_percentage_delta(value: float) -> str:
    return f"{value * 100:+.2f} pp"
