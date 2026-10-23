"""Shared waveform-to-Whisper feature preprocessing."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np


class WaveformFeatureExtractor:
    def __init__(
        self,
        backbone: str,
        window_seconds: float,
        sample_rate: int,
        revision: str | None = None,
        local_files_only: bool = False,
        return_tensors: str = "pt",
    ):
        from transformers import WhisperFeatureExtractor

        self.extractor = WhisperFeatureExtractor.from_pretrained(
            backbone,
            chunk_length=window_seconds,
            revision=revision,
            local_files_only=local_files_only,
        )
        self.sample_rate = sample_rate
        self.window_seconds = window_seconds
        self.return_tensors = return_tensors

    def __call__(self, waveforms: Sequence[Any]):
        arrays = [
            waveform.detach().cpu().numpy()
            if hasattr(waveform, "detach")
            else np.asarray(waveform, dtype=np.float32)
            for waveform in waveforms
        ]
        return self.extractor(
            arrays,
            sampling_rate=self.sample_rate,
            return_tensors=self.return_tensors,
        ).input_features

    def save_pretrained(self, output: str) -> None:
        self.extractor.save_pretrained(output)
