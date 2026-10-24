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


class StreamingWhisperFeatureBuffer:
    """Incrementally maintain a fixed Whisper log-mel window."""

    def __init__(self, feature_extractor: WaveformFeatureExtractor):
        from transformers.audio_utils import spectrogram, window_function

        extractor = feature_extractor.extractor
        if float(extractor.dither) != 0.0:
            raise ValueError("streaming feature reuse requires dither=0")
        self.extractor = extractor
        self.spectrogram = spectrogram
        self.window = window_function(extractor.n_fft, "hann")
        self.sample_rate = int(feature_extractor.sample_rate)
        self.window_samples = int(round(feature_extractor.window_seconds * self.sample_rate))
        self.hop_length = int(extractor.hop_length)
        self.frame_count = int(extractor.nb_max_frames)
        self.waveform = np.zeros(self.window_samples, dtype=np.float32)
        self.pending = np.empty(0, dtype=np.float32)
        self.total_received_samples = 0
        self.total_processed_samples = 0
        self.raw_log_mel = self._full_raw_log_mel()

    def _full_raw_log_mel(self) -> np.ndarray:
        return self.spectrogram(
            self.waveform,
            self.window,
            frame_length=self.extractor.n_fft,
            hop_length=self.hop_length,
            power=2.0,
            mel_filters=self.extractor.mel_filters,
            log_mel="log10",
        )[:, :-1]

    def _selected_raw_log_mel(self, indices: np.ndarray) -> np.ndarray:
        padding = self.extractor.n_fft // 2
        waveform = np.pad(self.waveform, (padding, padding), mode="reflect").astype(
            np.float64
        )
        window = self.window.astype(np.float64)
        frames = np.stack(
            [
                waveform[
                    index * self.hop_length : index * self.hop_length
                    + self.extractor.n_fft
                ]
                for index in indices
            ]
        )
        spectra = np.fft.rfft(frames * window, axis=1).astype(np.complex64)
        power = np.abs(spectra, dtype=np.float64) ** 2
        mel = np.maximum(
            1e-10,
            np.dot(self.extractor.mel_filters.T, power.T),
        )
        return np.asarray(np.log10(mel), dtype=np.float32)

    def _advance(self, samples: np.ndarray) -> tuple[int, int]:
        sample_count = len(samples)
        frame_shift = sample_count // self.hop_length
        if sample_count >= self.window_samples or frame_shift >= self.frame_count - 2:
            self.waveform = samples[-self.window_samples :]
            if len(self.waveform) < self.window_samples:
                self.waveform = np.pad(
                    self.waveform, (self.window_samples - len(self.waveform), 0)
                )
            self.raw_log_mel = self._full_raw_log_mel()
            return self.frame_count, 0

        self.waveform[:-sample_count] = self.waveform[sample_count:]
        self.waveform[-sample_count:] = samples
        updated = np.empty_like(self.raw_log_mel)
        reuse_end = self.frame_count - frame_shift - 1
        if reuse_end > 2:
            updated[:, 2:reuse_end] = self.raw_log_mel[
                :, 2 + frame_shift : self.frame_count - 1
            ]
        recompute = np.concatenate(
            (
                np.arange(0, 2),
                np.arange(max(2, reuse_end), self.frame_count),
            )
        )
        updated[:, recompute] = self._selected_raw_log_mel(recompute)
        self.raw_log_mel = updated
        return len(recompute), self.frame_count - len(recompute)

    def append_pcm_s16le(self, payload: bytes) -> tuple[np.ndarray, dict[str, int]]:
        if not payload or len(payload) % 2:
            raise ValueError("PCM payload must contain complete 16-bit samples")
        samples = np.frombuffer(payload, dtype="<i2").astype(np.float32) / 32768.0
        self.total_received_samples += len(samples)
        combined = np.concatenate((self.pending, samples))
        process_count = len(combined) // self.hop_length * self.hop_length
        processed = combined[:process_count]
        self.pending = combined[process_count:]
        if process_count:
            recomputed, reused = self._advance(processed)
            self.total_processed_samples += process_count
        else:
            recomputed, reused = 0, self.frame_count

        features = np.maximum(
            self.raw_log_mel,
            self.raw_log_mel.max() - 8.0,
        )
        features = np.asarray((features + 4.0) / 4.0, dtype=np.float32)
        return np.ascontiguousarray(features[np.newaxis]), {
            "received_samples": len(samples),
            "processed_samples": process_count,
            "buffered_samples": len(self.pending),
            "feature_frames_recomputed": recomputed,
            "feature_frames_reused": reused,
            "total_received_samples": self.total_received_samples,
            "total_processed_samples": self.total_processed_samples,
        }
