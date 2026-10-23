"""Model definitions, preprocessing, and checkpoint loading."""

from .features import WaveformFeatureExtractor

__all__ = ["WaveformFeatureExtractor", "WhisperTurnDetector", "load_model_checkpoint"]


def __getattr__(name: str):
    if name in {"WhisperTurnDetector", "load_model_checkpoint"}:
        from .whisper_turn import WhisperTurnDetector, load_model_checkpoint

        return {
            "WhisperTurnDetector": WhisperTurnDetector,
            "load_model_checkpoint": load_model_checkpoint,
        }[name]
    raise AttributeError(name)
