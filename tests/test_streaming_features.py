import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
PREPROCESSOR = ROOT / "model/preprocessor"

try:
    import transformers  # noqa: F401

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

from turn_detector.models import StreamingWhisperFeatureBuffer, WaveformFeatureExtractor
from turn_detector.serving.streaming import StreamingSessionStore


@unittest.skipUnless(
    TRANSFORMERS_AVAILABLE and PREPROCESSOR.is_dir(),
    "local Whisper preprocessor is unavailable",
)
class StreamingFeatureBufferTest(unittest.TestCase):
    @staticmethod
    def extractor():
        return WaveformFeatureExtractor(
            PREPROCESSOR.as_posix(),
            8.0,
            16000,
            local_files_only=True,
            return_tensors="np",
        )

    def test_incremental_features_match_full_preprocessing(self):
        extractor = self.extractor()
        stream = StreamingWhisperFeatureBuffer(extractor)
        generator = np.random.default_rng(1337)
        received = np.empty(0, dtype=np.float32)

        for sample_count in (5601, 5599, 5763, 157, 16003, 64009):
            samples = generator.integers(
                -32767, 32768, size=sample_count, dtype=np.int16
            )
            received = np.concatenate(
                (received, samples.astype(np.float32) / 32768.0)
            )
            features, details = stream.append_pcm_s16le(
                samples.astype("<i2").tobytes()
            )
            processed = received[: details["total_processed_samples"]]
            waveform = processed[-stream.window_samples :]
            if len(waveform) < stream.window_samples:
                waveform = np.pad(
                    waveform, (stream.window_samples - len(waveform), 0)
                )
            expected = extractor.extractor._np_extract_fbank_features(
                np.asarray([waveform], dtype=np.float32), "cpu"
            )
            np.testing.assert_array_equal(features, expected)
            self.assertLess(details["buffered_samples"], stream.hop_length)

    def test_stream_sessions_are_isolated_and_bounded(self):
        store = StreamingSessionStore(self.extractor(), capacity=2)
        first = store.create()
        second = store.create()
        self.assertNotEqual(first.session_id, second.session_id)
        self.assertIsNot(first.features, second.features)
        with self.assertRaisesRegex(RuntimeError, "capacity"):
            store.create()
        self.assertTrue(store.delete(first.session_id))
        self.assertIsNone(store.get(first.session_id))
        self.assertEqual(store.snapshot()["active"], 1)


if __name__ == "__main__":
    unittest.main()
