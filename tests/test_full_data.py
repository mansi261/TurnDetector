import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    import pyarrow  # noqa: F401
    import soundfile  # noqa: F401

    DATA_DEPS_AVAILABLE = True
except ImportError:
    DATA_DEPS_AVAILABLE = False

from turn_detector.data_loader import PipecatParquetDataset


FULL_DATA_ROOT = ROOT / "data" / "pipecat_smart_turn_v3_2_full"


@unittest.skipUnless(
    DATA_DEPS_AVAILABLE and (FULL_DATA_ROOT / "snapshot_manifest.json").is_file(),
    "complete local snapshot or optional data dependencies are unavailable",
)
class FullPipecatDatasetTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = PipecatParquetDataset(FULL_DATA_ROOT)

    def test_verified_counts(self):
        self.assertEqual(len(self.dataset.shards), 83)
        self.assertEqual(self.dataset.count_rows(), 270_946)
        self.assertEqual(self.dataset.count_rows(language="eng"), 65_802)
        self.assertEqual(
            self.dataset.count_rows(language="eng", label="complete"), 32_443
        )
        self.assertEqual(
            self.dataset.count_rows(language="eng", label="incomplete"), 33_359
        )

    def test_embedded_audio_decodes(self):
        import soundfile as sf
        from io import BytesIO

        example = next(self.dataset.iter_examples(language="eng", batch_size=1))
        audio, sample_rate = sf.read(BytesIO(example.audio_bytes), always_2d=True)
        self.assertEqual(sample_rate, 16_000)
        self.assertEqual(audio.shape[1], 1)
        self.assertGreater(audio.shape[0], 0)


if __name__ == "__main__":
    unittest.main()
