import json
import sqlite3
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    import pyarrow  # noqa: F401
    import soundfile  # noqa: F401
    import torch  # noqa: F401

    DATA_DEPS_AVAILABLE = True
except ImportError:
    DATA_DEPS_AVAILABLE = False

from turn_detector.data_loader import CanonicalPipecatDataset


FULL_DATA_ROOT = ROOT / "data" / "pipecat_smart_turn_v3_2_full"
CANONICAL_INDEX = ROOT / "data" / "canonical" / "dataset.sqlite3"


@unittest.skipUnless(
    DATA_DEPS_AVAILABLE
    and (FULL_DATA_ROOT / "snapshot_manifest.json").is_file()
    and CANONICAL_INDEX.is_file(),
    "complete snapshot, canonical index, or training dependencies are unavailable",
)
class FullPipecatDatasetTest(unittest.TestCase):
    def test_verified_counts(self):
        manifest = json.loads(
            (FULL_DATA_ROOT / "snapshot_manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(manifest["shard_count"], 83)
        with sqlite3.connect(CANONICAL_INDEX) as connection:
            total, english, complete, incomplete = connection.execute(
                """
                SELECT
                    COUNT(*),
                    SUM(language = 'eng'),
                    SUM(language = 'eng' AND label = 'end'),
                    SUM(language = 'eng' AND label = 'hold')
                FROM decisions WHERE source = 'pipecat'
                """
            ).fetchone()
        self.assertEqual(total, 270_946)
        self.assertEqual(english, 65_802)
        self.assertEqual(complete, 32_443)
        self.assertEqual(incomplete, 33_359)

    def test_embedded_audio_decodes(self):
        dataset = CanonicalPipecatDataset(
            CANONICAL_INDEX,
            split="train",
            language="eng",
            shuffle=False,
            max_examples=1,
        )
        example = next(iter(dataset))
        self.assertEqual(tuple(example["waveform"].shape), (128_000,))
        self.assertIn(float(example["label"]), (0.0, 1.0))


if __name__ == "__main__":
    unittest.main()
