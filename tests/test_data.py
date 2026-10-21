from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from turn_detector.data import PipecatDataset  # noqa: E402


class PipecatDatasetTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = PipecatDataset(
            ROOT / "data/pipecat_smart_turn_v3_2_sample/dataset.sqlite3"
        )

    def test_balanced_dataset(self):
        self.assertEqual(len(self.dataset), 50)
        self.assertEqual(
            self.dataset.class_counts(), {"complete": 25, "incomplete": 25}
        )

    def test_audio_paths_exist(self):
        examples = list(self.dataset.iter_examples())
        self.assertEqual(len(examples), 50)
        self.assertTrue(all(example.audio_path.is_file() for example in examples))

    def test_filter(self):
        complete = list(self.dataset.iter_examples(label="complete"))
        self.assertEqual(len(complete), 25)
        self.assertTrue(all(example.endpoint_bool for example in complete))


if __name__ == "__main__":
    unittest.main()
