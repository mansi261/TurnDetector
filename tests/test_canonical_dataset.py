import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data/canonical"


class CanonicalDatasetTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        subprocess.run(
            [sys.executable, str(ROOT / "scripts/build_canonical_dataset.py")],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )

    def test_record_counts(self):
        summary = json.loads((OUTPUT / "summary.json").read_text())
        self.assertEqual(summary["train"]["records"], 50)
        self.assertEqual(summary["evaluation"]["records"], 154)
        self.assertEqual(summary["evaluation"]["hold"], 104)
        self.assertEqual(summary["evaluation"]["end"], 50)

    def test_modalities_are_fair(self):
        with sqlite3.connect(OUTPUT / "dataset.sqlite3") as connection:
            transcript_count, context_count = connection.execute(
                """
                SELECT COUNT(model_transcript), COUNT(model_dialogue_context)
                FROM decisions
                """
            ).fetchone()
        self.assertEqual((transcript_count, context_count), (0, 0))

    def test_livekit_is_causal_at_200ms(self):
        with sqlite3.connect(OUTPUT / "dataset.sqlite3") as connection:
            violations = connection.execute(
                """
                SELECT COUNT(*) FROM decisions
                WHERE split = 'evaluation'
                  AND decision_time_s > silence_start_s + 0.200001
                """
            ).fetchone()[0]
        self.assertEqual(violations, 0)


if __name__ == "__main__":
    unittest.main()
