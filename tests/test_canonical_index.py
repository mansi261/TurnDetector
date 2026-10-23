import json
from pathlib import Path
import sqlite3
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from turn_detector.data_fetching.build_canonical_index import deterministic_split


DATABASE = ROOT / "data/canonical/dataset.sqlite3"
SUMMARY = ROOT / "data/canonical/summary.json"


class CanonicalSplitTest(unittest.TestCase):
    def test_split_is_deterministic(self):
        first = deterministic_split("example-id", "seed", 0.1)
        second = deterministic_split("example-id", "seed", 0.1)
        self.assertEqual(first, second)
        self.assertIn(first, {"train", "validation"})


@unittest.skipUnless(
    DATABASE.is_file() and SUMMARY.is_file(),
    "complete local canonical index is unavailable",
)
class CanonicalIndexIntegrationTest(unittest.TestCase):
    def test_record_counts(self):
        summary = json.loads(SUMMARY.read_text())
        training_source_records = (
            summary["splits"]["train"]["records"]
            + summary["splits"]["validation"]["records"]
        )
        self.assertEqual(training_source_records, 270_946)
        self.assertEqual(summary["languages"]["eng"], 65_802)
        self.assertEqual(summary["splits"]["evaluation"]["records"], 1_250)
        self.assertEqual(summary["evaluation_partitions"]["ENG_evaluation_dev"]["turns"], 198)
        self.assertEqual(summary["evaluation_partitions"]["ENG_holdout_a"]["turns"], 102)
        self.assertEqual(summary["evaluation_partitions"]["ENG_holdout_b"]["turns"], 100)

    def test_language_indexes_exist(self):
        with sqlite3.connect(DATABASE) as connection:
            indexes = {
                row[0]
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'index'"
                )
            }
        self.assertIn("decisions_language_idx", indexes)
        self.assertIn("decisions_split_language_label_idx", indexes)
        self.assertIn("decisions_evaluation_partition_idx", indexes)

    def test_modalities_are_consistent(self):
        with sqlite3.connect(DATABASE) as connection:
            transcript_count, context_count = connection.execute(
                """
                SELECT COUNT(model_transcript), COUNT(model_dialogue_context)
                FROM decisions
                """
            ).fetchone()
        self.assertEqual((transcript_count, context_count), (0, 0))

    def test_evaluation_preserves_full_spans_without_fixed_decision_time(self):
        with sqlite3.connect(DATABASE) as connection:
            fixed_decisions, invalid_spans = connection.execute(
                """
                SELECT SUM(decision_time_s IS NOT NULL),
                       SUM(silence_start_s < 0 OR silence_end_s <= silence_start_s
                           OR silence_end_s > source_duration_s + 0.000001)
                FROM decisions WHERE split = 'evaluation'
                """
            ).fetchone()
        self.assertEqual((fixed_decisions, invalid_spans), (0, 0))

    def test_evaluation_conversation_groups_are_disjoint(self):
        with sqlite3.connect(DATABASE) as connection:
            overlaps = connection.execute(
                """
                SELECT COUNT(*) FROM (
                    SELECT group_id FROM decisions
                    WHERE split = 'evaluation'
                    GROUP BY group_id HAVING COUNT(DISTINCT evaluation_partition) > 1
                )
                """
            ).fetchone()[0]
        self.assertEqual(overlaps, 0)


if __name__ == "__main__":
    unittest.main()
