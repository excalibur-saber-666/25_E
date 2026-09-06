from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from q1_transfer_v2 import feature_names
from q3_refined_pipeline import ablation_sets


class TransferV2AblationTests(unittest.TestCase):
    def test_all_predeclared_ablation_sets_are_distinct(self):
        sets = ablation_sets(feature_names())
        self.assertEqual(set(sets), {"full", "no_absolute_hz", "no_envelope", "no_order"})
        self.assertEqual(len({tuple(v) for v in sets.values()}), 4)
        self.assertTrue(all(values for values in sets.values()))


if __name__ == "__main__":
    unittest.main()
