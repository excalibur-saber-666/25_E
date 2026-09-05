from pathlib import Path
import sys
import unittest

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from q3_refined_pipeline import SEEDS


class RefinedSeedStabilityTests(unittest.TestCase):
    def test_all_fixed_encoder_seeds_are_recorded(self):
        table = pd.read_csv("outputs/q3_refined/encoder_seed_predictions.csv")
        self.assertEqual(set(table.seed), set(SEEDS))
        self.assertEqual(table.groupby("file_id").size().to_dict(), {letter: 5 for letter in "ABCDEFGHIJKLMNOP"})


if __name__ == "__main__":
    unittest.main()
