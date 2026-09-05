from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from q3_pipeline import LABELS


class CoralFinalOutputTests(unittest.TestCase):
    def setUp(self):
        self.final = pd.read_csv("outputs/q3/target_predictions_final.csv")
        self.coral = pd.read_csv("outputs/q3/coral_predictions.csv")

    def test_final_label_matches_final_probabilities(self):
        labels = np.asarray(LABELS)[self.final[[f"prob_{label}" for label in LABELS]].to_numpy().argmax(axis=1)]
        np.testing.assert_array_equal(self.final.final_candidate_label.to_numpy(), labels)

    def test_coral_final_fields_are_coral_fields(self):
        self.assertTrue((self.final.final_method == "CORAL").all())
        fields = ["confidence", "window_vote_ratio", "normalized_entropy", "probability_margin", "window_probability_std", *[f"prob_{label}" for label in LABELS]]
        merged = self.final[["file_id", *fields]].merge(self.coral[["file_id", *fields]], on="file_id", suffixes=("_final", "_coral"))
        self.assertEqual(len(merged), 16)
        for field in fields:
            np.testing.assert_allclose(merged[f"{field}_final"], merged[f"{field}_coral"], atol=1e-12)


if __name__ == "__main__":
    unittest.main()
