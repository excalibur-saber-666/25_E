from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from q3_refined_pipeline import LABELS


class RefinedFinalProbabilityTests(unittest.TestCase):
    def test_final_label_is_probability_argmax(self):
        final = pd.read_csv("outputs/q3_refined/target_predictions_final.csv")
        self.assertEqual(final.file_id.nunique(), 16)
        expected = np.asarray(LABELS)[final[[f"prob_{label}" for label in LABELS]].to_numpy().argmax(axis=1)]
        np.testing.assert_array_equal(expected, final.candidate_label.to_numpy())
        self.assertFalse(any("accuracy" in name.lower() or "recall" in name.lower() for name in final.columns))


if __name__ == "__main__":
    unittest.main()
