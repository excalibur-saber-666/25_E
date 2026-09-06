from pathlib import Path
import sys
import unittest
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import q3_pipeline as q3


class FinalOutputTests(unittest.TestCase):
    def test_sixteen_unlabelled_candidates_match_probability_argmax(self):
        table = pd.read_csv("outputs/q3/target_predictions.csv")
        self.assertEqual(len(table), 16)
        self.assertFalse(set(["true_label", "target_label", "label"]) & set(table.columns))
        probabilities = table[[f"prob_{label}" for label in q3.LABELS]].to_numpy(float)
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
        self.assertTrue(np.array_equal(table.candidate_label, np.asarray(q3.LABELS)[probabilities.argmax(axis=1)]))

    def test_verification_records_official_pot_and_target_isolation(self):
        import json
        record = json.loads(Path("outputs/q3/verification.json").read_text(encoding="utf-8"))
        self.assertTrue(record["pot_used"])
        self.assertFalse(record["custom_sinkhorn_used"])
        self.assertFalse(record["target_labels_used"])
        self.assertEqual(record["target_file_count"], 16)
        self.assertTrue(record["formal_pot_converged"])
        self.assertEqual(record["formal_pot_convergence_warning_count"], 0)


if __name__ == "__main__":
    unittest.main()
