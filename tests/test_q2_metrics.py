from pathlib import Path
import sys
import unittest
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import q2_pipeline as q


class MetricTests(unittest.TestCase):
    def test_window_probabilities_are_arithmetic_file_means(self):
        frame = pd.DataFrame({"file_id": ["a", "a", "b"], "label": ["N", "N", "B"], "load": [0, 0, 1]})
        probabilities = np.array([[.8, .1, .05, .05], [.2, .4, .2, .2], [.1, .6, .1, .2]])
        output = q.aggregate_window_probabilities(frame, probabilities).set_index("file_id")
        np.testing.assert_allclose(output.loc["a", [f"prob_{x}" for x in q.LABELS]].to_numpy(float), probabilities[:2].mean(axis=0))
        self.assertEqual(output.loc["a", "predicted_label"], "N")

    def test_metrics_have_fixed_four_class_order(self):
        metrics = q.class_metrics(np.array(q.LABELS), np.array(q.LABELS))
        self.assertEqual(metrics["macro_f1"], 1.0)
        self.assertEqual([key for key in metrics if key.startswith("recall_")], ["recall_N", "recall_B", "recall_IR", "recall_OR"])


if __name__ == "__main__":
    unittest.main()
