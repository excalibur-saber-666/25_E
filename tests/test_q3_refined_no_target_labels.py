from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from q3_refined_pipeline import load_inputs


class RefinedTargetIsolationTests(unittest.TestCase):
    def test_target_label_column_is_rejected(self):
        source = pd.read_csv("outputs/q3_refined/features_source_transfer_v2.csv")
        target = pd.read_csv("outputs/q3_refined/features_target_transfer_v2_rpm600.csv")
        target["label"] = "N"
        with patch("q3_refined_pipeline.pd.read_csv", side_effect=[source, target]):
            with self.assertRaisesRegex(ValueError, "Target labels are forbidden"):
                load_inputs(Path("outputs/q3_refined"))


if __name__ == "__main__":
    unittest.main()
