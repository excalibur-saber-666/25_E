from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from q3_pipeline import load_data


class TargetIsolationTests(unittest.TestCase):
    def test_target_label_column_is_rejected(self):
        source = pd.read_csv("outputs/q1/features_source_transfer.csv")
        target = pd.read_csv("outputs/q1/features_target_transfer.csv")
        target["label"] = "N"
        # The guard is deliberately before schema/metadata use: target labels
        # must be rejected rather than silently ignored.
        with patch("q3_pipeline.pd.read_csv", side_effect=[source, target]):
            with self.assertRaisesRegex(ValueError, "Target labels are forbidden"):
                load_data(Path("outputs/q1"), Path("outputs/q2_refined/models"))


if __name__ == "__main__":
    unittest.main()
