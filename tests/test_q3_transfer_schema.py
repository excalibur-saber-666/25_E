from pathlib import Path
import json
import sys
import unittest

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from q3_pipeline import load_data


class TransferSchemaTests(unittest.TestCase):
    def test_frozen_transfer20_schema_and_file_counts(self):
        source, target, names, audit = load_data(Path("outputs/q1"), Path("outputs/q2_refined/models"))
        saved = json.loads(Path("outputs/q2_refined/models/q2_transfer20_feature_names.json").read_text(encoding="utf-8"))["features"]
        self.assertEqual(names, saved)
        self.assertEqual(len(names), 20)
        self.assertEqual(source.file_id.nunique(), 56)
        self.assertEqual(target.file_id.nunique(), 16)
        self.assertEqual(sorted(target.file_id.unique()), list("ABCDEFGHIJKLMNOP"))
        self.assertFalse(audit["target_label_column_present"])


if __name__ == "__main__":
    unittest.main()
