from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import q1_transfer_v2 as v2


class TransferV2DuplicateTests(unittest.TestCase):
    def test_schema_has_no_duplicate_feature_name_or_known_duplicate(self):
        names = v2.feature_names()
        self.assertEqual(len(names), len(set(names)))
        self.assertNotIn("angle_psd_entropy", names)
        self.assertIn("angle_order_entropy", names)


if __name__ == "__main__":
    unittest.main()
