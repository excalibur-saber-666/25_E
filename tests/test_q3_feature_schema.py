from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import q3_pipeline as q3


class SchemaTests(unittest.TestCase):
    def test_frozen_q2_interface_is_diagnostic26_and_32d(self):
        names, _, encoder, config = q3.load_frozen_q2(Path("outputs/q2"))
        self.assertEqual(len(names), 26)
        self.assertEqual(config["label_order"], list(q3.LABELS))
        self.assertFalse(encoder.training)
        self.assertTrue(all(not parameter.requires_grad for parameter in encoder.parameters()))


if __name__ == "__main__":
    unittest.main()
