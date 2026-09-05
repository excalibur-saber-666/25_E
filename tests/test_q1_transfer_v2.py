from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import q1_transfer_v2 as v2


class TransferV2Tests(unittest.TestCase):
    def test_fixed_angle_schema_and_finite_features(self):
        signal = np.sin(np.linspace(0, 96 * np.pi, 10000)) + .1 * np.random.default_rng(2).normal(size=10000)
        row = v2.transfer_v2_feature(signal)
        self.assertEqual(list(row), v2.feature_names())
        self.assertEqual(len(row), len(v2.feature_names()))
        self.assertTrue(np.isfinite(list(row.values())).all())
        self.assertEqual(v2.ANGLE_SAMPLES, 2048)


if __name__ == "__main__":
    unittest.main()
