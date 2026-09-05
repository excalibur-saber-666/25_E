from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from q3_pipeline import coral_align, proxy_a_distance, rbf_mmd


class DomainMetricTests(unittest.TestCase):
    def test_coral_is_finite_and_covariance_is_aligned(self):
        rng = np.random.default_rng(7)
        source, target = rng.normal(size=(30, 6)), rng.normal(loc=.5, scale=1.5, size=(20, 6))
        aligned = coral_align(source, target)
        self.assertTrue(np.isfinite(aligned).all())
        self.assertTrue(np.allclose(np.cov(aligned, rowvar=False), np.cov(target, rowvar=False), atol=2e-3))

    def test_mmd_and_pad_are_finite(self):
        rng = np.random.default_rng(8)
        x, y = rng.normal(size=(20, 4)), rng.normal(loc=.2, size=(12, 4))
        self.assertGreaterEqual(rbf_mmd(x, y), 0)
        values = proxy_a_distance(x, y)
        self.assertTrue(np.isfinite(values["proxy_a_distance"]))
        self.assertGreaterEqual(values["proxy_a_distance"], 0)
        self.assertLessEqual(values["proxy_a_distance"], 2)


if __name__ == "__main__":
    unittest.main()
