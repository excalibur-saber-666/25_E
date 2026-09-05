from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from q3_pipeline import coral_align, file_balanced_weights, proxy_a_distance, rbf_mmd, weighted_mean_covariance


class DomainMetricTests(unittest.TestCase):
    def test_coral_is_finite_and_file_balanced(self):
        rng = np.random.default_rng(7)
        source, target = rng.normal(size=(30, 6)), rng.normal(loc=.5, scale=1.5, size=(20, 6))
        source_ids = np.repeat(["s1", "s2", "s3"], [4, 10, 16])
        target_ids = np.repeat(["a", "b"], [5, 15])
        aligned = coral_align(source, target, source_ids, target_ids)
        self.assertTrue(np.isfinite(aligned).all())
        _, covariance, weights = weighted_mean_covariance(source, source_ids, 1e-4)
        self.assertTrue(np.allclose(covariance, covariance.T))
        self.assertAlmostEqual(float(weights.sum()), 1.0)
        self.assertAlmostEqual(float(weights[source_ids == "s1"].sum()), 1 / 3)
        self.assertAlmostEqual(float(weights[source_ids == "s2"].sum()), 1 / 3)

    def test_copying_windows_does_not_change_file_total_weight(self):
        ids = np.array(["a", "a", "b"])
        copied_ids = np.array(["a", "a", "a", "a", "b"])
        x = np.array([[0., 1.], [2., 3.], [10., 11.]])
        copied_x = np.vstack([x[:2], x[:2], x[2:]])
        mean, cov, _ = weighted_mean_covariance(x, ids)
        copied_mean, copied_cov, _ = weighted_mean_covariance(copied_x, copied_ids)
        np.testing.assert_allclose(mean, copied_mean)
        np.testing.assert_allclose(cov, copied_cov)
        np.testing.assert_allclose(file_balanced_weights(ids)[[0, 1]].sum(), .5)

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
