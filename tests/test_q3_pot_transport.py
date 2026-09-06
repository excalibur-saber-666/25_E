from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import q3_pipeline as q3


class PotTransportTests(unittest.TestCase):
    def test_file_balanced_official_transport_is_finite(self):
        rng = np.random.default_rng(2025)
        source = rng.normal(size=(8, 32)); target = rng.normal(size=(4, 32)); labels = np.repeat(np.arange(4), 2)
        _, moved, coupling, source_weights, target_weights = q3.fit_pot(source, labels, target)
        self.assertEqual(moved.shape, source.shape)
        self.assertEqual(coupling.shape, (8, 4))
        self.assertTrue(np.isfinite(moved).all())
        self.assertTrue(np.isfinite(coupling).all())
        np.testing.assert_allclose(source_weights, np.full(8, 1 / 8))
        np.testing.assert_allclose(target_weights, np.full(4, 1 / 4))


if __name__ == "__main__":
    unittest.main()
