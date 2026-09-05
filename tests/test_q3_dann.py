from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from q3_pipeline import GradientReverse, DomainDiscriminator


class DANNTests(unittest.TestCase):
    def test_gradient_reversal_reverses_gradient(self):
        x = torch.tensor([2.0], requires_grad=True)
        GradientReverse.apply(x, .3).sum().backward()
        self.assertAlmostEqual(float(x.grad[0]), -.3)

    def test_domain_discriminator_has_one_logit(self):
        disc = DomainDiscriminator()
        result = disc(torch.randn(5, 32), 1.)
        self.assertEqual(tuple(result.shape), (5, 1))


if __name__ == "__main__":
    unittest.main()
