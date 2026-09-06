from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from q3_refined_pipeline import _method_probabilities, LABELS


class SurrogateTargetIsolationTests(unittest.TestCase):
    def test_adaptation_refuses_a_labelled_pseudo_target(self):
        source = pd.DataFrame({"file_id": ["s1", "s2", "s3", "s4"], "label": list(LABELS)})
        target = pd.DataFrame({"file_id": ["t1", "t2"], "label": ["N", "B"]})
        zsource = np.eye(4); ztarget = np.zeros((2, 4)); source_prob = np.full((2, 4), .25)
        with self.assertRaisesRegex(ValueError, "must not enter adaptation"):
            _method_probabilities("source_only", source, target, zsource, ztarget, source_prob, 2025)


if __name__ == "__main__":
    unittest.main()
