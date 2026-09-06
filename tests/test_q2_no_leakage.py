from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import q2_pipeline as q


class LeakageTests(unittest.TestCase):
    def test_file_split_is_disjoint_and_lolo_holds_one_load(self):
        files = pd.DataFrame([{"file_id": f"{label}{load}", "label": label, "load": load} for label in q.LABELS for load in range(4)])
        for _, train, test in q.outer_splits(files, "lolo"):
            self.assertFalse(set(files.iloc[train].file_id) & set(files.iloc[test].file_id))
            self.assertEqual(len(files.iloc[test].load.unique()), 1)

    def test_window_sampling_gives_equal_class_file_mass_and_scaler_uses_train_only(self):
        train = pd.DataFrame([{"file_id": f"{label}{i}", "label": label, "load": 0, "x": float(j)} for label in q.LABELS for i in range(2) for j in range(2 + i * 4)])
        weights = q.window_weights(train)
        mass = train.assign(weight=weights).groupby(["label", "file_id"]).weight.sum()
        np.testing.assert_allclose(mass.to_numpy(), np.full(8, 1 / 8))
        scaler = q.fit_weighted_scaler(train, ["x"])
        self.assertAlmostEqual(scaler.mean_[0], np.average(train.x, weights=weights))

    def test_source_loader_does_not_open_target_files(self):
        opened, original = [], pd.read_csv
        def reader(path, *args, **kwargs):
            opened.append(str(path)); self.assertNotIn("target", str(path).lower()); return original(path, *args, **kwargs)
        with patch.object(pd, "read_csv", side_effect=reader):
            windows, files, names = q.load_inputs(Path("outputs/q1"))
        self.assertEqual((len(windows), len(files), len(names)), (806, 56, 26))
        self.assertEqual(len(opened), 2)


if __name__ == "__main__":
    unittest.main()
