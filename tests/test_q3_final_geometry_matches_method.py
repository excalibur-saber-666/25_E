import unittest

import pandas as pd


class FinalGeometryTests(unittest.TestCase):
    def test_final_geometry_is_file_aligned_with_final_method(self):
        final = pd.read_csv("outputs/q3_refined/target_predictions_final.csv")
        geometry = pd.read_csv("outputs/q3_refined/target_class_geometry_final.csv")
        merged = final[["file_id", "final_method", "candidate_label"]].merge(
            geometry[["file_id", "final_method", "final_candidate_label"]], on="file_id")
        self.assertEqual(len(merged), 16)
        self.assertTrue((merged.final_method_x == merged.final_method_y).all())
        self.assertTrue((merged.candidate_label == merged.final_candidate_label).all())


if __name__ == "__main__":
    unittest.main()
