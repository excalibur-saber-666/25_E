from pathlib import Path
import inspect
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import ot
import q3_pipeline as q3


class PotImportTests(unittest.TestCase):
    def test_official_pot_class_is_imported(self):
        self.assertTrue(hasattr(ot.da, "SinkhornLpl1Transport"))
        self.assertIn("SinkhornLpl1Transport", inspect.getsource(q3.fit_pot))

    def test_no_custom_sinkhorn_solver_exists(self):
        source = Path(q3.__file__).read_text(encoding="utf-8").lower()
        self.assertNotIn("def _sinkhorn", source)
        self.assertNotIn("def sinkhorn", source)


if __name__ == "__main__":
    unittest.main()
