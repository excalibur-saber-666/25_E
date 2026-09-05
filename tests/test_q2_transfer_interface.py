from pathlib import Path
import sys
import json
import unittest
import numpy as np
import pandas as pd
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from q2_transfer_pretrain import load_interface


class InterfaceTests(unittest.TestCase):
    def test_saved_transfer20_interface(self):
        names=json.loads(Path('outputs/q1/feature_names_transfer.json').read_text())['features']
        encoder,head,scaler,schema=load_interface('outputs/q2_refined/models',expected_features=names)
        self.assertEqual(len(schema),20)
        values=pd.read_csv('outputs/q1/features_source_transfer.csv')[names].iloc[:5].to_numpy(float)
        x=torch.tensor(scaler.transform(values),dtype=torch.float32)
        with torch.no_grad():
            embedding=encoder(x); logits=head(embedding)
        self.assertEqual(tuple(embedding.shape),(5,32))
        self.assertEqual(tuple(logits.shape),(5,4))
        self.assertTrue(torch.isfinite(logits).all())
        self.assertFalse(encoder.training)
        with self.assertRaises(ValueError): load_interface('outputs/q2_refined/models',expected_features=names[::-1])
        with self.assertRaises(ValueError): scaler.transform(np.zeros((2,26)))

    def test_transfer_summary_has_source_only_gate(self):
        obj=json.loads(Path('outputs/q2_refined/transfer20_source_summary.json').read_text())
        self.assertFalse(obj['target_data_used'])
        self.assertIsNone(obj['target_accuracy'])
        expected=obj['macro_f1_seed_mean']>=.85 and obj['minimum_class_mean_recall']>=.70
        self.assertEqual(obj['eligible_for_q3_initialization'],expected)


if __name__ == '__main__': unittest.main()
