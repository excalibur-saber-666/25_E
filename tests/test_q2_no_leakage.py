import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import q2_robustness as r


class LeakageTests(unittest.TestCase):
    def test_group_splits_and_rare_normal_class(self):
        for count,expected in [(3,3),(2,2)]:
            f=pd.DataFrame([dict(file_id=f'{label}{i}',label=label) for label in r.q.LABELS for i in range(count)])
            cv=r.splits(f,2025,'unit_test')
            self.assertEqual(len(cv),expected)
            held=[]
            for a,b in cv:
                self.assertFalse(set(f.iloc[a].file_id)&set(f.iloc[b].file_id))
                self.assertEqual(set(f.iloc[b].label),set(r.q.LABELS))
                held.extend(f.iloc[b].file_id)
            self.assertEqual(sorted(held),sorted(f.file_id))

    def test_overlap_rejected(self):
        with self.assertRaises(AssertionError): r.assert_disjoint(['a','b'],['b'],'unit_test')

    def test_training_scaler_and_equal_file_mass(self):
        torch.set_num_threads(1)
        train=pd.DataFrame([dict(file_id=c+str(i),label=c,load=0,rpm=1800,x=float(j))
            for c in r.q.LABELS for i in range(2) for j in range(2+i*5)])
        valid=train.copy()
        valid.file_id='validation_'+valid.file_id
        valid.x=1e9
        weights=r.q.window_weights(train)
        mass=train.assign(weight=weights).groupby(['label','file_id']).weight.sum()
        np.testing.assert_allclose(mass.to_numpy(),np.ones(8)/8)
        _,scaler,_,_=r.train_mlp(train,valid,['x'],dict(dropout=.1,weight_decay=.0001),2025,1,1)
        self.assertAlmostEqual(scaler.mean_[0],np.average(train.x,weights=weights))
        self.assertLess(scaler.mean_[0],100)

    def test_source_loader_does_not_open_target(self):
        original=pd.read_csv
        opened=[]
        def read(path,*a,**kw):
            opened.append(str(path))
            self.assertNotIn('target',str(path))
            return original(path,*a,**kw)
        with patch.object(pd,'read_csv',side_effect=read):
            for mode in ['diagnostic','transfer']:
                w,f,names=r.inputs(Path('outputs/q1'),mode)
                self.assertEqual(len(f),56)
                self.assertEqual(len(w),806)
        self.assertEqual(len(opened),4)

    def test_saved_nested_split_audits(self):
        audit_paths=list(Path('outputs/q2_refined/runs').glob('*/splits.json'))
        audit_paths+=list(Path('outputs/q2_refined/calibration_runs').glob('*/splits.json'))
        self.assertTrue(audit_paths)
        for path in audit_paths:
            context=path.parent.name
            load=int(context.rsplit('load',1)[1])
            forbidden=set(pd.read_csv('outputs/q1/source_metadata.csv').query('load == @load').file_id)
            for record in json.loads(path.read_text()):
                a,b=set(record['train_file_ids']),set(record['validation_file_ids'])
                self.assertFalse(a&b)
                if not record['context'].startswith('calibration_outer/'):
                    self.assertFalse((a|b)&forbidden)


if __name__ == '__main__': unittest.main()
