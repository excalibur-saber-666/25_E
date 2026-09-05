from pathlib import Path
import sys
import unittest
import numpy as np
import pandas as pd
import joblib
import json

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import q2_pipeline as q
import q2_analysis as a


class MetricTests(unittest.TestCase):
    def test_window_probability_arithmetic_mean(self):
        frame=pd.DataFrame(dict(file_id=['a','a','b'],label=['N','N','B'],load=[0,0,1],rpm=[1800]*3))
        prob=np.array([[.8,.1,.05,.05],[.2,.4,.2,.2],[.1,.6,.1,.2]])
        out=q.aggregate_window_probabilities(frame,prob).set_index('file_id')
        np.testing.assert_allclose(out.loc['a',a.r.PROBS].to_numpy(float),prob[:2].mean(0))
        self.assertEqual(len(out),2)

    def test_vectorized_bootstrap_matches_sklearn(self):
        y=np.array(q.LABELS*3)
        pred=y.copy(); pred[2]='B';pred[7]='IR'
        indices=np.array([np.arange(12),np.arange(12)[::-1]])
        result=a.resampled_metrics(y,pred,indices)
        expected=q.class_metrics(y,pred)
        np.testing.assert_allclose(result[0],[expected[k] for k in ['macro_f1','balanced_accuracy','accuracy','recall_IR','recall_OR']])
        np.testing.assert_allclose(result[0],result[1])

    def test_temperature_and_probability_metrics(self):
        p=np.eye(4)*.8+.05
        np.testing.assert_allclose(a.temperature_prob(p,1),p)
        for t in [.05,2,20]:
            result=a.temperature_prob(p,t)
            np.testing.assert_allclose(result.sum(1),1)
            np.testing.assert_array_equal(result.argmax(1),p.argmax(1))
        perfect=a.probability_metrics(q.LABELS,np.eye(4))
        self.assertAlmostEqual(perfect['nll'],0)
        self.assertAlmostEqual(perfect['brier'],0)
        self.assertAlmostEqual(perfect['ece'],0)
        t=a.fit_temperature(q.LABELS,p)
        self.assertLessEqual(a.probability_metrics(q.LABELS,a.temperature_prob(p,t))['nll'],a.probability_metrics(q.LABELS,p)['nll'])

    def test_real_checkpoint_reproduces_mean_window_file_probabilities(self):
        path=Path('outputs/q2_refined/runs/Diagnostic26_mlp_2025_load0')
        bundle=joblib.load(path/'bundle.joblib')
        w,f,names=a.r.inputs(Path('outputs/q1'),'diagnostic')
        test=f[f.load==0]
        predicted=a.r.predict(bundle,'mlp',test,w).sort_values('file_id')
        saved=pd.read_csv(path/'predictions.csv').sort_values('file_id')
        np.testing.assert_allclose(predicted[a.r.PROBS],saved[a.r.PROBS],rtol=1e-6,atol=1e-7)

    def test_calibration_decision_is_preserved_without_regenerable_predictions(self):
        # Per-file calibration predictions are deliberately excluded from the
        # frozen repository; the aggregate decision and leakage audit remain.
        p=pd.read_csv('outputs/q2_refined/calibration_metrics.csv')
        self.assertTrue((p.file_leakage_count==0).all())
        self.assertTrue((p.temperature>0).all())
        decision=json.loads(Path('outputs/q2_refined/calibration_decision.json').read_text(encoding='utf-8'))
        self.assertFalse(decision['mlp']['recommend_source_calibration'])


if __name__ == '__main__': unittest.main()
