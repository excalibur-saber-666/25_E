"""20D source-only initialization; no target file is opened.

Standalone: python src/q2_transfer_pretrain.py
"""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import torch

import q2_pipeline as q
import q2_robustness as r


def export_mlp(bundle, prefix, directory, purpose):
    directory.mkdir(parents=True, exist_ok=True)
    model = bundle['model']
    torch.save(model.encoder.state_dict(), directory/f'{prefix}_encoder.pth')
    torch.save(model.classifier.state_dict(), directory/f'{prefix}_classifier.pth')
    joblib.dump(bundle['scaler'], directory/f'{prefix}_scaler.pkl')
    r.json_write(directory/f'{prefix}_feature_names.json', dict(features=bundle['names'], n_features=len(bundle['names'])))
    r.json_write(directory/f'{prefix}_config.json', dict(input_dim=len(bundle['names']), encoder=[128,64,32],
        labels=list(q.LABELS), params=bundle['params'], purpose=purpose, target_data_used=False,
        aggregation='arithmetic mean of window softmax probabilities', calibration='none',
        inference='encoder.eval(); classifier.eval(); scaler.transform in exact feature order'))


def load_interface(directory, prefix='q2_transfer20', expected_features=None):
    """Returns inference-ready encoder, head, scaler, ordered schema; rejects mismatch."""
    directory = Path(directory)
    config = json.loads((directory/f'{prefix}_config.json').read_text(encoding='utf-8'))
    schema = json.loads((directory/f'{prefix}_feature_names.json').read_text(encoding='utf-8'))['features']
    if expected_features is not None and list(expected_features) != schema:
        raise ValueError('Feature names/order mismatch; do not mix Diagnostic26 and Transfer20')
    scaler = joblib.load(directory/f'{prefix}_scaler.pkl')
    assert scaler.n_features_in_ == config['input_dim'] == len(schema)
    encoder = q.FeatureEncoder(len(schema), config['params']['dropout'])
    classifier = q.SourceClassifier()
    encoder.load_state_dict(torch.load(directory/f'{prefix}_encoder.pth', map_location='cpu', weights_only=True))
    classifier.load_state_dict(torch.load(directory/f'{prefix}_classifier.pth', map_location='cpu', weights_only=True))
    return encoder.eval(), classifier.eval(), scaler, schema


def pretrain(q1, output, seeds, fingerprint):
    w, f, names = r.inputs(q1, 'transfer')
    assert len(names) == 20
    results, predictions = r.run_lolo(w, f, names, 'Transfer20', ('logistic_regression', 'mlp'), seeds, output, fingerprint)
    q.save_df(output/'transfer20_lolo_results.csv', results)
    q.save_df(output/'transfer20_oof_predictions.csv', predictions)
    mlp = results[results.model == 'mlp']
    seed_mean = mlp.groupby('seed')[r.METRICS].mean()
    f1 = float(seed_mean.macro_f1.mean())
    minimum = float(seed_mean[[f'recall_{c}' for c in q.LABELS]].mean().min())
    # Fixed reference seed, not the best observed seed. Retune on ALL source files.
    bundle = r.fit_model('mlp', f, w, names, 2025)
    q.save_df(output/'transfer20_refit_history.csv', pd.DataFrame(bundle['history']))
    export_mlp(bundle, 'q2_transfer20', output/'models', 'Question 3 source initialization candidate')
    enc, head, scaler, schema = load_interface(output/'models', expected_features=names)
    sample = scaler.transform(w[names].iloc[:3].to_numpy(float))
    with torch.no_grad():
        embedding = enc(torch.tensor(sample, dtype=torch.float32))
        logits = head(embedding)
    assert embedding.shape == (3,32) and logits.shape == (3,4) and torch.isfinite(logits).all()
    r.json_write(output/'transfer20_source_summary.json', dict(
        purpose='Question 3 source initialization candidate', target_data_used=False, target_accuracy=None,
        eligible_for_q3_initialization=bool(f1 >= .85 and minimum >= .70),
        gate='engineering rule: seed-mean LOLO F1 >= .85 AND minimum class mean recall >= .70',
        macro_f1_seed_mean=f1, macro_f1_seed_std=float(seed_mean.macro_f1.std()),
        balanced_accuracy_seed_mean=float(seed_mean.balanced_accuracy.mean()),
        balanced_accuracy_seed_std=float(seed_mean.balanced_accuracy.std()),
        minimum_class_mean_recall=minimum, per_class_mean_recall=seed_mean[[f'recall_{c}' for c in q.LABELS]].mean().to_dict(),
        seeds=seeds, files=len(f), window_count=len(w), input_dim=20, feature_order=names,
        interface_test=dict(embedding_shape=list(embedding.shape), logits_shape=list(logits.shape), passed=True),
        final_fit_seed=2025, final_fit_params=bundle['params'],
        limitation='Source-only evidence. No target transfer or target calibration established; full refit is not an independent test.'))


if __name__ == '__main__':
    import sys
    sys.argv += ['--stage', 'transfer']
    r.main()
