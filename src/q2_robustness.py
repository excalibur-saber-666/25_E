"""Source-only refinement. Original q2 outputs and q1 inputs are immutable.

Run from repository root: python src/q2_robustness.py
Caches are content-addressed by source inputs, training code and protocol.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.model_selection import StratifiedKFold, GridSearchCV

import q2_pipeline as q

MODELS = ('logistic_regression', 'rbf_svm', 'random_forest', 'mlp')
PROBS = [f'prob_{c}' for c in q.LABELS]
METRICS = ['macro_f1', 'balanced_accuracy', 'accuracy', 'macro_precision',
           'macro_recall', *[f'recall_{c}' for c in q.LABELS]]
SPLIT_AUDIT = []
CANDIDATES = [dict(dropout=.1, weight_decay=1e-4), dict(dropout=.2, weight_decay=1e-3)]


def json_write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str), encoding='utf-8')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def assert_disjoint(train, valid, context):
    a, b = set(train), set(valid)
    assert a and b and not a & b, f'Overlapping or empty file split: {context}'
    SPLIT_AUDIT.append(dict(context=context, train_file_ids=sorted(a),
                            validation_file_ids=sorted(b), file_leakage_count=len(a & b)))


def splits(files, seed, context):
    """Exactly one row per group: stratify groups directly, preserving rare N.

    SGKF's shuffle heuristic can omit N from folds even when three N groups
    are available. StratifiedKFold on unique file rows is a grouped split.
    Calibration's sub-inner fits have two N files, so use two folds there.
    """
    assert files.file_id.is_unique
    n = min(3, int(files.label.value_counts().min()))
    if n < 2:
        raise ValueError('Need two files in each class for nested tuning')
    result = list(StratifiedKFold(n, shuffle=True, random_state=seed).split(files, files.label))
    for i, (a, b) in enumerate(result):
        assert_disjoint(files.iloc[a].file_id, files.iloc[b].file_id, f'{context}/{i}')
        assert set(files.iloc[a].label) == set(q.LABELS)
        assert set(files.iloc[b].label) == set(q.LABELS)
    return result


def inputs(q1, feature_set):
    names = json.loads((q1 / f'feature_names_{feature_set}.json').read_text(encoding='utf-8'))['features']
    w = pd.read_csv(q1 / f'features_source_{feature_set}.csv')
    f = pd.read_csv(q1 / 'source_metadata.csv', dtype={'fault_size': str, 'fault_position': str})
    f = f[['file_id', 'label', 'load', 'rpm', 'fault_size', 'fault_position']].copy()
    assert f.file_id.is_unique and len(f) == 56 and set(w.file_id) == set(f.file_id)
    assert set(f.label) == set(q.LABELS)
    assert np.isfinite(w[names].to_numpy(float)).all()
    assert not w.duplicated(['file_id', 'window_id']).any()
    assert w.groupby('file_id').label.nunique().eq(1).all()
    assert w.groupby('file_id').label.first().sort_index().equals(f.set_index('file_id').label.sort_index())
    w = w.merge(f[['file_id', 'load']], on='file_id', validate='many_to_one')
    return w, f, names


def aggregate(w, f, names):
    means = w.groupby('file_id')[names].mean().add_suffix('_mean')
    std = w.groupby('file_id')[names].std(ddof=0).add_suffix('_std')
    cols = [f'{n}_{s}' for n in names for s in ('mean', 'std')]
    return f.set_index('file_id').join(means).join(std).reset_index(), cols


def train_mlp(train, valid, names, params, seed, epochs=100, patience=15):
    random.seed(seed)
    if valid is not None:
        assert_disjoint(train.file_id, valid.file_id, f'early_stop/{seed}')
    return q.train_mlp(train, valid, names, params, seed, epochs, patience)


def tune_mlp(f, w, names, seed):
    cv = splits(f, seed, f'mlp_tune/{seed}')
    choices, histories = [], []
    for candidate, params in enumerate(CANDIDATES):
        scores, epochs = [], []
        for fold, (a, b) in enumerate(cv):
            train = w[w.file_id.isin(f.iloc[a].file_id)]
            valid = w[w.file_id.isin(f.iloc[b].file_id)]
            _, _, history, epoch = train_mlp(train, valid, names, params, seed + fold, 100, 15)
            best = max(history, key=lambda r: (r['val_macro_f1'], r['val_balanced_accuracy']))
            scores.append((best['val_macro_f1'], best['val_balanced_accuracy']))
            epochs.append(epoch)
            histories.extend(dict(candidate=candidate, inner_fold=fold, **r) for r in history)
        choices.append((np.mean(scores, axis=0).tolist(), dict(**params, epochs=int(np.median(epochs)))))
    return max(choices, key=lambda r: r[0])[1], histories


def fit_model(model_name, train_files, w, names, seed):
    """Only train_files enter scaler, hyperparameter selection and early stopping."""
    w = w[w.file_id.isin(train_files.file_id)]
    random.seed(seed)
    if model_name == 'mlp':
        params, history = tune_mlp(train_files, w, names, seed)
        model, scaler, fit_history, _ = train_mlp(w, None, names, params, seed + 500, params['epochs'], params['epochs'])
        history = [dict(stage='inner_tuning', **r) for r in history] + [dict(stage='refit', **r) for r in fit_history]
        return dict(model=model, scaler=scaler, params=params, names=names, history=history)
    table, cols = aggregate(w, train_files, names)
    model, grid = q.build_sklearn_model(model_name)
    model.set_params(model__random_state=seed)
    cv = splits(train_files, seed, f'{model_name}_tune/{seed}')
    # SVC.predict can disagree with argmax(predict_proba); tune the exact rule
    # used for file-level evaluation rather than a different decision function.
    search = GridSearchCV(model, grid, cv=cv, scoring=probability_f1, n_jobs=1)
    search.fit(table[cols].to_numpy(float), table.label)
    return dict(model=search.best_estimator_, params=search.best_params_, names=names, cols=cols, history=[])


def probability_f1(estimator, x, y):
    probabilities = q.align_probabilities(estimator.predict_proba(x), estimator.named_steps['model'].classes_)
    return q.class_metrics(np.asarray(y), np.array(q.LABELS)[probabilities.argmax(1)])['macro_f1']


def predict(bundle, model_name, files, w):
    w = w[w.file_id.isin(files.file_id)]
    if model_name == 'mlp':
        x = bundle['scaler'].transform(w[bundle['names']].to_numpy(float))
        pred = q.aggregate_window_probabilities(w, q.mlp_probabilities(bundle['model'], x))
    else:
        table, cols = aggregate(w, files, bundle['names'])
        prob = q.align_probabilities(bundle['model'].predict_proba(table[cols].to_numpy(float)),
                                     bundle['model'].named_steps['model'].classes_)
        pred = table[['file_id', 'label', 'load', 'rpm']].copy()
        pred[PROBS] = prob
        pred['predicted_label'] = np.array(q.LABELS)[prob.argmax(1)]
        pred['confidence'] = prob.max(1)
    pred = pred.merge(files[['file_id', 'fault_size', 'fault_position']], on='file_id', validate='one_to_one')
    return pred.rename(columns={'label': 'true_label', 'load': 'held_out_load'})


def run_lolo(w, f, names, group, models, seeds, output, fingerprint):
    rows, predictions = [], []
    for seed in seeds:
        for load in range(4):
            train, test = f[f.load != load], f[f.load == load]
            assert_disjoint(train.file_id, test.file_id, f'{group}/{seed}/load{load}')
            for model in models:
                path = output / 'runs' / f'{group}_{model}_{seed}_load{load}'
                path.mkdir(parents=True, exist_ok=True)
                stamp = path / 'complete.json'
                if stamp.exists() and json.loads(stamp.read_text())['fingerprint'] == fingerprint:
                    rows.append(json.loads((path/'metrics.json').read_text()))
                    predictions.append(pd.read_csv(path/'predictions.csv', dtype={'fault_size': str, 'fault_position': str}))
                    SPLIT_AUDIT.extend(json.loads((path/'splits.json').read_text()))
                    continue
                audit_start = len(SPLIT_AUDIT)
                started = time.perf_counter()
                bundle = fit_model(model, train, w, names, seed + 100 * (load + 1))
                fit_seconds = time.perf_counter() - started
                started = time.perf_counter()
                p = predict(bundle, model, test, w)
                infer_seconds = time.perf_counter() - started
                p['seed'], p['model'], p['feature_group'] = seed, model, group
                metric = dict(seed=seed, model=model, feature_group=group, held_out_load=load,
                              n_train_files=len(train), n_test_files=len(test), file_leakage_count=0,
                              training_seconds=fit_seconds, inference_seconds=infer_seconds,
                              best_params=json.dumps(bundle['params']),
                              **q.class_metrics(p.true_label.to_numpy(), p.predicted_label.to_numpy()))
                q.save_df(path/'predictions.csv', p)
                q.save_df(path/'history.csv', pd.DataFrame(bundle.pop('history')))
                joblib.dump(bundle, path/'bundle.joblib')
                json_write(path/'metrics.json', metric)
                json_write(path/'splits.json', SPLIT_AUDIT[audit_start:])
                json_write(stamp, dict(fingerprint=fingerprint))
                rows.append(metric)
                predictions.append(p)
                print(f'{group} {model} seed={seed} load={load} F1={metric["macro_f1"]:.3f}', flush=True)
    return pd.DataFrame(rows), pd.concat(predictions, ignore_index=True)


def baseline_report(q1, output):
    original = pd.read_csv('outputs/q2/predictions_file_level.csv')
    rerun = pd.read_csv('outputs/q2_recheck_baseline/predictions_file_level.csv')
    keys = ['validation_scheme', 'fold', 'model', 'file_id']
    a, b = original.sort_values(keys).reset_index(drop=True), rerun.sort_values(keys).reset_index(drop=True)
    assert a[keys].equals(b[keys]) and a.predicted_label.equals(b.predicted_label)
    assert np.allclose(a[PROBS], b[PROBS], rtol=0, atol=1e-10)
    result = pd.read_csv('outputs/q2_recheck_baseline/lolo_results.csv')
    w, f, names = inputs(q1, 'diagnostic')
    assert result.file_leakage_count.max() == 0
    mlp = b[(b.model == 'mlp') & (b.validation_scheme == 'lolo')]
    json_write(output/'baseline_reproduction.json', dict(passed=True, files=len(f), windows=len(w),
        features=len(names), max_probability_difference=float(np.abs(a[PROBS]-b[PROBS]).to_numpy().max()),
        lolo_mlp=result[result.model == 'mlp'].to_dict('records'),
        pooled_confusion=q.confusion_matrix(mlp.true_label, mlp.predicted_label, labels=q.LABELS).tolist(),
        wrong_files=mlp[mlp.true_label != mlp.predicted_label].to_dict('records')))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--q1-dir', type=Path, default=Path('outputs/q1'))
    parser.add_argument('--output-dir', type=Path, default=Path('outputs/q2_refined'))
    parser.add_argument('--seeds', type=int, nargs='+', default=list(range(2025, 2030)))
    parser.add_argument('--stage', choices=['all', 'diagnostic', 'calibration', 'ablation', 'transfer', 'analysis'], default='all')
    args = parser.parse_args()
    out = args.output_dir
    assert out.resolve() not in [Path('outputs/q1').resolve(), Path('outputs/q2').resolve()]
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    source_paths = [args.q1_dir/n for n in ['features_source_diagnostic.csv', 'features_source_transfer.csv',
                    'feature_names_diagnostic.json', 'feature_names_transfer.json', 'source_metadata.csv']]
    hashes = {str(p): sha(p) for p in source_paths + [Path(__file__), Path(q.__file__)]}
    fingerprint = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    config = dict(seeds=args.seeds, hashes=hashes, fingerprint=fingerprint, target_data_used=False,
        backend='python', bootstrap_repetitions=2000, ece_bins=10, torch_threads=1,
        python=platform.python_version(), numpy=np.__version__, sklearn=sklearn.__version__, torch=torch.__version__,
        mlp_candidates=CANDIDATES, max_epochs=100, patience=15, random_state='seed + 100 * (load+1)',
        inner_split='StratifiedKFold on unique MAT file rows; 3 folds, 2 for calibration sub-inner N=2',
        sklearn_selection='Macro-F1 of argmax aligned predict_proba, consistent with final evaluation',
        sampler='1/(4 * files_in_class * windows_in_file)', window=16384, hop=8192,
        calibration='softmax(log(mean_window_softmax)/T), T fitted on nested cross-fitted outer-training files',
        bootstrap_scope='class-stratified paired file resampling; predictions fixed; seed estimates averaged, not stacked',
        feature_groups={'Diagnostic26':'26 fixed diagnostic features', 'Transfer20':'ordered transfer schema',
                        'NoAmplitude':'remove amp_rms, amp_peak, env_rms', 'NoOrder':'remove order_*', 'NoEnvelope':'remove env_*'})
    json_write(out/'q2_refined_config.json', config)
    baseline_report(args.q1_dir, out)
    w, f, names = inputs(args.q1_dir, 'diagnostic')
    if args.stage in ('all', 'diagnostic'):
        r, p = run_lolo(w, f, names, 'Diagnostic26', MODELS, args.seeds, out, fingerprint)
        q.save_df(out/'multiseed_lolo_results.csv', r)
        q.save_df(out/'multiseed_oof_predictions.csv', p)
    if args.stage in ('all', 'calibration'):
        from q2_analysis import calibration
        calibration(w, f, names, args.seeds, out, fingerprint)
    if args.stage in ('all', 'ablation'):
        ablations = {'NoAmplitude':[n for n in names if n not in ('amp_rms','amp_peak','env_rms')],
                     'NoOrder':[n for n in names if not n.startswith('order_')],
                     'NoEnvelope':[n for n in names if not n.startswith('env_')]}
        results = []
        for group, subset in ablations.items():
            r, _ = run_lolo(w, f, subset, group, ('logistic_regression', 'mlp'), args.seeds, out, fingerprint)
            results.append(r)
        q.save_df(out/'diagnostic_ablation_folds.csv', pd.concat(results))
    if args.stage in ('all', 'transfer'):
        from q2_transfer_pretrain import pretrain
        pretrain(args.q1_dir, out, args.seeds, fingerprint)
    if args.stage in ('all', 'analysis'):
        from q2_analysis import analyze
        analyze(args.q1_dir, out, args.seeds)
    json_write(out/f'split_audit_{args.stage}.json', SPLIT_AUDIT)
    assert all(sha(p) == hashes[str(p)] for p in source_paths)


if __name__ == '__main__':
    # Keep checkpoint class/module paths stable for CLI and later imports.
    import sys
    sys.modules.setdefault('q2_robustness', sys.modules[__name__])
    main()
