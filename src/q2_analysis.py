"""File-based statistics, nested probability calibration and evidence figures."""
import json
import hashlib
import inspect
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import q2_pipeline as q
import q2_robustness as r


def save_figure(fig, base):
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(base.with_suffix('.png'), dpi=400, bbox_inches='tight')
    fig.savefig(base.with_suffix('.svg'), bbox_inches='tight')
    fig.savefig(base.with_suffix('.pdf'), bbox_inches='tight')
    plt.close(fig)


def temperature_prob(prob, temperature):
    z = np.log(np.clip(np.asarray(prob), 1e-12, 1)) / temperature
    z -= z.max(axis=1, keepdims=True)
    exp = np.exp(z)
    return exp / exp.sum(axis=1, keepdims=True)


def probability_metrics(y, prob, bins=10):
    y = np.array([q.LABELS.index(str(v)) for v in y])
    prob = np.asarray(prob)
    confidence = prob.max(1)
    correct = prob.argmax(1) == y
    ids = np.minimum((confidence*bins).astype(int), bins-1)
    ece = sum(np.mean(ids == i)*abs(correct[ids == i].mean()-confidence[ids == i].mean())
              for i in range(bins) if np.any(ids == i))
    return dict(nll=float(-np.log(np.clip(prob[np.arange(len(y)), y], 1e-12, 1)).mean()),
                brier=float(np.square(prob-np.eye(4)[y]).sum(1).mean()), ece=float(ece))


def fit_temperature(y, prob):
    grid = np.exp(np.linspace(np.log(.05), np.log(20), 241))
    losses = [probability_metrics(y, temperature_prob(prob,t))['nll'] for t in grid]
    return float(grid[np.argmin(losses)])


def calibration(w, f, names, seeds, output, fingerprint):
    fingerprint=hashlib.sha256((fingerprint+inspect.getsource(calibration)+
        inspect.getsource(fit_temperature)+inspect.getsource(temperature_prob)+
        inspect.getsource(probability_metrics)).encode()).hexdigest()
    results = pd.read_csv(output/'multiseed_lolo_results.csv')
    leader = results.groupby('model').macro_f1.mean().idxmax()
    models = sorted(set([leader, 'mlp']))
    metric_rows, prediction_rows, oof_rows = [], [], []
    for seed in seeds:
        for load in range(4):
            train, test = f[f.load != load], f[f.load == load]
            for model in models:
                path = output/'calibration_runs'/f'{model}_{seed}_load{load}'
                path.mkdir(parents=True, exist_ok=True)
                stamp = path/'complete.json'
                if stamp.exists() and json.loads(stamp.read_text())['fingerprint'] == fingerprint:
                    prediction_rows.append(pd.read_csv(path/'predictions.csv'))
                    metric_rows.extend(json.loads((path/'metrics.json').read_text()))
                    oof_rows.append(pd.read_csv(path/'training_oof.csv'))
                    r.SPLIT_AUDIT.extend(json.loads((path/'splits.json').read_text()))
                    continue
                start = len(r.SPLIT_AUDIT)
                r.assert_disjoint(train.file_id, test.file_id, f'calibration_outer/{model}/{seed}/{load}')
                oof = []
                for fold, (a, b) in enumerate(r.splits(train, seed+100*(load+1), 'calibration_crossfit')):
                    inner_train, inner_test = train.iloc[a], train.iloc[b]
                    # All hyperparameters/epochs fitted without inner_test labels.
                    bundle = r.fit_model(model, inner_train, w, names, seed+2000+load*100+fold)
                    p = r.predict(bundle, model, inner_test, w)
                    p['calibration_fold'] = fold
                    oof.append(p)
                oof = pd.concat(oof, ignore_index=True)
                assert oof.file_id.is_unique and set(oof.file_id) == set(train.file_id)
                temperature = fit_temperature(oof.true_label, oof[r.PROBS])
                outer_path = output/'runs'/f'Diagnostic26_{model}_{seed}_load{load}'
                outer = pd.read_csv(outer_path/'predictions.csv')
                assert set(outer.file_id) == set(test.file_id)
                after = temperature_prob(outer[r.PROBS], temperature)
                output_rows, metrics = [], []
                for stage, prob in [('before', outer[r.PROBS].to_numpy()), ('after', after)]:
                    p = outer.copy()
                    p[r.PROBS] = prob
                    p['confidence'] = prob.max(1)
                    p['predicted_label'] = np.array(q.LABELS)[prob.argmax(1)]
                    p['calibration'], p['temperature'] = stage, temperature
                    output_rows.append(p)
                    metrics.append(dict(seed=seed, held_out_load=load, model=model, calibration=stage,
                        temperature=temperature, n_files=len(p), file_leakage_count=0,
                        **probability_metrics(p.true_label, prob)))
                pred = pd.concat(output_rows, ignore_index=True)
                oof['outer_load'], oof['seed'], oof['model'] = load, seed, model
                q.save_df(path/'predictions.csv', pred)
                q.save_df(path/'training_oof.csv', oof)
                r.json_write(path/'metrics.json', metrics)
                r.json_write(path/'splits.json', r.SPLIT_AUDIT[start:])
                r.json_write(stamp, dict(fingerprint=fingerprint))
                prediction_rows.append(pred); metric_rows.extend(metrics); oof_rows.append(oof)
                print(f'calibration {model} seed={seed} load={load} T={temperature:.3f}', flush=True)
    q.save_df(output/'calibration_metrics.csv', pd.DataFrame(metric_rows))
    q.save_df(output/'calibrated_oof_predictions.csv', pd.concat(prediction_rows, ignore_index=True))
    q.save_df(output/'calibration_training_oof_predictions.csv', pd.concat(oof_rows, ignore_index=True))


def resampled_metrics(y, pred, indices):
    """Vectorized confusion metrics for file-resampled predictions."""
    y = np.array([q.LABELS.index(str(v)) for v in y])
    pred = np.array([q.LABELS.index(str(v)) for v in pred])
    code = (y[indices]*4+pred[indices])
    counts = (code[:,:,None] == np.arange(16)[None,None,:]).sum(axis=1).reshape(-1,4,4)
    tp = counts.diagonal(axis1=1,axis2=2)
    support = counts.sum(2)
    predicted = counts.sum(1)
    recalls = np.divide(tp,support,out=np.zeros_like(tp,dtype=float),where=support>0)
    f1 = np.divide(2*tp,support+predicted,out=np.zeros_like(tp,dtype=float),where=(support+predicted)>0)
    return np.column_stack([f1.mean(1), recalls.mean(1), tp.sum(1)/counts.sum((1,2)), recalls[:,2], recalls[:,3]])


def bootstrap(predictions, output, repetitions=2000):
    keys = sorted(predictions.file_id.unique())
    seeds = sorted(predictions.seed.unique())
    reference = predictions[(predictions.seed == seeds[0]) & (predictions.model == 'mlp')].set_index('file_id').loc[keys]
    y = reference.true_label.to_numpy()
    rng = np.random.default_rng(2025)
    indices = np.concatenate([rng.choice(np.flatnonzero(y == c), size=(repetitions,(y == c).sum()), replace=True)
                              for c in q.LABELS],axis=1)
    metric_names = ['macro_f1','balanced_accuracy','accuracy','recall_IR','recall_OR']
    draws, estimates, rows = {}, {}, []
    for model in sorted(predictions.model.unique()):
        model_draws, model_est = [], []
        for seed in seeds:
            p = predictions[(predictions.seed == seed)&(predictions.model == model)].set_index('file_id').loc[keys]
            assert len(p) == 56 and p.index.is_unique and np.array_equal(p.true_label,y)
            dist = resampled_metrics(y,p.predicted_label.to_numpy(),indices)
            est = resampled_metrics(y,p.predicted_label.to_numpy(),np.arange(56)[None,:])[0]
            model_draws.append(dist); model_est.append(est)
            for j, name in enumerate(metric_names):
                lo, hi = np.quantile(dist[:,j],[.025,.975])
                rows.append(dict(model=model,seed=seed,metric=name,estimate=est[j],ci_low=lo,ci_high=hi,n_files=56))
        draws[model], estimates[model] = np.mean(model_draws,axis=0), np.mean(model_est,axis=0)
        for j, name in enumerate(metric_names):
            lo,hi = np.quantile(draws[model][:,j],[.025,.975])
            rows.append(dict(model=model,seed='mean_of_seeds',metric=name,estimate=estimates[model][j],ci_low=lo,ci_high=hi,n_files=56))
    q.save_df(output/'bootstrap_confidence_intervals.csv',pd.DataFrame(rows))
    paired = []
    for other in ['logistic_regression','random_forest']:
        for j,name in enumerate(metric_names[:2]):
            lo,hi = np.quantile(draws['mlp'][:,j]-draws[other][:,j],[.025,.975])
            paired.append(dict(comparison=f'mlp - {other}',metric=name,
                estimate=estimates['mlp'][j]-estimates[other][j],ci_low=lo,ci_high=hi,
                includes_zero=bool(lo<=0<=hi),n_files=56,seeds=len(seeds)))
    q.save_df(output/'paired_model_comparison.csv',pd.DataFrame(paired))


def error_analysis(q1, output, predictions):
    w,f,names = r.inputs(q1,'diagnostic')
    for factor in ['label','fault_size','fault_position']:
        counts=pd.crosstab(f[factor].fillna('not_applicable'),f.load).reset_index()
        q.save_df(output/f'metadata_load_by_{factor}.csv',counts)
    rows = []
    for (model,file_id), p in predictions.groupby(['model','file_id']):
        wrong = p[p.true_label != p.predicted_label]
        first = p.iloc[0]
        rows.append(dict(model=model,file_id=file_id,true_label=first.true_label,n_wrong_seeds=len(wrong),
            n_seeds=p.seed.nunique(),most_common_wrong_label=wrong.predicted_label.mode().iloc[0] if len(wrong) else '',
            mean_true_class_probability=p[f'prob_{first.true_label}'].mean(),mean_predicted_confidence=p.confidence.mean(),
            fault_size=first.fault_size,fault_position=first.fault_position,load=first.held_out_load,rpm=first.rpm))
    persistent = pd.DataFrame(rows).sort_values(['n_wrong_seeds','model','file_id'],ascending=[False,True,True])
    q.save_df(output/'persistent_misclassifications.csv',persistent)
    rates=[]
    data=predictions.assign(wrong=predictions.true_label != predictions.predicted_label)
    for factor in ['held_out_load','true_label','fault_size','fault_position']:
        for (model,level),g in data.groupby(['model',factor],dropna=False):
            rates.append(dict(model=model,factor=factor,level=level,n_unique_files=g.file_id.nunique(),
                seed_average_error_rate=g.groupby('seed').wrong.mean().mean()))
    q.save_df(output/'error_factor_summary.csv',pd.DataFrame(rates))
    table,cols=r.aggregate(w,f,names)
    distances, shifts = [], []
    for load in range(4):
        train=table[table.load != load]; test=table[table.load == load]
        scaler=StandardScaler().fit(train[cols])
        xtr=scaler.transform(train[cols]); xte=scaler.transform(test[cols])
        centers=np.stack([xtr[train.label.to_numpy()==c].mean(0) for c in q.LABELS])
        dist=np.linalg.norm(xte[:,None,:]-centers[None,:,:],axis=2)
        for i,(_,row) in enumerate(test.iterrows()):
            other=[k for k,c in enumerate(q.LABELS) if c != row.label]
            own=q.LABELS.index(row.label)
            distances.append(dict(file_id=row.file_id,true_label=row.label,held_out_load=load,
                nearest_class=q.LABELS[dist[i].argmin()],true_class_distance=dist[i,own],
                nearest_other_distance=dist[i,other].min(),distance_margin=dist[i,other].min()-dist[i,own],
                **{f'distance_{c}':dist[i,k] for k,c in enumerate(q.LABELS)}))
        for c in q.LABELS:
            for n in names:
                col=f'{n}_mean'
                a=train[train.label==c][col]; b=test[test.label==c][col]
                sd=train[col].std(ddof=0)
                shifts.append(dict(held_out_load=load,label=c,feature=n,n_train_files=len(a),n_test_files=len(b),
                    train_class_mean=a.mean(),test_class_mean=b.mean(),train_overall_sd=sd,
                    standardized_shift=(b.mean()-a.mean())/sd if sd>0 else 0))
    q.save_df(output/'file_class_distances.csv',pd.DataFrame(distances))
    q.save_df(output/'class_conditional_feature_shift.csv',pd.DataFrame(shifts))
    # Explanatory only: fit scaler/PCA on non-0 hp files, project all 56.
    train=table[table.load != 0]
    scaler=StandardScaler().fit(train[cols])
    pca=PCA(2).fit(scaler.transform(train[cols]))
    xy=pca.transform(scaler.transform(table[cols]))
    projection=table[['file_id','label','load']].copy()
    projection[['pc1','pc2']]=xy
    q.save_df(output/'error_pca_projection.csv',projection)
    r.json_write(output/'error_pca_config.json',dict(fit_loads=[1,2,3],purpose='post-hoc explanation only',
        explained_variance_ratio=pca.explained_variance_ratio_.tolist()))


def calibration_summary(output):
    p=pd.read_csv(output/'calibrated_oof_predictions.csv')
    rows=[]
    for (model,seed,stage),g in p.groupby(['model','seed','calibration']):
        rows.append(dict(model=model,seed=seed,calibration=stage,**probability_metrics(g.true_label,g[r.PROBS])))
    scores=pd.DataFrame(rows)
    q.save_df(output/'calibration_pooled_by_seed.csv',scores)
    decisions={}
    for model,g in scores.groupby('model'):
        before=g[g.calibration=='before'].set_index('seed')
        after=g[g.calibration=='after'].set_index('seed')
        delta=after[['nll','ece','brier']]-before[['nll','ece','brier']]
        stable=bool((delta[['nll','ece']].mean()<0).all() and ((delta.nll<0)&(delta.ece<0)).sum()>=4)
        decisions[model]=dict(recommend_source_calibration=stable,mean_delta=delta.mean().to_dict(),
            n_seeds_both_nll_ece_improved=int(((delta.nll<0)&(delta.ece<0)).sum()),
            target_calibration_validated=False)
    r.json_write(output/'calibration_decision.json',decisions)
    return decisions


def analyze(q1, output, seeds):
    config=json.loads((output/'q2_refined_config.json').read_text(encoding='utf-8'))
    config['analysis_source_sha256']=r.sha(__file__)
    config['transfer_source_sha256']=r.sha(Path(__file__).with_name('q2_transfer_pretrain.py'))
    r.json_write(output/'q2_refined_config.json',config)
    results=pd.read_csv(output/'multiseed_lolo_results.csv')
    pred=pd.read_csv(output/'multiseed_oof_predictions.csv',dtype={'fault_size':str,'fault_position':str})
    assert set(results.seed)==set(seeds) and len(results)==len(seeds)*4*4
    assert pred.groupby(['model','seed']).file_id.nunique().eq(56).all()
    for _,v in results.iterrows():
        stamp=output/'runs'/f'Diagnostic26_{v.model}_{v.seed}_load{v.held_out_load}'/'complete.json'
        assert json.loads(stamp.read_text())['fingerprint']==config['fingerprint'], 'Rerun stale diagnostic stage'
    seed_scores=results.groupby(['model','seed'])[r.METRICS].mean().reset_index()
    q.save_df(output/'seed_level_lolo_means.csv',seed_scores)
    summary=seed_scores.groupby('model')[r.METRICS].agg(['mean','std','min','max'])
    summary.columns=['_'.join(c) for c in summary.columns]
    summary=summary.reset_index().sort_values('macro_f1_mean',ascending=False)
    q.save_df(output/'multiseed_model_summary.csv',summary)
    loads=results.groupby(['model','held_out_load'])[r.METRICS].agg(['mean','std','min','max'])
    loads.columns=['_'.join(c) for c in loads.columns]
    q.save_df(output/'per_load_multiseed_metrics.csv',loads.reset_index())
    bootstrap(pred,output)
    error_analysis(q1,output,pred)
    decision=calibration_summary(output)
    groups=pd.concat([results[results.model.isin(['mlp','logistic_regression'])],
        pd.read_csv(output/'diagnostic_ablation_folds.csv'),pd.read_csv(output/'transfer20_lolo_results.csv')])
    assert len(groups)==5*2*4*len(seeds)
    for _,v in groups.iterrows():
        stamp=output/'runs'/f'{v.feature_group}_{v.model}_{v.seed}_load{v.held_out_load}'/'complete.json'
        assert json.loads(stamp.read_text())['fingerprint']==config['fingerprint'], 'Rerun stale feature-group stage'
    q.save_df(output/'feature_group_ablation_folds.csv',groups)
    group_seed=groups.groupby(['feature_group','model','seed'])[r.METRICS].mean().reset_index()
    ab=group_seed.groupby(['feature_group','model'])[r.METRICS].agg(['mean','std'])
    ab.columns=['_'.join(c) for c in ab.columns]
    ab=ab.reset_index()
    ab['min_class_mean_recall']=ab[[f'recall_{c}_mean' for c in q.LABELS]].min(axis=1)
    zero=groups[groups.held_out_load==0].groupby(['feature_group','model']).macro_f1.mean()
    ab['load0_macro_f1_mean']=[zero.loc[(g,m)] for g,m in zip(ab.feature_group,ab.model)]
    q.save_df(output/'feature_group_ablation.csv',ab)
    leader=str(summary.iloc[0].model)
    r.json_write(output/'best_model_summary.json',dict(recommended_model=leader,
        selection='highest mean over five seed-level LOLO fold means; no significance claim',
        multiseed_summary=summary.to_dict('records'),target_data_used=False,
        diagnostic_feature_count=26,reference_baseline='outputs/q2',
        group_cv='Only original single-seed protocol reproduced; not rerun under refined protocol'))
    reports=[]
    for (model,seed),p in pred.groupby(['model','seed']):
        report_table=pd.DataFrame(q.classification_report(p.true_label,p.predicted_label,
            labels=q.LABELS,output_dict=True,zero_division=0)).T.reset_index().rename(columns={'index':'class_or_average'})
        report_table['model'],report_table['seed']=model,seed
        reports.append(report_table)
    q.save_df(output/'classification_reports_by_seed.csv',pd.concat(reports,ignore_index=True))
    w,f,names=r.inputs(q1,'diagnostic')
    # Save strongest traditional pipeline even if MLP leads; fixed refit seed.
    traditional=summary[summary.model!='mlp'].iloc[0].model
    bundle=r.fit_model(traditional,f,w,names,2025)
    joblib.dump(bundle['model'],output/'models'/f'q2_diagnostic26_{traditional}.joblib')
    r.json_write(output/'models'/'q2_diagnostic26_traditional_schema.json',dict(
        model=traditional,features=bundle['cols'],params=bundle['params'],labels=list(q.LABELS),
        input='file mean+std, ddof=0',purpose='all-source refit; no independent test score'))
    from q2_transfer_pretrain import export_mlp
    bundle=r.fit_model('mlp',f,w,names,2025)
    export_mlp(bundle,'q2_diagnostic26',output/'models','Question 2 Diagnostic26 all-source refit')
    draw_figures(output,summary,pred,ab,leader)
    report(output,summary,seed_scores,leader,decision)


def draw_figures(output,summary,pred,ab,leader):
    """Quantitative comparison; all five seeds shown, SD never a confidence interval.

    Python style-only inheritance of q1/q2 fonts/export; no inherited statistics.
    183 mm wide at export, 7 pt editable text, PNG/PDF/SVG, no data exclusions.
    """
    plt.rcParams.update({'font.size':7,'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
                         'svg.fonttype':'none','pdf.fonttype':42})
    figures=output/'figures'
    scores=pd.read_csv(output/'seed_level_lolo_means.csv')
    fig,axes=plt.subplots(1,2,figsize=(7.2,3))
    for ax,metric in zip(axes,['macro_f1','balanced_accuracy']):
        for i,model in enumerate(summary.model):
            v=scores[scores.model==model][metric].to_numpy()
            ax.scatter(i+np.linspace(-.1,.1,len(v)),v,s=16,color='#32688E')
            ax.plot([i-.18,i+.18],[v.mean()]*2,color='#303030')
        ax.set(xticks=range(len(summary)),xticklabels=[m.replace('_','\n') for m in summary.model],
               ylabel=metric,ylim=(0,1.03),title='5 seeds; each point = mean of 4 LOLO folds')
    save_figure(fig,figures/'model_multiseed_performance')
    folds=pd.read_csv(output/'multiseed_lolo_results.csv')
    fig,ax=plt.subplots(figsize=(7.2,3))
    for (model,g),marker in zip(folds.groupby('model'),['o','s','^','D']):
        stat=g.groupby('held_out_load').macro_f1.agg(['mean','std'])
        ax.errorbar(stat.index,stat['mean'],yerr=stat['std'],marker=marker,capsize=3,label=model)
    ax.set(xlabel='Held-out load (hp)',ylabel='Macro-F1',xticks=range(4),ylim=(0,1.1),title='Mean ± SD over 5 seeds; 14 test files per load')
    ax.legend(fontsize=6); save_figure(fig,figures/'per_load_performance')
    matrix=sum(q.confusion_matrix(g.true_label,g.predicted_label,labels=q.LABELS) for _,g in pred[pred.model==leader].groupby('seed'))/5
    q.save_df(output/'best_model_confusion_mean_counts.csv',pd.DataFrame(matrix,columns=q.LABELS).assign(true_label=q.LABELS))
    fig,ax=plt.subplots(figsize=(3.5,3))
    normalized=matrix/matrix.sum(1,keepdims=True)
    ax.imshow(normalized,vmin=0,vmax=1,cmap='Blues')
    for i in range(4):
        for j in range(4): ax.text(j,i,f'{matrix[i,j]:.1f}',ha='center',va='center',color='white' if normalized[i,j]>.55 else 'black')
    ax.set(xticks=range(4),xticklabels=q.LABELS,yticks=range(4),yticklabels=q.LABELS,
           xlabel='Predicted',ylabel='True',title=f'{leader}: mean file counts over seeds')
    save_figure(fig,figures/'best_model_confusion_matrix')
    shift=pd.read_csv(output/'class_conditional_feature_shift.csv')
    ir=shift[(shift.held_out_load==0)&(shift.label=='IR')].copy()
    ir=ir.reindex(ir.standardized_shift.abs().sort_values(ascending=False).index)
    fig,ax=plt.subplots(figsize=(7.2,4.8))
    ax.barh(ir.feature,ir.standardized_shift,color='#32688E'); ax.invert_yaxis()
    ax.set(xlabel='(0 hp IR mean - training IR mean) / training overall SD',
           title='All 26 features; file means; 3 held-out IR files versus 9 training IR files')
    save_figure(fig,figures/'0hp_key_feature_shift')
    errors=pd.read_csv(output/'persistent_misclassifications.csv')
    err=errors[(errors.model==leader)&(errors.n_wrong_seeds>0)]
    fig,ax=plt.subplots(figsize=(7.2,max(2.5,.22*len(err)+1)))
    ax.barh(err.file_id,err.n_wrong_seeds,color='#A75A49')
    ax.set(xlabel='Number of incorrect seeds (of 5)',xlim=(0,5.5),title=f'{leader}: all files wrong in at least one seed')
    save_figure(fig,figures/'persistent_error_cases')
    xy=pd.read_csv(output/'error_pca_projection.csv')
    fig,ax=plt.subplots(figsize=(7.2,3.6))
    for (c,g),marker in zip(xy.groupby('label'),['s','^','o','D']):
        ax.scatter(g.pc1,g.pc2,s=18,label=c,marker=marker)
    focus=xy[xy.file_id.isin(['IR014_0','IR021_0','OR014@6_0'])]
    for _,v in focus.iterrows(): ax.annotate(v.file_id,(v.pc1,v.pc2),xytext=(4,6),textcoords='offset points',fontsize=6)
    ax.set(xlabel='PC1',ylabel='PC2',title='Post-hoc projection: fit loads 1–3, project all 56 files'); ax.legend()
    save_figure(fig,figures/'error_cases_feature_space')
    fig,axes=plt.subplots(1,2,figsize=(7.2,3.5))
    for ax,(model,g) in zip(axes,ab.groupby('model')):
        ax.errorbar(range(len(g)),g.macro_f1_mean,yerr=g.macro_f1_std,fmt='o',capsize=3,color='#32688E')
        ax.set(xticks=range(len(g)),xticklabels=g.feature_group,ylabel='LOLO Macro-F1',ylim=(0,1.05),title=model)
        ax.tick_params(axis='x',rotation=40)
    save_figure(fig,figures/'feature_group_ablation')
    calibrated=pd.read_csv(output/'calibrated_oof_predictions.csv')
    bins=[]
    models=sorted(calibrated.model.unique())
    fig,axes=plt.subplots(1,len(models),figsize=(7.2,3.3),squeeze=False)
    for ax,model in zip(axes[0],models):
        for stage in ['before','after']:
            p=calibrated[(calibrated.model==model)&(calibrated.calibration==stage)]
            conf=p[r.PROBS].max(axis=1).to_numpy()
            correct=(p.predicted_label==p.true_label).to_numpy()
            bid=np.minimum((conf*10).astype(int),9)
            points=[]
            for k in range(10):
                mask=bid==k
                if mask.any():
                    points.append((conf[mask].mean(),correct[mask].mean()))
                    bins.append(dict(model=model,calibration=stage,bin=k,n_seed_file_predictions=int(mask.sum()),
                                     n_unique_files=p[mask].file_id.nunique(),confidence=points[-1][0],accuracy=points[-1][1]))
            ax.plot(*np.array(points).T,'o-',label=stage,markersize=3)
        ax.plot([0,1],[0,1],'--',color='gray'); ax.set(xlabel='Confidence',ylabel='Empirical accuracy',
            xlim=(-.01,1.01),ylim=(-.02,1.02),title=model); ax.legend()
    save_figure(fig,figures/'reliability_diagram')
    q.save_df(output/'reliability_bins.csv',pd.DataFrame(bins))
    (output/'figure_notes.md').write_text('''# Figure contract and interpretation

Python quantitative comparisons, editable SVG/PDF and 400 dpi PNG. Models are
compared on 56 unique source MAT files, four load folds, five seeds. Repeated
seeds reuse the same files; they are not extra independent observations.
Seed scatter and ablation error bars: mean over four load scores within each
seed, then seed mean and sample SD (not a 95% CI). All five seeds retained.
Confusion: cell counts averaged over seeds, sums to 56. Persistent errors:
all files with at least one wrong seed for the displayed model.
Reliability: ten fixed equal-width confidence bins, pooled seed-file predictions
for descriptive visualization only, empty bins absent. Unique-file counts and
prediction counts are exported separately. No independence or p-value claim.
PCA is explanatory, fitted on loads 1–3 only. Feature shift uses all 26 file-mean
features; scaling uses training-file overall SD, not held-out SD. Associations
are not causal evidence. Models, errors, calibration and ablation each have
machine-readable CSV source data beside this file. No target data used.
Static QA warnings reviewed: TIFF is not required by this task, PNG is a 400 dpi
preview with SVG/PDF as vector masters. Random draws are file-bootstrap indices,
not synthetic observations. Log probabilities are clipped at 1e-12 and the
temperature search uses strictly positive bounds. These are not unguarded logs.
''',encoding='utf-8')


def report(output,summary,seed_scores,leader,decision):
    transfer=json.loads((output/'transfer20_source_summary.json').read_text(encoding='utf-8'))
    pairs=pd.read_csv(output/'paired_model_comparison.csv')
    folds=pd.read_csv(output/'multiseed_lolo_results.csv')
    errors=pd.read_csv(output/'persistent_misclassifications.csv')
    mlp=seed_scores[seed_scores.model=='mlp'].set_index('seed')
    wins={}
    for model in ['logistic_regression','random_forest']:
        other=seed_scores[seed_scores.model==model].set_index('seed')
        wins[model]=int((mlp.macro_f1>other.macro_f1).sum())
    weak={model:g.loc[g.groupby('seed').macro_f1.idxmin()].held_out_load.value_counts().to_dict()
          for model,g in folds.groupby('model')}
    distances=pd.read_csv(output/'file_class_distances.csv').set_index('file_id')
    shifts=pd.read_csv(output/'class_conditional_feature_shift.csv')
    ir_shift=shifts[(shifts.held_out_load==0)&(shifts.label=='IR')]
    top=ir_shift.loc[ir_shift.standardized_shift.abs().nlargest(3).index]
    ablation=pd.read_csv(output/'feature_group_ablation.csv')
    cal_means=pd.read_csv(output/'calibration_pooled_by_seed.csv').groupby(['model','calibration'])[['nll','brier','ece']].mean().reset_index()
    lines=['# 第二问深入完善：实际结果与交接', '',
        '本次只使用第一问已冻结的源域特征与 metadata。outputs/q2 保留为旧基线；本目录为完善版。', '',
        '## 复现与必要修正', '',
        '原基线 448 条 LOLO/Group CV 文件预测和概率逐项完全复现，56 文件、806 窗口、26 维，外层泄漏为零。',
        '完善版保留文件级外层 LOLO、类/文件等权采样与窗口 softmax 均值。修正 sklearn 随机模型固定种子问题；',
        '所有 MLP 候选共用同一组内层折和初始化种子。因每个 MAT 已聚合为一行，用 StratifiedKFold 直接分层文件组，',
        '确保稀少 N 类进入每折；通常三折，校准子内层仅剩两个 N 文件时降为二折。传统模型内层也统一按 argmax(predict_proba) 计分，修复 SVM 决策函数与概率分类规则不一致。该修正会改变 seed=2025 结果，不能称其为原版同协议重跑。', '',
        '## 多种子主结果', '',
        '以下 ± 是五个随机种子之间的样本标准差；每个 seed 先取四个 LOLO 折的算术均值。', '',
        '| 模型 | Macro-F1 | BA | F1 min–max |', '|---|---|---|---|']
    for _,v in summary.iterrows():
        lines.append(f'| {v.model} | {v.macro_f1_mean:.3f} ± {v.macro_f1_std:.3f} | {v.balanced_accuracy_mean:.3f} ± {v.balanced_accuracy_std:.3f} | {v.macro_f1_min:.3f}–{v.macro_f1_max:.3f} |')
    lines += ['',f'当前均值领先模型：{leader}。MLP 分别胜过 LR / RF 的种子数为 {wins}（共五个）。',
        '主模型推荐依据完整五种子均值；模型选择后的同批分数仍属于比较性估计，没有额外独立测试集证明获选模型的性能。',
        f'各模型每个 seed 最弱载荷计数：{weak}。若载荷并列，此计数按首个最小值计；完整每折表保留。', '',
        '## 错误文件与解释范围', '',
        '重点文件的错误种子数如下（无错误也保留）：', '',
        '| 模型 | 文件 | 错误 seeds | 常见错误类别 | 真类平均概率 |','|---|---|---|---|---|']
    for _,v in errors[errors.file_id.isin(['IR014_0','IR021_0','OR014@6_0'])].iterrows():
        wrong_label=v.most_common_wrong_label if pd.notna(v.most_common_wrong_label) else '无错误'
        lines.append(f'| {v.model} | {v.file_id} | {v.n_wrong_seeds}/5 | {wrong_label} | {v.mean_true_class_probability:.3f} |')
    lines += ['', '全部文件的反复错误、按载荷/尺寸/位置错误率、训练类中心距离、类条件特征漂移分别见 persistent_misclassifications.csv、error_factor_summary.csv、file_class_distances.csv、class_conditional_feature_shift.csv。',
        '距离基于外层训练文件拟合的 52D 标准化空间；PCA 仅作事后解释。错误与尺寸/位置/载荷的关联不是独立因果证据，且类中心距离不是概率。', '',
        f'IR014_0 距离真实 IR 类中心 {distances.loc["IR014_0","true_class_distance"]:.3f}、最近其他类中心 {distances.loc["IR014_0","nearest_other_distance"]:.3f}（最近类别 {distances.loc["IR014_0","nearest_class"]}）；OR014@6_0 最近类别为 {distances.loc["OR014@6_0","nearest_class"]}。这支持局部特征重叠与错误有关。',
        f'但 IR021_0 最近中心仍为 {distances.loc["IR021_0","nearest_class"]}，中心距离无法解释其多数种子误判。需要保留非线性边界、样本稀少和训练随机性的可能性。',
        '0 hp IR 组相对其他载荷 IR 组的最大标准化位移（分母为训练文件整体 SD）为：'+', '.join(f'{v.feature}={v.standardized_shift:.3f}' for _,v in top.iterrows())+'。每折测试 IR 只有三个文件，无法据此确定物理因果。',
        'OR014@6_3 的跨种子持续错误说明薄弱点不限于 0 hp。metadata_load_by_*.csv 确认四个载荷具有相同的类别、故障尺寸和外圈位置计数，不支持“0 hp 某种尺寸/位置样本数量少”这个解释。但同类内部位置/尺寸差异、转速变化及物理试验关联仍限制因果归因。', '',
        '## Bootstrap 与模型差异', '',
        '每个模型每个 seed 的 56 个 OOF 文件按真实类别重采样 2000 次；模型及 seed 使用相同文件抽样索引。',
        '同时报告每 seed 区间与 seed 平均 pooled 指标区间。这里先合并四折计算 pooled F1，再平均 seeds；与前文先平均四折 F1 定义不同。',
        '以下差值为 seed 平均 pooled 指标；95% 为 percentile 区间，未对多个比较作校正。', '',
        '| 比较 | 指标 | 差值 | 95% CI |','|---|---|---|---|']
    for _,v in pairs.iterrows(): lines.append(f'| {v.comparison} | {v.metric} | {v.estimate:.3f} | [{v.ci_low:.3f}, {v.ci_high:.3f}] |')
    lines += ['', '区间包含零时不能称显著优于。所有区间条件于现有训练/预测和类比例，不覆盖重训不确定性或未见载荷总体；四折模型训练集重叠，文件也可能共享物理工况。N 仅四文件且若全判对，bootstrap Recall 可能退化为 [1,1]，不表示总体 Recall 必为一。', '',
        '## 校准与第三问概率边界', '',
        '校准当前均值领先模型及 MLP。每个外层训练集内部再次交叉拟合，校准文件既不参与该次模型训练，也不参与超参数和早停选择。',
        '先算文件内窗口 softmax 均值，再以 log(文件概率)/T 校准；区别于“窗口 logits/T 后平均”。T 在 [0.05,20] 的固定对数网格上以文件 NLL 选择。正 T 不改变文件 argmax。',
        'NLL 按文件平均，Brier 为四类平方误差之和的文件均值，ECE 用十个等宽置信区间。采用建议要求平均 NLL/ECE 均改善且至少四个 seed 两者同时改善；记录所有结果，不选取有利折。',
        '', '| 模型 | 校准 | NLL | Brier | ECE |', '|---|---|---|---|---|',
        *[f'| {v.model} | {v.calibration} | {v.nll:.3f} | {v.brier:.3f} | {v.ece:.3f} |' for _,v in cal_means.iterrows()], '',
        '```json',json.dumps(decision,ensure_ascii=False,indent=2),'```',
        '内层校准文件来自已见的训练载荷，而外层是未见载荷，概率可靠性并不必然外推。当前结果支持“不采用这次校准”，不能据此断言所有校准方法普遍无效。',
        '校准参数按折保存在 calibration_runs/；默认导出的全源模型保持未经校准。源域 T 不可直接当作目标域可信度保证；Transfer20 概率也尚未校准。', '',
        '## 消融与 20D 接口', '',
        '预先固定 Diagnostic26、Transfer20、去绝对幅值、去阶次、去包络五组；LR/MLP 各跑五 seed × 四折，见 feature_group_ablation.csv。未依据测试折动态筛特征。', '',
        '| 特征组 | 模型 | Macro-F1 | BA | 最低类 Recall | 0 hp F1 |',
        '|---|---|---|---|---|---|',
        *[f'| {v.feature_group} | {v.model} | {v.macro_f1_mean:.3f} ± {v.macro_f1_std:.3f} | {v.balanced_accuracy_mean:.3f} | {v.min_class_mean_recall:.3f} | {v.load0_macro_f1_mean:.3f} |' for _,v in ablation.iterrows()], '',
        '消融分数差异属于预定义特征组的关联性比较；未进行等效性检验，分数接近不等于证明统计等效，也不自动改变第二问冻结的 26D 主任务。',
        f'Transfer20 MLP：LOLO Macro-F1={transfer["macro_f1_seed_mean"]:.3f} ± {transfer["macro_f1_seed_std"]:.3f}，BA={transfer["balanced_accuracy_seed_mean"]:.3f}；最低类别均值 Recall={transfer["minimum_class_mean_recall"]:.3f}。',
        f'eligible_for_q3_initialization={transfer["eligible_for_q3_initialization"]}；target_data_used=false；target_accuracy=null。',
        '门槛 0.85/0.70 仅为工程规则。20→32→4 加载与推理检查通过；20 维特征顺序严格对应第一问 transfer schema。固定 seed=2025 在全源域重新调参与重训，未选择最好的 seed。',
        'Diagnostic26 和 Transfer20 均独立导出 encoder/classifier/scaler/schema/config；最佳传统模型亦保存完整 pipeline。全源域重训没有新的独立测试分数。', '',
        '## 实际运行与复现', '',
        '```powershell', 'python src/q2_pipeline.py --output-dir outputs/q2_recheck_baseline',
        'python src/q2_robustness.py --stage diagnostic', 'python src/q2_robustness.py --stage calibration',
        'python src/q2_robustness.py --stage ablation', 'python src/q2_robustness.py --stage transfer',
        'python src/q2_robustness.py --stage analysis', 'python -m unittest discover -s tests -p "test_q2_*.py"', '```',
        '可用 --stage all 一次运行。runs 缓存按输入/训练代码哈希验证；q2_refined_config.json 记录版本、特征输入哈希、随机种子、采样与校准定义。', '',
        '## 修改范围与后续工作', '',
        '新增 src/q2_robustness.py、src/q2_analysis.py、src/q2_transfer_pretrain.py 及 tests；第一问源数据、标签、metadata、原 q2_pipeline.py 和 outputs/q2 基线未修改。',
        '现有项目交接文件为根目录 项目总体进程实现.txt；本报告与 项目说明/第二问深入完善交接.md 是本轮续接入口。',
        '第二问已完成源域证据补强，尚未训练目标域迁移算法。进入第三问前先查工程门槛与源/目标特征 schema；从 source-only 基线开始设计无监督验证，需检测域漂移、负迁移和置信度变化。',
        '源域只有 56 个独立文件、N 仅四文件；需区分各载荷数量覆盖和故障尺寸/位置的物理混杂。CWRU 跨载荷泛化不等同高速列车目标泛化。',
        '补充审计：第一问 Diagnostic26 来自全源域方差/相关性筛选及预定可疑特征排除；MI 用作审计而未用于该筛选条件。故本轮嵌套验证只覆盖固定 schema 下第二问的拟合过程，不能声称从特征筛选开始的端到端完全独立。若需这种声明，必须把第一问数据驱动筛选纳入外层训练折重做；本次按任务要求保持第一问不变。Transfer20 则是第一问预定义的固定名单。']
    (output/'q2_refined_summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
