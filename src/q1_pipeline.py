"""Question 1: auditable source construction and file-level validation only.

The script stops before target classification. A--P are exported only as
unlabelled, homogeneous transfer features.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, re, time
from fractions import Fraction
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal, stats
from scipy.io import loadmat
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, recall_score
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

plt.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Arial","DejaVu Sans"],"svg.fonttype":"none","pdf.fonttype":42,"font.size":8,"axes.spines.right":False,"axes.spines.top":False})
FS_MAIN,FS_COMMON=32000,12000
LABELS=("N","OR","IR","B"); COLORS={"N":"#4D4D4D","OR":"#B64342","IR":"#0F4D92","B":"#42949E"}
ID_COLUMNS=("file_id","file_path","window_id","label","branch","rpm")
SUSPECT_FEATURES=("acf_first_peak_s","order_dominant")
CORE_TRANSFER_FEATURES=("shape_skew","shape_kurtosis","shape_crest","zcr","psd_entropy","psd_peak_ratio","psd_0_500","psd_500_1000","psd_1000_2000","psd_4000_8000","env_kurtosis","env_entropy","env_peak_ratio","env_0_500","env_500_1000","env_2000_4000","order_0.5_2","order_2_4","order_4_8","order_8_16")

def save_csv(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    if rows:
        with path.open("w",encoding="utf-8-sig",newline="") as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def save_df(path,frame):
    path.parent.mkdir(parents=True,exist_ok=True);frame.to_csv(path,index=False,encoding="utf-8-sig")
def savefig(fig,base,tight=True):
    base.parent.mkdir(parents=True,exist_ok=True)
    if tight:fig.tight_layout()
    for suffix,kwargs in ((".svg",{}),(".pdf",{}),(".png",{"dpi":600}),(".tiff",{"dpi":600})):fig.savefig(base.with_suffix(suffix),bbox_inches="tight",**kwargs)
    plt.close(fig)
def branch(path):
    text=path.as_posix()
    for key,name,fs in (("12kHz_DE_data","12k_DE",12000),("12kHz_FE_data","12k_FE",12000),("48kHz_DE_data","48k_DE",48000),("48kHz_Normal_data","48k_normal",48000)):
        if key in text:return name,fs
    return ("target_32k",32000) if "目标域数据集" in text else ("unknown",0)
def label(path,data_branch):
    if data_branch=="48k_normal":return "N"
    return next((x for x in ("OR","IR","B") if f"/{x}/" in path.as_posix()),None)
def load(path,data_branch=None):
    data=loadmat(path);keys=[k for k in data if not k.startswith("__")]
    if path.stem in "ABCDEFGHIJKLMNOP" and "目标域数据集" in path.as_posix():return np.asarray(data[path.stem]).reshape(-1).astype(float),600.,keys,path.stem
    channel="FE" if data_branch=="12k_FE" else "DE"; key=next((k for k in keys if k.endswith("_"+channel+"_time")),None)
    if not key:key=next((k for k in keys if k.endswith("_DE_time") or k.endswith("_FE_time") or k.endswith("_BA_time")),None)
    if not key:raise ValueError("no acceleration variable")
    rpm_key=next((k for k in keys if k.endswith("RPM")),None);rpm=float(np.asarray(data[rpm_key]).reshape(-1)[0]) if rpm_key else None
    if rpm is None:
        m=re.search(r"\((\d{4})rpm\)",path.name);rpm=float(m.group(1)) if m else None
    return np.asarray(data[key]).reshape(-1).astype(float),rpm,keys,key
def resample(x,from_fs,to_fs):
    if from_fs==to_fs:return x
    q=Fraction(to_fs,from_fs).limit_denominator();return signal.resample_poly(x,q.numerator,q.denominator)
def starts(length,window,hop,limit=None):
    values=list(range(0,length-window+1,hop));return values if not limit or len(values)<=limit else sorted(set(np.linspace(0,values[-1],limit,dtype=int)))
def entropy(power):
    power=np.maximum(power,0);power=power/max(power.sum(),np.finfo(float).eps);valid=power[power>0]
    return float(-(valid*np.log(valid)).sum()/np.log(max(2,len(power))))
def feature_vector(x,fs,rpm):
    x=signal.detrend(x);mad=max(1.4826*np.median(abs(x-np.median(x))),np.finfo(float).eps);z=(x-np.median(x))/mad
    f,p=signal.welch(x,fs=fs,nperseg=min(2048,len(x)),scaling="spectrum");e=abs(signal.hilbert(z));ef,ep=signal.welch(e-e.mean(),fs=fs,nperseg=min(2048,len(e)),scaling="spectrum")
    ps,es=max(p.sum(),np.finfo(float).eps),max(ep.sum(),np.finfo(float).eps);ac=signal.correlate(z,z,mode="full",method="fft")[len(z)-1:len(z)-1+min(len(z),fs//4)];ac/=max(ac[0],np.finfo(float).eps);peaks,_=signal.find_peaks(ac[1:],prominence=.03);lag=(peaks[0]+1)/fs if len(peaks) else 0.
    d={"amp_rms":np.sqrt(np.mean(x*x)),"amp_std":np.std(x),"amp_peak":abs(x).max(),"amp_ptp":np.ptp(x),"amp_absmean":abs(x).mean(),"amp_energy":np.mean(x*x),"shape_skew":stats.skew(z),"shape_kurtosis":stats.kurtosis(z,fisher=False),"shape_crest":abs(z).max()/max(np.sqrt(np.mean(z*z)),1e-12),"shape_impulse":abs(z).max()/max(abs(z).mean(),1e-12),"zcr":np.mean(np.diff(np.signbit(z))!=0),"psd_centroid":(f*p).sum()/ps,"psd_bandwidth":np.sqrt((((f-(f*p).sum()/ps)**2)*p).sum()/ps),"psd_entropy":entropy(p),"psd_peak_ratio":p.max()/ps,"env_rms":np.sqrt(np.mean(e*e)),"env_kurtosis":stats.kurtosis(e,fisher=False),"env_entropy":entropy(ep),"env_peak_ratio":ep.max()/es,"acf_first_peak_s":lag,"acf_strength":ac[1:].max()}
    for lo,hi in ((0,500),(500,1000),(1000,2000),(2000,4000),(4000,8000)):
        d[f"psd_{lo}_{hi}"]=p[(f>=lo)&(f<hi)].sum()/ps;d[f"env_{lo}_{hi}"]=ep[(ef>=lo)&(ef<hi)].sum()/es
    if rpm:
        order=ef/(rpm/60);d["order_dominant"]=order[ep.argmax()];d["order_entropy"]=entropy(ep)
        for lo,hi in ((.5,2),(2,4),(4,8),(8,16),(16,32)):d[f"order_{lo}_{hi}"]=ep[(order>=lo)&(order<hi)].sum()/es
    return {k:float(np.nan_to_num(v)) for k,v in d.items()}
def records(root):return [(p,*branch(p),label(p,branch(p)[0])) for p in sorted(root.rglob("*.mat"))]
def rows_for(rs,target_fs,window,hop,limit=None):
    rows=[]
    for path,data_branch,native_fs,item_label in rs:
        x,rpm,_,_=load(path,data_branch);x=resample(x,native_fs,target_fs)
        for window_id,start in enumerate(starts(len(x),window,hop,limit)):
            row={"file_id":path.stem,"file_path":path.as_posix(),"window_id":window_id,"label":item_label or "","branch":data_branch,"rpm":rpm or np.nan};row.update(feature_vector(x[start:start+window],target_fs,rpm));rows.append(row)
    return rows
def audit(rs,out):
    rows=[];errors=[]
    for path,data_branch,native_fs,item_label in rs:
        try:
            x,rpm,keys,key=load(path,data_branch);rows.append({"file_id":path.stem,"file_path":path.as_posix(),"domain":"target" if data_branch=="target_32k" else "source","branch":data_branch,"label":item_label or "","sampling_rate":native_fs,"signal_variable":key,"all_variables":";".join(keys),"rpm":rpm or "","signal_length":len(x),"status":"ok"})
        except Exception as exc:rows.append({"file_id":path.stem,"file_path":path.as_posix(),"branch":data_branch,"label":item_label or "","status":str(exc)});errors.append(str(path))
    save_csv(out/"data_audit.csv",rows);(out/"data_audit_summary.json").write_text(json.dumps({"mat_files":len(rs),"errors":errors,"branch_counts":pd.DataFrame(rows).groupby("branch").size().to_dict()},ensure_ascii=False,indent=2),encoding="utf-8");return rows
def signal_quality_audit(rs,out):
    """Check raw recordings without altering them; amplitude differences are not errors by themselves."""
    rows=[]
    for path,data_branch,native_fs,item_label in rs:
        x,rpm,_,_=load(path,data_branch);finite=np.isfinite(x);valid=x[finite];maximum=float(np.max(abs(valid))) if len(valid) else np.nan
        rows.append({"file_id":path.stem,"file_path":path.as_posix(),"domain":"target" if data_branch=="target_32k" else "source","branch":data_branch,"label":item_label or "","sampling_rate":native_fs,"rpm":rpm or "","signal_length":len(x),"finite_fraction":float(finite.mean()),"zero_fraction":float(np.mean(valid==0)) if len(valid) else np.nan,"mean":float(np.mean(valid)) if len(valid) else np.nan,"std":float(np.std(valid)) if len(valid) else np.nan,"rms":float(np.sqrt(np.mean(valid*valid))) if len(valid) else np.nan,"max_abs":maximum,"peak_tie_fraction":float(np.mean(abs(valid)==maximum)) if len(valid) else np.nan,"sha256":hashlib.sha256(np.ascontiguousarray(valid).tobytes()).hexdigest() if len(valid) else ""})
    frame=pd.DataFrame(rows);frame["exact_duplicate_count"]=frame.groupby(["branch","sha256"])["file_id"].transform("size")
    frame["robust_z_log_rms"]=0.;frame["robust_z_log_peak"]=0.;frame["robust_z_length"]=0.
    for _,idx in frame.groupby(["branch","label"],dropna=False).groups.items():
        group=frame.loc[idx]
        if len(group)<4:continue
        for source,target in ((np.log1p(group.rms),"robust_z_log_rms"),(np.log1p(group.max_abs),"robust_z_log_peak"),(np.log1p(group.signal_length),"robust_z_length")):
            median=float(np.median(source));scale=1.4826*float(np.median(abs(source-median)))
            if scale>np.finfo(float).eps:frame.loc[idx,target]=(source-median)/scale
    flags=[]
    for _,row in frame.iterrows():
        problems=[]
        if row.finite_fraction<1:problems.append("nonfinite")
        if row["std"]<=1e-12:problems.append("near_constant")
        if row.zero_fraction>.99:problems.append("mostly_zero")
        if row.peak_tie_fraction>.01:problems.append("possible_clipping")
        if row.exact_duplicate_count>1:problems.append("exact_duplicate")
        if max(abs(row.robust_z_log_rms),abs(row.robust_z_log_peak),abs(row.robust_z_length))>5:problems.append("distribution_review")
        flags.append(";".join(problems) if problems else "ok")
    frame["quality_status"]=flags;save_df(out/"signal_quality_audit.csv",frame)
    summary={"files_scanned":len(frame),"hard_failures":int((frame.quality_status.isin(["nonfinite","near_constant","mostly_zero"])).sum()),"review_candidates":int((frame.quality_status!="ok").sum()),"status_counts":frame.quality_status.value_counts().to_dict(),"interpretation":"Amplitude differences are retained as physical recordings unless a hard signal-integrity failure is present. No automatic filtering or deletion is performed by this audit."}
    (out/"signal_quality_summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8");return frame
def metadata(source):
    rows=[]
    for path,data_branch,native_fs,item_label in source:
        x,rpm,_,key=load(path,data_branch);parts=path.parts;fault_size=next((part for part in reversed(parts) if re.fullmatch(r"\d{4}",part)),"");position=parts[parts.index("OR")+1] if item_label=="OR" else "";m=re.search(r"_(\d)(?:\D|$)",path.stem)
        rows.append({"file_id":path.stem,"file_path":path.as_posix(),"label":item_label,"load":m.group(1) if m else "","rpm":rpm or "","fault_size":fault_size,"fault_position":position,"sampling_rate":native_fs,"signal_length":len(x),"signal_variable":key})
    return pd.DataFrame(rows)
def feature_columns(frame):return [c for c in frame.columns if c not in ID_COLUMNS]
def candidate_selection(rs,out):
    target=pd.DataFrame(rows_for([r for r in rs if r[1]=="target_32k"],FS_COMMON,4096,4096,8));result=[]
    for name,groups in (("12k_DE",("12k_DE",)),("12k_FE",("12k_FE",)),("48k_DE",("48k_DE",)),("48k_DE_plus_normal",("48k_DE","48k_normal"))):
        source=pd.DataFrame(rows_for([r for r in rs if r[1] in groups],FS_COMMON,4096,4096,8));names=feature_columns(source);allx=pd.concat([source[names],target[names]]).replace([np.inf,-np.inf],np.nan).fillna(0);allx=StandardScaler().fit_transform(allx);x,y=allx[:len(source)],allx[len(source):];n=min(250,len(x),len(y));rng=np.random.default_rng(2025);x=x[rng.choice(len(x),n,False)];y=y[rng.choice(len(y),n,False)];dist=np.sqrt(((np.r_[x,y][:,None]-np.r_[x,y][None,:])**2).sum(2));gamma=1/max(np.median(dist[dist>0])**2,1e-12);mmd=float(rbf_kernel(x,x,gamma=gamma).mean()+rbf_kernel(y,y,gamma=gamma).mean()-2*rbf_kernel(x,y,gamma=gamma).mean());files=pd.concat([source.groupby("file_id")[names].mean(),target.groupby("file_id")[names].mean()]).fillna(0);domain=np.r_[np.ones(source.file_id.nunique()),np.zeros(target.file_id.nunique())];files=StandardScaler().fit_transform(files);cv=StratifiedKFold(min(5,int(min(sum(domain==0),sum(domain==1)))),shuffle=True,random_state=2025);accuracy=np.mean([LogisticRegression(max_iter=2000,class_weight="balanced").fit(files[a],domain[a]).score(files[b],domain[b]) for a,b in cv.split(files,domain)]);result.append({"candidate_domain":name,"sample_rate_common":12000,"common_band_hz":"0-6000","mmd":mmd,"proxy_a_distance":max(0,2*(2*accuracy-1)),"n_files":source.file_id.nunique(),"notes":"Auxiliary distance only; DE location, bandwidth and label coverage also govern selection."})
    save_csv(out/"source_domain_selection.csv",result)
def median_cv(raw,name):
    return float(np.median([np.std(g[name])/max(abs(np.mean(g[name])),1e-12) for _,g in raw.groupby("file_id")]))
def screen_features(raw):
    names=feature_columns(raw);x=raw[names].replace([np.inf,-np.inf],np.nan).fillna(0);mi_window=mutual_info_classif(x,raw.label,random_state=2025);file_mean=raw.groupby("file_id")[names].mean();file_meta=raw.groupby("file_id")[["label","rpm"]].first().reindex(file_mean.index);mi_file=mutual_info_classif(file_mean,file_meta.label,n_neighbors=3,random_state=2025);selected=[];rows=[]
    for i,name in enumerate(names):
        variance=float(x[name].var());reason="" if variance>=1e-10 and not any(abs(x[name].corr(x[prior]))>.95 for prior in selected) else "near_zero_variance_or_correlated"
        if not reason:selected.append(name)
        rho,pvalue=stats.spearmanr(file_mean[name],file_meta.rpm.astype(float),nan_policy="omit");rows.append({"feature":name,"variance":variance,"median_file_window_cv":median_cv(raw,name),"mutual_information_window_level":float(mi_window[i]),"mutual_information_file_level":float(mi_file[i]),"spearman_rho_file_mean_vs_rpm":float(np.nan_to_num(rho)),"spearman_p_file_mean_vs_rpm":float(np.nan_to_num(pvalue,nan=1.0)),"selected_initial":not bool(reason),"drop_reason":reason,"suspect_for_default_transfer":name in SUSPECT_FEATURES})
    return selected,pd.DataFrame(rows)
def aggregate(raw,meta,names):
    means=raw.groupby("file_id")[names].mean().add_suffix("_mean");stds=raw.groupby("file_id")[names].std(ddof=0).fillna(0).add_suffix("_std");result=meta.set_index("file_id").join(means).join(stds).reset_index()
    if result[["label","load"]].isna().any().any():raise ValueError("file-level aggregation lost metadata")
    return result
def models():return {"logistic_regression":lambda:make_pipeline(StandardScaler(),LogisticRegression(max_iter=5000,class_weight="balanced",random_state=2025)),"rbf_svm":lambda:make_pipeline(StandardScaler(),SVC(C=1.,kernel="rbf",gamma="scale",class_weight="balanced"))}
def lolo(file_features,base_names,set_name):
    names=[name+suffix for name in base_names for suffix in ("_mean","_std")];missing=[name for name in names if name not in file_features]
    if missing:raise ValueError(f"missing aggregate features: {missing}")
    x=file_features[names].replace([np.inf,-np.inf],np.nan).fillna(0);y=file_features.label.to_numpy();loads=file_features.load.astype(str).to_numpy();results=[];predictions=[]
    for held in ("0","1","2","3"):
        train,test=loads!=held,loads==held
        for model_name,factory in models().items():
            model=factory();model.fit(x.loc[train],y[train]);pred=model.predict(x.loc[test]);recalls=recall_score(y[test],pred,labels=LABELS,average=None,zero_division=0);results.append({"feature_set":set_name,"model":model_name,"test_load":int(held),"n_train_files":int(train.sum()),"n_test_files":int(test.sum()),"n_base_features":len(base_names),"n_model_features":len(names),"macro_f1":f1_score(y[test],pred,labels=LABELS,average="macro",zero_division=0),"balanced_accuracy":balanced_accuracy_score(y[test],pred),**{f"recall_{name}":value for name,value in zip(LABELS,recalls)}})
            for index,value in zip(file_features.index[test],pred):
                row=file_features.loc[index];predictions.append({"feature_set":set_name,"model":model_name,"test_load":int(held),"file_id":row.file_id,"label_true":row.label,"label_pred":value,"load":row.load,"rpm":row.rpm})
    return pd.DataFrame(results),pd.DataFrame(predictions)
def lolo_summary(results):
    metrics=["macro_f1","balanced_accuracy",*[f"recall_{name}" for name in LABELS]];return {model:{metric:{"mean":float(group[metric].mean()),"std":float(group[metric].std(ddof=1))} for metric in metrics} for model,group in results.groupby("model")}
def ablation_sets(all_names,selected):
    time_names=[name for name in all_names if name.startswith(("amp_","shape_")) or name=="zcr"];frequency=[name for name in all_names if name.startswith("psd_")];envelope=[name for name in all_names if name.startswith("env_")];order=[name for name in all_names if name.startswith("order_")];current=[name for name in selected if name in all_names]
    return {"A_all_candidate":all_names,"B_current_28":current,"C_core_transfer_20":list(CORE_TRANSFER_FEATURES),"D_time_only":time_names,"E_time_frequency":time_names+frequency,"F_time_frequency_envelope":time_names+frequency+envelope,"G_time_frequency_envelope_order":time_names+frequency+envelope+order,"B_without_acf_first_peak_s":[name for name in current if name!="acf_first_peak_s"],"B_without_order_dominant":[name for name in current if name!="order_dominant"]}
def run_ablation(file_features,sets,out):
    result_frames=[];prediction_frames=[]
    for name,names in sets.items():
        result,prediction=lolo(file_features,names,name);result_frames.append(result);prediction_frames.append(prediction)
    results=pd.concat(result_frames,ignore_index=True);summary=[]
    for (feature_set,model),group in results.groupby(["feature_set","model"],sort=False):
        recalls=[f"recall_{name}" for name in LABELS];summary.append({"feature_set":feature_set,"model":model,"n_features":int(group.n_base_features.iloc[0]),"macro_f1":group.macro_f1.mean(),"std_macro_f1":group.macro_f1.std(ddof=1),"balanced_accuracy":group.balanced_accuracy.mean(),"std_balanced_accuracy":group.balanced_accuracy.std(ddof=1),"min_class_recall":group[recalls].mean().min()})
    save_df(out/"feature_ablation.csv",pd.DataFrame(summary));save_df(out/"feature_ablation_folds.csv",results);save_df(out/"feature_ablation_predictions.csv",pd.concat(prediction_frames,ignore_index=True))
def plot_confusion(predictions,out):
    fig,axes=plt.subplots(1,2,figsize=(7.5,3.1),sharey=True)
    for axis,(model,group) in zip(axes,predictions.groupby("model",sort=False)):
        matrix=confusion_matrix(group.label_true,group.label_pred,labels=LABELS,normalize="true");image=axis.imshow(matrix,vmin=0,vmax=1,cmap="Blues");axis.set(title=model.replace("_"," "),xlabel="Predicted label",xticks=range(4),xticklabels=LABELS,ylabel="True label",yticks=range(4),yticklabels=LABELS)
        for i in range(4):
            for j in range(4):axis.text(j,i,f"{matrix[i,j]:.2f}",ha="center",va="center",fontsize=7,color="white" if matrix[i,j]>.55 else "black")
    fig.subplots_adjust(left=.08,right=.87,bottom=.17,top=.86,wspace=.36);color_axis=fig.add_axes([.90,.20,.018,.60]);fig.colorbar(image,cax=color_axis,label="Row-normalized recall");savefig(fig,out/"figures"/"lolo_confusion_matrix",tight=False)
def plot_distributions(raw,meta,out):
    chosen=("amp_rms","shape_kurtosis","env_kurtosis","psd_entropy","order_2_4","order_4_8");merged=raw.merge(meta[["file_id","load"]],on="file_id",how="left");fig,axes=plt.subplots(2,3,figsize=(9.2,5.2));rng=np.random.default_rng(2025)
    for axis,name in zip(axes.flat,chosen):
        boxes=axis.boxplot([merged.loc[merged.label==label,name].to_numpy() for label in LABELS],tick_labels=LABELS,patch_artist=True,showfliers=False)
        for patch,item_label in zip(boxes["boxes"],LABELS):patch.set(facecolor=COLORS[item_label],alpha=.25)
        for position,item_label in enumerate(LABELS,1):
            group=merged[merged.label==item_label]
            for load,color in zip(("0","1","2","3"),("#6A3D9A","#1B9E77","#D95F02","#7570B3")):
                values=group.loc[group.load.astype(str)==load,name];axis.scatter(position+rng.uniform(-.13,.13,len(values)),values,s=5,color=color,alpha=.32,label=f"{load} hp" if position==1 else None)
        axis.set_title(name)
    axes.flat[0].legend(fontsize=6,title="Load");savefig(fig,out/"figures"/"key_feature_distributions")
def mechanism_and_quality_figures(all_records,raw,selected,out):
    source=[record for record in all_records if record[1] in ("48k_DE","48k_normal")];chosen=[]
    for item_label,pattern in (("N",r"N_0\.mat$"),("B",r"B007_0\.mat$"),("IR",r"IR007_0\.mat$"),("OR",r"OR007@6_0\.mat$")):chosen.append(next(record for record in source if record[3]==item_label and re.search(pattern,record[0].name)))
    fig,axes=plt.subplots(4,4,figsize=(12,9))
    for i,(path,data_branch,native_fs,item_label) in enumerate(chosen):
        x,rpm,_,_=load(path,data_branch);x=resample(x,native_fs,FS_MAIN)[:16384];f,p=signal.welch(signal.detrend(x),fs=FS_MAIN,nperseg=2048);e=abs(signal.hilbert(x-x.mean()));ef,ep=signal.welch(e-e.mean(),fs=FS_MAIN,nperseg=2048);ac=signal.correlate(x,x,mode="full",method="fft")[len(x)-1:len(x)+3199];ac/=ac[0]
        for j,(xx,yy,xlabel) in enumerate(((np.arange(3200)/FS_MAIN*1000,x[:3200],"Time (ms)"),(f,p,"Frequency (Hz)"),(ef,ep,"Envelope frequency (Hz)"),(np.arange(len(ac))/FS_MAIN*1000,ac,"Lag (ms)"))):
            axes[i,j].plot(xx,yy,color=COLORS[item_label],lw=.7);axes[i,j].set_xlabel(xlabel);axes[i,j].set_title(item_label if j==0 else "");axes[i,j].set_xlim(0,100 if j in (0,3) else 8000)
    savefig(fig,out/"figures"/"mechanism_multimodal")
    fig,axes=plt.subplots(2,2,figsize=(9,6));theory={"OR":9/2*(1-.3126/1.537),"IR":9/2*(1+.3126/1.537),"B":1.537/(2*.3126)*(1-(.3126/1.537)**2)}
    for axis,(path,data_branch,native_fs,item_label) in zip(axes.flat,chosen):
        x,rpm,_,_=load(path,data_branch);x=resample(x,native_fs,FS_MAIN)[:16384];e=abs(signal.hilbert(x-x.mean()));ef,ep=signal.welch(e-e.mean(),fs=FS_MAIN,nperseg=2048);order=ef/(rpm/60);axis.plot(order,ep,color=COLORS[item_label],lw=.8);axis.set(xlim=(0,20),xlabel="Envelope order (f/fr)",ylabel="Power",title=item_label)
        if item_label in theory:
            axis.axvline(theory[item_label],color="#272727",ls="--",lw=.8,label="CWRU BPFO/BPFI/BSF")
            if item_label=="B":axis.axvline(2*theory[item_label],color="#767676",ls=":",lw=.8,label="2×BSF convention")
            axis.legend(fontsize=6)
    savefig(fig,out/"figures"/"mechanism_envelope_order")
    corr=raw[selected].corr().abs();fig,axis=plt.subplots(figsize=(9,7));image=axis.imshow(corr,cmap="viridis",vmin=0,vmax=1);fig.colorbar(image,ax=axis,label="|Pearson r|");axis.set(title="Selected-feature correlation",xticks=[],yticks=[]);savefig(fig,out/"figures"/"feature_correlation")
    file_means=raw.groupby(["file_id","label"])[selected].mean().reset_index();coordinates=PCA(2).fit_transform(StandardScaler().fit_transform(file_means[selected]));fig,axis=plt.subplots(figsize=(6,4))
    for item_label in LABELS:
        mask=file_means.label==item_label;axis.scatter(coordinates[mask,0],coordinates[mask,1],s=28,color=COLORS[item_label],label=item_label)
    axis.set(xlabel="PC1",ylabel="PC2",title="Full-source file-level PCA (descriptive)");axis.legend();savefig(fig,out/"figures"/"source_feature_pca")
def window_sensitivity(source,meta,default_raw,out):
    rows=[]
    for window in (8192,16384,32768):
        start=time.perf_counter();raw=pd.DataFrame(rows_for(source,FS_MAIN,window,window//2));selected,quality=screen_features(raw);features=aggregate(raw,meta,list(CORE_TRANSFER_FEATURES));results,_=lolo(features,list(CORE_TRANSFER_FEATURES),"C_core_transfer_20");runtime=time.perf_counter()-start
        for model,group in results.groupby("model"):
            recalls=group[[f"recall_{name}" for name in LABELS]].mean();rows.append({"window_samples":window,"duration_seconds":window/FS_MAIN,"hop_samples":window//2,"total_windows":len(raw),"mean_windows_per_file":raw.groupby("file_id").size().mean(),"selected_feature_count_descriptive":len(selected),"median_file_window_cv":quality.median_file_window_cv.median(),"model":model,"macro_f1":group.macro_f1.mean(),"balanced_accuracy":group.balanced_accuracy.mean(),"min_class_recall":recalls.min(),"runtime_seconds":runtime})
    result=pd.DataFrame(rows);save_df(out/"validation"/"window_sensitivity.csv",result);primary=result[result.model=="logistic_regression"];default=primary[primary.window_samples==16384].iloc[0];best_f1,best_ba=primary.macro_f1.max(),primary.balanced_accuracy.max()
    if default.macro_f1>=best_f1-.02 and default.balanced_accuracy>=best_ba-.02:decision="16384 samples retained: the predeclared primary Logistic probe is within 0.02 of the best LOLO Macro-F1 and balanced accuracy; it also uses fewer windows and has lower median file-window CV than 8192. RBF-SVM remains a sensitivity probe, not the window-selection rule."
    else:
        best=primary.sort_values(["macro_f1","balanced_accuracy"],ascending=False).iloc[0];decision=f"{int(best.window_samples)} samples has the highest primary-Logistic LOLO performance; 16384 did not meet the predeclared 0.02 tolerance."
    text="# 窗口长度敏感性\n\n"+decision+"\n\n```csv\n"+result.to_csv(index=False,float_format="%.4f",lineterminator="\n")+"```\n\n所有结果均按原始 MAT 文件做 LOLO；本表不是目标域准确率。特征配置固定为 20 维 Transfer 特征，未在测试载荷上进行监督选择。\n";(out/"validation"/"window_selection_summary.md").write_text(text,encoding="utf-8");return result
def export_sets(raw,target,selected,window,hop,out):
    diagnostic=[name for name in selected if name not in SUSPECT_FEATURES];transfer=list(CORE_TRANSFER_FEATURES);base=list(ID_COLUMNS);save_df(out/"features_source_diagnostic.csv",raw[base+diagnostic]);save_df(out/"features_source_transfer.csv",raw[base+transfer]);target_raw=pd.DataFrame(rows_for(target,FS_MAIN,window,hop));target_frame=target_raw[["file_id","file_path","window_id","branch","rpm"]+transfer].copy();target_frame["rpm_is_nominal"]=True;target_frame["rpm_note"]="600 rpm is the problem-stated nominal speed, not instantaneous synchronous speed.";save_df(out/"features_target_transfer.csv",target_frame);(out/"feature_names_diagnostic.json").write_text(json.dumps({"features":diagnostic,"n_features":len(diagnostic),"purpose":"Question 2 source-domain diagnosis","excluded_suspects":list(SUSPECT_FEATURES)},ensure_ascii=False,indent=2),encoding="utf-8");(out/"feature_names_transfer.json").write_text(json.dumps({"features":transfer,"n_features":len(transfer),"purpose":"Question 3 source/target transfer representation","excludes_absolute_amplitude":True,"excludes_target_bearing_geometry":True},ensure_ascii=False,indent=2),encoding="utf-8");return diagnostic,transfer
def main():
    parser=argparse.ArgumentParser();parser.add_argument("--data-root",type=Path,default=Path("数据集")/"数据集");parser.add_argument("--output-dir",type=Path,default=Path("outputs")/"q1");parser.add_argument("--window",type=int,default=16384);parser.add_argument("--hop",type=int,default=8192);args=parser.parse_args()
    if args.hop!=args.window//2:raise ValueError("Question 1 fixes hop at 50% overlap: hop must equal window // 2")
    args.output_dir.mkdir(parents=True,exist_ok=True);all_records=records(args.data_root);source=[record for record in all_records if record[1] in ("48k_DE","48k_normal")];target=[record for record in all_records if record[1]=="target_32k"];audit_rows=audit(all_records,args.output_dir);signal_quality_audit(all_records,args.output_dir);candidate_selection(all_records,args.output_dir);meta=metadata(source);save_df(args.output_dir/"source_metadata.csv",meta);raw=pd.DataFrame(rows_for(source,FS_MAIN,args.window,args.hop));save_df(args.output_dir/"features_source_raw.csv",raw);selected,quality=screen_features(raw);save_df(args.output_dir/"feature_quality.csv",quality);save_df(args.output_dir/"features_source_selected.csv",raw[list(ID_COLUMNS)+selected]);(args.output_dir/"feature_names.json").write_text(json.dumps({"selected_features":selected,"n_features":len(selected),"rule":"Descriptive low-variance removal and greedy |Pearson r| > 0.95 removal; MI is descriptive only. LOLO uses pre-specified sets."},ensure_ascii=False,indent=2),encoding="utf-8")
    file_features=aggregate(raw,meta,feature_columns(raw));save_df(args.output_dir/"validation"/"file_level_features.csv",file_features);diagnostic,transfer=export_sets(raw,target,selected,args.window,args.hop,args.output_dir);lolo_results,lolo_predictions=lolo(file_features,diagnostic,"diagnostic_default");save_df(args.output_dir/"validation"/"lolo_results.csv",lolo_results);save_df(args.output_dir/"validation"/"lolo_predictions.csv",lolo_predictions);(args.output_dir/"validation"/"lolo_summary.json").write_text(json.dumps({"independent_unit":"raw MAT file","split":"Leave-One-Load-Out; 0/1/2/3 hp held out in turn","default_feature_set":diagnostic,"metrics":lolo_summary(lolo_results),"scope_boundary":"Source-domain cross-load validation only; it does not estimate A-P target accuracy."},ensure_ascii=False,indent=2),encoding="utf-8");run_ablation(file_features,ablation_sets(feature_columns(raw),selected),args.output_dir/"validation");plot_confusion(lolo_predictions,args.output_dir);plot_distributions(raw,meta,args.output_dir);mechanism_and_quality_figures(all_records,raw,selected,args.output_dir);sensitivity=window_sensitivity(source,meta,raw,args.output_dir)
    summary=f"# 第一问总结\n\n审计 MAT 文件：{len(audit_rows)}。正式源域：48 kHz DE 全部故障文件加 48 kHz Normal，共 {len(meta)} 个文件；类别计数：{meta.groupby('label').size().to_dict()}。\n\n正式特征使用 48→32 kHz 抗混叠重采样、{args.window} 点窗口和 {args.hop} 点步长。描述性相关性筛选保留 {len(selected)} 个特征；默认 Diagnostic 特征为 {len(diagnostic)} 维，Transfer 特征为 {len(transfer)} 维。\n\n已完成文件级 LOLO、特征消融、窗口敏感性和目标域同构 Transfer 特征输出。所有 LOLO 以原始 MAT 文件为独立样本；该结果只说明源域跨载荷诊断价值，不能构成 A-P 目标域准确率证据。\n\n窗口敏感性结果保存在 `validation/window_sensitivity.csv`，共 {len(sensitivity)} 条模型-窗口记录。第一问不包含 A-P 分类、CORAL 或 DANN 结论。\n";(args.output_dir/"q1_summary.md").write_text(summary,encoding="utf-8");print(json.dumps({"audit_files":len(audit_rows),"full_source_files":len(meta),"windows":len(raw),"selected_features":len(selected),"diagnostic_features":len(diagnostic),"transfer_features":len(transfer)},ensure_ascii=False))
if __name__=="__main__":main()
