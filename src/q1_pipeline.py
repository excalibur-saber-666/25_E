"""Question 1 only: audit, source selection, mechanism figures and feature data."""
from __future__ import annotations
import argparse, csv, json, re
from fractions import Fraction
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Arial","DejaVu Sans"],"svg.fonttype":"none","pdf.fonttype":42,"font.size":8,"axes.spines.right":False,"axes.spines.top":False})
import numpy as np, pandas as pd
from scipy import signal, stats
from scipy.io import loadmat
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

FS_MAIN, FS_COMMON = 32000, 12000
LABELS=("N","OR","IR","B"); COLORS={"N":"#4D4D4D","OR":"#B64342","IR":"#0F4D92","B":"#42949E"}

def save_csv(path, rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    if rows:
        with path.open("w",encoding="utf-8-sig",newline="") as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
def savefig(fig, base):
    base.parent.mkdir(parents=True,exist_ok=True); fig.tight_layout()
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(base.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)
def branch(path):
    s=path.as_posix()
    for key,name,fs in (("12kHz_DE_data","12k_DE",12000),("12kHz_FE_data","12k_FE",12000),("48kHz_DE_data","48k_DE",48000),("48kHz_Normal_data","48k_normal",48000)):
        if key in s: return name,fs
    return ("target_32k",32000) if "目标域数据集" in s else ("unknown",0)
def label(path,b):
    s=path.as_posix()
    return "N" if b=="48k_normal" else next((x for x in ("OR","IR","B") if f"/{x}/" in s),None)
def load(path,b=None):
    d=loadmat(path); keys=[k for k in d if not k.startswith("__")]
    if path.stem in "ABCDEFGHIJKLMNOP" and "目标域数据集" in path.as_posix(): return np.asarray(d[path.stem]).reshape(-1).astype(float),600.,keys,path.stem
    chan="FE" if b=="12k_FE" else "DE"; k=next((x for x in keys if x.endswith("_"+chan+"_time")),None)
    if not k: k=next((x for x in keys if x.endswith("_DE_time") or x.endswith("_FE_time") or x.endswith("_BA_time")),None)
    if not k: raise ValueError("no acceleration variable")
    r=next((x for x in keys if x.endswith("RPM")),None); rpm=float(np.asarray(d[r]).reshape(-1)[0]) if r else None
    if rpm is None:
        m=re.search(r"\((\d{4})rpm\)",path.name); rpm=float(m.group(1)) if m else None
    return np.asarray(d[k]).reshape(-1).astype(float),rpm,keys,k
def resample(x,a,b):
    if a==b:return x
    q=Fraction(b,a).limit_denominator(); return signal.resample_poly(x,q.numerator,q.denominator)
def starts(n,w,h,limit=None):
    z=list(range(0,n-w+1,h)); return z if not limit or len(z)<=limit else sorted(set(np.linspace(0,z[-1],limit,dtype=int)))
def ent(p):
    p=np.maximum(p,0); p=p/max(p.sum(),np.finfo(float).eps); return float(-(p[p>0]*np.log(p[p>0])).sum()/np.log(max(2,len(p))))
def feat(x,fs,rpm):
    x=signal.detrend(x); mad=max(1.4826*np.median(abs(x-np.median(x))),np.finfo(float).eps); z=(x-np.median(x))/mad
    f,p=signal.welch(x,fs=fs,nperseg=min(2048,len(x)),scaling="spectrum"); e=abs(signal.hilbert(z)); ef,ep=signal.welch(e-e.mean(),fs=fs,nperseg=min(2048,len(e)),scaling="spectrum")
    ps=max(p.sum(),np.finfo(float).eps); es=max(ep.sum(),np.finfo(float).eps); ac=signal.correlate(z,z,mode="full",method="fft")[len(z)-1:len(z)-1+min(len(z),fs//4)]; ac/=max(ac[0],np.finfo(float).eps); pk,_=signal.find_peaks(ac[1:],prominence=.03); lag=(pk[0]+1)/fs if len(pk) else 0
    d={"amp_rms":np.sqrt(np.mean(x*x)),"amp_std":np.std(x),"amp_peak":abs(x).max(),"amp_ptp":np.ptp(x),"amp_absmean":abs(x).mean(),"amp_energy":np.mean(x*x),"shape_skew":stats.skew(z),"shape_kurtosis":stats.kurtosis(z,fisher=False),"shape_crest":abs(z).max()/max(np.sqrt(np.mean(z*z)),1e-12),"shape_impulse":abs(z).max()/max(abs(z).mean(),1e-12),"zcr":np.mean(np.diff(np.signbit(z))!=0),"psd_centroid":(f*p).sum()/ps,"psd_bandwidth":np.sqrt((((f-(f*p).sum()/ps)**2)*p).sum()/ps),"psd_entropy":ent(p),"psd_peak_ratio":p.max()/ps,"env_rms":np.sqrt(np.mean(e*e)),"env_kurtosis":stats.kurtosis(e,fisher=False),"env_entropy":ent(ep),"env_peak_ratio":ep.max()/es,"acf_first_peak_s":lag,"acf_strength":ac[1:].max()}
    for lo,hi in ((0,500),(500,1000),(1000,2000),(2000,4000),(4000,8000)): d[f"psd_{lo}_{hi}"]=p[(f>=lo)&(f<hi)].sum()/ps; d[f"env_{lo}_{hi}"]=ep[(ef>=lo)&(ef<hi)].sum()/es
    if rpm:
        o=ef/(rpm/60); d["order_dominant"]=o[ep.argmax()]; d["order_entropy"]=ent(ep)
        for lo,hi in ((.5,2),(2,4),(4,8),(8,16),(16,32)):d[f"order_{lo}_{hi}"]=ep[(o>=lo)&(o<hi)].sum()/es
    return {k:float(np.nan_to_num(v)) for k,v in d.items()}
def records(root):
    return [(p,*branch(p),label(p,branch(p)[0])) for p in sorted(root.rglob("*.mat"))]
def rows_for(rs,targetfs,w,h,limit):
    out=[]
    for p,b,fs,lab in rs:
        x,r,_,_=load(p,b); x=resample(x,fs,targetfs)
        for i,s in enumerate(starts(len(x),w,h,limit)):
            q={"file_id":p.stem,"file_path":p.as_posix(),"window_id":i,"label":lab or "","branch":b,"rpm":r or np.nan}; q.update(feat(x[s:s+w],targetfs,r));out.append(q)
    return out
def audit(rs,out):
    z=[]; bad=[]
    for p,b,fs,lab in rs:
        try:
            x,r,keys,k=load(p,b); z.append({"file_id":p.stem,"file_path":p.as_posix(),"domain":"target" if b=="target_32k" else "source","branch":b,"label":lab or "","sampling_rate":fs,"signal_variable":k,"all_variables":";".join(keys),"rpm":r or "","signal_length":len(x),"status":"ok"})
        except Exception as e:z.append({"file_id":p.stem,"file_path":p.as_posix(),"branch":b,"label":lab or "","status":str(e)});bad.append(str(p))
    save_csv(out/"data_audit.csv",z); (out/"data_audit_summary.json").write_text(json.dumps({"mat_files":len(rs),"errors":bad,"branch_counts":pd.DataFrame(z).groupby("branch").size().to_dict()},ensure_ascii=False,indent=2),encoding="utf-8");return z
def selection(rs,out):
    target=[r for r in rs if r[1]=="target_32k"]; t=pd.DataFrame(rows_for(target,FS_COMMON,4096,4096,8)); result=[]
    for name,groups in (("12k_DE",("12k_DE",)),("12k_FE",("12k_FE",)),("48k_DE",("48k_DE",)),("48k_DE_plus_normal",("48k_DE","48k_normal"))):
        c=pd.DataFrame(rows_for([r for r in rs if r[1] in groups],FS_COMMON,4096,4096,8)); cols=[x for x in c if x not in ("file_id","file_path","window_id","label","branch","rpm")]; allx=pd.concat([c[cols],t[cols]]).replace([np.inf,-np.inf],np.nan).fillna(0); allx=StandardScaler().fit_transform(allx); x,y=allx[:len(c)],allx[len(c):]; n=min(250,len(x),len(y)); rng=np.random.default_rng(2025);x=x[rng.choice(len(x),n,False)];y=y[rng.choice(len(y),n,False)];d=np.sqrt(((np.r_[x,y][:,None]-np.r_[x,y][None,:])**2).sum(2));g=1/max(np.median(d[d>0])**2,1e-12);m=float(rbf_kernel(x,x,gamma=g).mean()+rbf_kernel(y,y,gamma=g).mean()-2*rbf_kernel(x,y,gamma=g).mean());fx=pd.concat([c.groupby("file_id")[cols].mean(),t.groupby("file_id")[cols].mean()]).fillna(0);ly=np.r_[np.ones(c.file_id.nunique()),np.zeros(t.file_id.nunique())]; fx=StandardScaler().fit_transform(fx); cv=StratifiedKFold(min(5,int(min(sum(ly==0),sum(ly==1)))),shuffle=True,random_state=2025);acc=np.mean([LogisticRegression(max_iter=2000,class_weight="balanced").fit(fx[a],ly[a]).score(fx[b],ly[b]) for a,b in cv.split(fx,ly)]);result.append({"candidate_domain":name,"sample_rate_common":12000,"common_band_hz":"0-6000","mmd":m,"proxy_a_distance":max(0,2*(2*acc-1)),"n_files":c.file_id.nunique(),"notes":"Auxiliary distance only; DE location, bandwidth and label coverage also govern selection."})
    save_csv(out/"source_domain_selection.csv",result);return result
def figures(rs,raw,keep,out):
    chosen=[]
    for lab,pat in (("N",r"N_0\.mat$"),("B",r"B007_0\.mat$"),("IR",r"IR007_0\.mat$"),("OR",r"OR007@6_0\.mat$")):chosen.append(next(r for r in rs if r[3]==lab and re.search(pat,r[0].name)))
    fig,ax=plt.subplots(4,4,figsize=(12,9));
    for i,(p,b,fs,lab) in enumerate(chosen):
        x,r,_,_=load(p,b);x=resample(x,fs,FS_MAIN)[:16384]; f,ps=signal.welch(signal.detrend(x),fs=FS_MAIN,nperseg=2048); e=abs(signal.hilbert(x-x.mean()));ef,ep=signal.welch(e-e.mean(),fs=FS_MAIN,nperseg=2048); ac=signal.correlate(x,x,mode="full",method="fft")[len(x)-1:len(x)+3199];ac/=ac[0]
        for j,(xx,yy,xl) in enumerate(((np.arange(3200)/FS_MAIN*1000,x[:3200],"Time (ms)"),(f,ps,"Frequency (Hz)"),(ef,ep,"Envelope frequency (Hz)"),(np.arange(len(ac))/FS_MAIN*1000,ac,"Lag (ms)"))):ax[i,j].plot(xx,yy,color=COLORS[lab],lw=.7);ax[i,j].set_xlabel(xl);ax[i,j].set_title(lab if j==0 else "");ax[i,j].set_xlim(0,100 if j in (0,3) else 8000)
    savefig(fig,out/"figures"/"mechanism_multimodal")
    # CWRU DE geometry: 9 balls, d=0.3126 in, D=1.537 in.  This source-only
    # annotation is deliberately not transferred to the unknown target bearing.
    fig,ax=plt.subplots(2,2,figsize=(9,6)); theory={"OR":9/2*(1-.3126/1.537),"IR":9/2*(1+.3126/1.537),"B":1.537/(2*.3126)*(1-(.3126/1.537)**2)}
    for aa,(p,b,fs,lab) in zip(ax.flat,chosen):
        x,r,_,_=load(p,b);x=resample(x,fs,FS_MAIN)[:16384];e=abs(signal.hilbert(x-x.mean()));ef,ep=signal.welch(e-e.mean(),fs=FS_MAIN,nperseg=2048);order=ef/(r/60);aa.plot(order,ep,color=COLORS[lab],lw=.8);aa.set(xlim=(0,20),xlabel="Envelope order (f/fr)",ylabel="Power",title=lab)
        if lab in theory:
            aa.axvline(theory[lab],color="#272727",ls="--",lw=.8,label="BPFO/BPFI/BSF")
            if lab=="B": aa.axvline(2*theory[lab],color="#767676",ls=":",lw=.8,label="2×BSF convention")
            aa.legend(fontsize=6)
    savefig(fig,out/"figures"/"mechanism_envelope_order")
    corr=raw[keep].corr().abs();fig,aa=plt.subplots(figsize=(9,7));im=aa.imshow(corr,cmap="viridis",vmin=0,vmax=1);fig.colorbar(im,ax=aa,label="|Pearson r|");aa.set_title("Selected-feature correlation");aa.set_xticks([]);aa.set_yticks([]);savefig(fig,out/"figures"/"feature_correlation")
    fm=raw.groupby(["file_id","label"])[keep].mean().reset_index();z=PCA(2).fit_transform(StandardScaler().fit_transform(fm[keep]));fig,aa=plt.subplots(figsize=(6,4));
    for lab in LABELS:m=fm.label==lab;aa.scatter(z[m,0],z[m,1],s=28,color=COLORS[lab],label=lab)
    aa.set(xlabel="PC1",ylabel="PC2",title="Full-source file-level PCA (descriptive)");aa.legend();savefig(fig,out/"figures"/"source_feature_pca")
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--data-root",type=Path,default=Path("数据集")/"数据集");ap.add_argument("--output-dir",type=Path,default=Path("outputs")/"q1");ap.add_argument("--window",type=int,default=16384);ap.add_argument("--hop",type=int,default=8192);a=ap.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True);rs=records(a.data_root); auditrows=audit(rs,a.output_dir);sel=selection(rs,a.output_dir);full=[r for r in rs if r[1] in ("48k_DE","48k_normal")];meta=[]
    for p,b,fs,lab in full:
        x,r,_,k=load(p,b);parts=p.parts;size=next((q for q in reversed(parts) if re.fullmatch(r"\d{4}",q)),"");pos=parts[parts.index("OR")+1] if lab=="OR" else "";meta.append({"file_id":p.stem,"file_path":p.as_posix(),"label":lab,"load":re.search(r"_(\d)(?:\D|$)",p.stem).group(1) if re.search(r"_(\d)(?:\D|$)",p.stem) else "","rpm":r or "","fault_size":size,"fault_position":pos,"sampling_rate":fs,"signal_length":len(x),"signal_variable":k})
    save_csv(a.output_dir/"source_metadata.csv",meta);raw=pd.DataFrame(rows_for(full,FS_MAIN,a.window,a.hop,None));save_csv(a.output_dir/"features_source_raw.csv",raw.to_dict("records"));cols=[c for c in raw if c not in ("file_id","file_path","window_id","label","branch","rpm")];X=raw[cols].replace([np.inf,-np.inf],np.nan).fillna(0);var=X.var();mi=mutual_info_classif(X,raw.label,random_state=2025);keep=[];quality=[]
    for i,c in enumerate(cols):
        why="" if var[c]>=1e-10 and not any(abs(X[c].corr(X[k]))>.95 for k in keep) else "near_zero_variance_or_correlated";keep.extend([c] if not why else []);cv=raw.groupby("file_id")[c].agg(lambda q:np.std(q)/max(abs(np.mean(q)),1e-12)).median();quality.append({"feature":c,"variance":var[c],"median_file_window_cv":cv,"mutual_information_auxiliary":mi[i],"selected":not bool(why),"drop_reason":why})
    save_csv(a.output_dir/"feature_quality.csv",quality);selected=raw[["file_id","file_path","window_id","label","branch","rpm",*keep]];save_csv(a.output_dir/"features_source_selected.csv",selected.to_dict("records"));(a.output_dir/"feature_names.json").write_text(json.dumps({"selected_features":keep,"rule":"near-zero variance removal and greedy |Pearson r| > 0.95 removal; MI is descriptive only"},ensure_ascii=False,indent=2),encoding="utf-8");figures(rs,raw,keep,a.output_dir);summary=f"# 第一问总结\n\n审计 MAT 文件：{len(auditrows)}。正式源域：48 kHz DE 全部故障文件加 48 kHz Normal，共 {len(meta)} 个文件；类别计数：{pd.DataFrame(meta).groupby('label').size().to_dict()}。\n\n候选域均在 12 kHz、0-6 kHz 公共视图下比较，MMD/PAD 仅作辅助。正式特征使用 48→32 kHz 抗混叠重采样、{a.window} 点窗口和 {a.hop} 点步长，保留 {len(keep)} 个低冗余特征。\n\n第一问不包含 A-P 分类、CORAL 或 DANN 结论；PCA、MMD 和 PAD 不构成目标准确率证据。\n";(a.output_dir/"q1_summary.md").write_text(summary,encoding="utf-8");print(json.dumps({"audit_files":len(auditrows),"full_source_files":len(meta),"windows":len(raw),"selected_features":len(keep)},ensure_ascii=False))
if __name__=="__main__":main()
