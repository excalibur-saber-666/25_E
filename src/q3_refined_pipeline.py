"""Leakage-aware refined Question 3 pipeline using Transfer-v2.

This is an independent experiment beside the frozen Transfer20/CORAL results.
It never reads a target label or reference answer.  The fixed formal method is
S56 class/file-balanced CORAL with the reference source encoder seed 2025;
other source variants and seeds are sensitivity evidence, not label voting.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

import q1_transfer_v2 as v2
import q2_pipeline as q2
import q3_pipeline as old

LABELS = q2.LABELS
SEEDS = (2025, 2026, 2027, 2028, 2029)
FORMAL_VARIANT = "s56_class_balanced"


def write_csv(path: Path, table: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False, encoding="utf-8-sig")


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=lambda x: x.item() if isinstance(x, np.generic) else str(x)), encoding="utf-8")


def load_inputs(output: Path, rpm: int = 600) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    source = pd.read_csv(output / "features_source_transfer_v2.csv")
    target = pd.read_csv(output / f"features_target_transfer_v2_rpm{rpm}.csv")
    names = json.loads((output / "transfer_v2_schema.json").read_text(encoding="utf-8"))["features"]
    if "label" in target:
        raise ValueError("Target labels are forbidden in Question 3 refined pipeline")
    for frame, domain in ((source, "source"), (target, "target")):
        required = {"file_id", "window_id", *names}
        if missing := required - set(frame.columns):
            raise ValueError(f"{domain} Transfer-v2 missing {sorted(missing)}")
        if not np.isfinite(frame[names].to_numpy(float)).all():
            raise ValueError(f"{domain} Transfer-v2 has NaN or Inf")
    if set(source.label.unique()) != set(LABELS) or source.file_id.nunique() != 56:
        raise ValueError("Expected exactly four labels across 56 source files")
    if target.file_id.nunique() != 16 or sorted(target.file_id.astype(str).unique()) != list("ABCDEFGHIJKLMNOP"):
        raise ValueError("Expected unlabelled target A--P exactly once")
    metadata = pd.read_csv("outputs/q1/source_metadata.csv")[["file_id", "load"]]
    source = source.merge(metadata, on="file_id", how="left", validate="many_to_one")
    if source.load.isna().any():
        raise ValueError("source load metadata is incomplete")
    return source, target, names


def class_file_window_weights(frame: pd.DataFrame) -> np.ndarray:
    """Each class total is 1/4, then each file/window is equal within class."""
    files = frame.groupby("file_id").label.first()
    class_file_count = files.value_counts().to_dict()
    per_file_windows = frame.groupby("file_id").size().to_dict()
    return np.asarray([1.0 / (len(LABELS) * class_file_count[row.label] * per_file_windows[row.file_id])
                       for _, row in frame.iterrows()], dtype=float)


def file_weights(frame: pd.DataFrame, variant: str) -> np.ndarray:
    if variant == "s56_file_balanced" or variant == "s16_balanced":
        return old.file_balanced_weights(frame.file_id)
    if variant == "s56_class_balanced":
        return class_file_window_weights(frame)
    raise ValueError(variant)


def weighted_covariance(values: np.ndarray, weights: np.ndarray, epsilon: float) -> tuple[np.ndarray, np.ndarray]:
    weights = np.asarray(weights, dtype=float); weights = weights / weights.sum()
    mean = np.sum(values * weights[:, None], axis=0)
    centered = values - mean
    return mean, centered.T @ (centered * weights[:, None]) + epsilon * np.eye(values.shape[1])


def _power(matrix: np.ndarray, exponent: float, epsilon: float) -> np.ndarray:
    value, vector = np.linalg.eigh(matrix)
    return (vector * np.maximum(value, epsilon) ** exponent) @ vector.T


def fit_coral(source_z: np.ndarray, source: pd.DataFrame, target_z: np.ndarray, target: pd.DataFrame, variant: str, epsilon: float = 1e-4) -> dict:
    """CORAL with declared source weighting; target is always file-balanced."""
    sm, cs = weighted_covariance(source_z, file_weights(source, variant), epsilon)
    tm, ct = weighted_covariance(target_z, old.file_balanced_weights(target.file_id), epsilon)
    return {"source_mean": sm, "target_mean": tm, "source_covariance": cs, "target_covariance": ct,
            "map": _power(cs, -.5, epsilon) @ _power(ct, .5, epsilon), "variant": variant, "epsilon": epsilon}


def apply_coral(values: np.ndarray, transform: dict) -> np.ndarray:
    values = (np.asarray(values, dtype=float) - transform["source_mean"]) @ transform["map"] + transform["target_mean"]
    if not np.isfinite(values).all():
        raise FloatingPointError("CORAL emitted NaN/Inf")
    return values


def train_encoder(source: pd.DataFrame, names: list[str], seed: int, epochs: int) -> tuple[q2.SourceMLP, StandardScaler]:
    q2.set_seed(seed)
    scaler = StandardScaler().fit(source[names].to_numpy(float), sample_weight=class_file_window_weights(source))
    x = scaler.transform(source[names].to_numpy(float)).astype(np.float32)
    y = np.asarray([LABELS.index(label) for label in source.label], dtype=np.int64)
    sampler = WeightedRandomSampler(torch.tensor(class_file_window_weights(source), dtype=torch.double), len(source), replacement=True,
                                   generator=torch.Generator().manual_seed(seed))
    loader = DataLoader(TensorDataset(torch.tensor(x), torch.tensor(y)), batch_size=64, sampler=sampler, num_workers=0)
    model = q2.SourceMLP(len(names), dropout=.10); optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            loss = F.cross_entropy(model(xb), yb)
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
    return model.eval(), scaler


def infer(model: q2.SourceMLP, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    zs, ps = [], []
    with torch.no_grad():
        for start in range(0, len(values), 256):
            x = torch.tensor(values[start:start + 256], dtype=torch.float32)
            z = model.encoder(x); zs.append(z.numpy()); ps.append(torch.softmax(model.classifier(z), dim=1).numpy())
    return np.vstack(zs), np.vstack(ps)


def fit_head(z: np.ndarray, source: pd.DataFrame, seed: int) -> LogisticRegression:
    head = LogisticRegression(max_iter=3000, class_weight=None, random_state=seed)
    head.fit(z, source.label.to_numpy(), sample_weight=class_file_window_weights(source))
    return head


def probabilities(head: LogisticRegression, z: np.ndarray) -> np.ndarray:
    out = np.zeros((len(z), len(LABELS)))
    out[:, [LABELS.index(str(c)) for c in head.classes_]] = head.predict_proba(z)
    if not np.allclose(out.sum(1), 1) or not np.isfinite(out).all():
        raise FloatingPointError("Invalid classifier probabilities")
    return out


def aggregate(frame: pd.DataFrame, p: np.ndarray, method: str) -> pd.DataFrame:
    return old.aggregate(frame, p, method)


def s16(source: pd.DataFrame) -> pd.DataFrame:
    wanted = {"B007_0", "B007_1", "B007_2", "B007_3", "IR007_0", "IR007_1", "IR007_2", "IR007_3",
              "OR007@6_0", "OR007@6_1", "OR007@6_2", "OR007@6_3"}
    normal = source.groupby("file_id").first().query("label == 'N'").sort_values("load").index.tolist()
    wanted.update(normal)
    out = source[source.file_id.isin(wanted)].copy()
    if out.file_id.nunique() != 16 or out.groupby("label").file_id.nunique().to_dict() != {label: 4 for label in LABELS}:
        raise AssertionError("S16 selection must be four fixed source files per class")
    return out


def retention(source: pd.DataFrame, target: pd.DataFrame, names: list[str], epochs: int) -> pd.DataFrame:
    """LOLO retention; every fold refits its encoder without held-out source files."""
    rows = []
    for load in sorted(source.load.unique()):
        tr, te = source[source.load != load], source[source.load == load]
        model, scaler = train_encoder(tr, names, 5000 + int(load), epochs)
        ztr, _ = infer(model, scaler.transform(tr[names].to_numpy(float)))
        zte, p_mlp = infer(model, scaler.transform(te[names].to_numpy(float)))
        ztarget, _ = infer(model, scaler.transform(target[names].to_numpy(float)))
        candidates = [("source_mlp", p_mlp)]
        for variant in ("s56_file_balanced", "s56_class_balanced"):
            transform = fit_coral(ztr, tr, ztarget, target, variant)
            head = fit_head(apply_coral(ztr, transform), tr, 7000 + int(load))
            candidates.append((variant, probabilities(head, apply_coral(zte, transform))))
        truth = te.groupby("file_id").label.first()
        for method, p in candidates:
            table = aggregate(te, p, method).set_index("file_id").loc[truth.index]
            for fid, label in truth.items():
                rows.append({"load": int(load), "method": method, "file_id": fid, "true_label": label,
                             "predicted_label": table.loc[fid, "predicted_label"]})
    raw = pd.DataFrame(rows); summary = []
    for method, part in raw.groupby("method"):
        y, pred = part.true_label.to_numpy(), part.predicted_label.to_numpy()
        summary.append({"method": method, "macro_f1": f1_score(y, pred, labels=LABELS, average="macro", zero_division=0),
                        "balanced_accuracy": balanced_accuracy_score(y, pred),
                        **{f"recall_{x}": v for x, v in zip(LABELS, recall_score(y, pred, labels=LABELS, average=None, zero_division=0))},
                        "unit": "held-out source raw MAT file"})
    return raw, pd.DataFrame(summary)


def s16_retention(source: pd.DataFrame, target: pd.DataFrame, names: list[str], epochs: int) -> tuple[pd.DataFrame, dict]:
    """The same LOLO retention protocol for the fixed 4-by-4 S16 sensitivity set."""
    rows = []
    for load in sorted(source.load.unique()):
        tr, te = source[source.load != load], source[source.load == load]
        model, scaler = train_encoder(tr, names, 8500 + int(load), epochs)
        ztr, _ = infer(model, scaler.transform(tr[names].to_numpy(float)))
        zte, _ = infer(model, scaler.transform(te[names].to_numpy(float)))
        ztarget, _ = infer(model, scaler.transform(target[names].to_numpy(float)))
        transform = fit_coral(ztr, tr, ztarget, target, "s16_balanced")
        head = fit_head(apply_coral(ztr, transform), tr, 8600 + int(load))
        table = aggregate(te, probabilities(head, apply_coral(zte, transform)), "s16_balanced").set_index("file_id")
        truth = te.groupby("file_id").label.first()
        for fid, label in truth.items(): rows.append({"load": int(load), "method": "s16_balanced", "file_id": fid, "true_label": label, "predicted_label": table.loc[fid, "predicted_label"]})
    raw = pd.DataFrame(rows); y, pred = raw.true_label.to_numpy(), raw.predicted_label.to_numpy()
    return raw, {"method": "s16_balanced", "macro_f1": f1_score(y, pred, labels=LABELS, average="macro", zero_division=0),
                 "balanced_accuracy": balanced_accuracy_score(y, pred),
                 **{f"recall_{x}": v for x, v in zip(LABELS, recall_score(y, pred, labels=LABELS, average=None, zero_division=0))},
                 "unit": "held-out source raw MAT file (fixed S16 subset)"}


def stability(frame: pd.DataFrame, p: np.ndarray, seed: int = 2025) -> dict[str, float]:
    return old.subsample_agreement(frame, p, repeats=20, fraction=.8, seed=seed)


def loto(source_z: np.ndarray, source: pd.DataFrame, target_z: np.ndarray, target: pd.DataFrame, variant: str) -> pd.DataFrame:
    rows = []
    for file_id, idx in target.groupby("file_id", sort=True).indices.items():
        keep = target.file_id.to_numpy() != file_id
        transform = fit_coral(source_z, source, target_z[keep], target.loc[keep], variant)
        head = fit_head(apply_coral(source_z, transform), source, 2025)
        rows.append(aggregate(target.iloc[np.asarray(idx)], probabilities(head, target_z[np.asarray(idx)]), "coral_loto"))
    return pd.concat(rows, ignore_index=True)


def geometry(source: pd.DataFrame, target: pd.DataFrame, zs: np.ndarray, zt: np.ndarray, transform: dict, final: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Source-standardized latent geometry; no target labels are used."""
    scaler = StandardScaler().fit(zs, sample_weight=class_file_window_weights(source))
    sb, tb = scaler.transform(zs), scaler.transform(zt)
    sa = scaler.transform(apply_coral(zs, transform))
    centers_before = {label: sb[source.label.to_numpy() == label].mean(0) for label in LABELS}
    centers_after = {label: sa[source.label.to_numpy() == label].mean(0) for label in LABELS}
    file_before = {str(fid): sb[np.asarray(idx)].mean(0) for fid, idx in source.groupby("file_id").indices.items()}
    rows = []
    for fid, idx in target.groupby("file_id", sort=True).indices.items():
        point = tb[np.asarray(idx)].mean(0)
        before = min(centers_before, key=lambda x: np.linalg.norm(point - centers_before[x]))
        after = min(centers_after, key=lambda x: np.linalg.norm(point - centers_after[x]))
        near_file = min(file_before, key=lambda x: np.linalg.norm(point - file_before[x]))
        row = {"file_id": str(fid), **{f"distance_to_{label}": float(np.linalg.norm(point - centers_after[label])) for label in LABELS},
               "nearest_source_class": after, "nearest_source_file": near_file, "before_coral_nearest_class": before,
               "after_coral_nearest_class": after}
        row.update(final.set_index("file_id").loc[str(fid), ["candidate_label", "probability_margin", "normalized_entropy"]].to_dict())
        rows.append(row)
    movement = pd.DataFrame([{"label": label, "center_movement_l2": float(np.linalg.norm(centers_after[label] - centers_before[label]))} for label in LABELS])
    return pd.DataFrame(rows), movement


def raw_feature_coral(source: pd.DataFrame, target: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    scaler = StandardScaler().fit(source[names].to_numpy(float), sample_weight=class_file_window_weights(source))
    xs, xt = scaler.transform(source[names]), scaler.transform(target[names])
    transform = fit_coral(xs, source, xt, target, "s56_class_balanced")
    head = fit_head(apply_coral(xs, transform), source, 2025)
    return aggregate(target, probabilities(head, xt), "raw_feature_coral")


def feature_ablation(source: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    """Fixed source-only LOLO ablations; they do not inspect target predictions."""
    sets = {"full": names, "no_absolute_hz": [x for x in names if not x.startswith("env_")],
            "no_order": [x for x in names if "order" not in x], "no_envelope": [x for x in names if not x.startswith("env_")]}
    rows = []
    files = source.groupby("file_id").first()[["label", "load"]].join(source.groupby("file_id")[names].mean())
    for name, cols in sets.items():
        predictions, truth = [], []
        for load in sorted(files.load.unique()):
            train, test = files[files.load != load], files[files.load == load]
            clf = StandardScaler().fit(train[cols]); xtr = clf.transform(train[cols]); xte = clf.transform(test[cols])
            weights = train.label.map(lambda value: 1 / train.label.value_counts()[value]).to_numpy()
            head = LogisticRegression(max_iter=3000, class_weight=None, random_state=2025).fit(xtr, train.label, sample_weight=weights)
            predictions.extend(head.predict(xte)); truth.extend(test.label)
        rows.append({"ablation": name, "feature_count": len(cols), "macro_f1": f1_score(truth, predictions, labels=LABELS, average="macro", zero_division=0),
                     "balanced_accuracy": balanced_accuracy_score(truth, predictions), "evaluation": "source file-level LOLO logistic probe only"})
    return pd.DataFrame(rows)


def _sinkhorn(cost: np.ndarray, source_mass: np.ndarray, target_mass: np.ndarray, epsilon: float = .05, iterations: int = 300) -> np.ndarray:
    """Balanced entropic OT in log-free form after robust cost normalization."""
    cost = np.asarray(cost, dtype=float)
    kernel = np.exp(-np.clip(cost / epsilon, 0, 700)) + 1e-300
    u = np.ones(len(source_mass)); v = np.ones(len(target_mass))
    for _ in range(iterations):
        u = source_mass / np.maximum(kernel @ v, 1e-300)
        v = target_mass / np.maximum(kernel.T @ u, 1e-300)
    plan = (u[:, None] * kernel) * v[None, :]
    if not np.isfinite(plan).all() or plan.sum() <= 0:
        raise FloatingPointError("Sinkhorn transport is non-finite")
    return plan / plan.sum()


def fit_class_regularized_ot(source_z: np.ndarray, source: pd.DataFrame, target_z: np.ndarray, target: pd.DataFrame,
                             epsilon: float = .05, class_regularization: float = .25) -> dict:
    """Unlabelled-target class-regularized OT with a source-to-target affine map.

    Source labels enter only through an entropy penalty on each target point's
    incoming source-class mass.  No target pseudo-label is fixed or optimized.
    """
    a, b = class_file_window_weights(source), old.file_balanced_weights(target.file_id)
    base = np.sum((source_z[:, None, :] - target_z[None, :, :]) ** 2, axis=2)
    scale = max(float(np.median(base[base > 0])), 1e-12); cost = base / scale
    class_index = np.asarray([LABELS.index(label) for label in source.label])
    for _ in range(5):
        plan = _sinkhorn(cost, a, b, epsilon)
        class_mass = np.vstack([plan[class_index == k].sum(0) for k in range(len(LABELS))])
        class_mass /= np.maximum(class_mass.sum(0, keepdims=True), 1e-12)
        # Reweighted penalty prefers a coherent source class per target point,
        # while leaving the class identity entirely data-driven and unlabelled.
        cost = base / scale + class_regularization * (-np.log(np.maximum(class_mass[class_index], 1e-8)))
    barycenter = (plan @ target_z) / np.maximum(plan.sum(1, keepdims=True), 1e-12)
    augmented = np.c_[source_z, np.ones(len(source_z))]
    weights = np.sqrt(a)[:, None]
    ridge = 1e-2 * np.eye(augmented.shape[1]); ridge[-1, -1] = 0.0
    affine = np.linalg.solve((augmented * weights).T @ (augmented * weights) + ridge,
                             (augmented * weights).T @ (barycenter * weights))
    return {"affine": affine, "plan": plan, "epsilon": epsilon, "class_regularization": class_regularization,
            "source_class_mass_entropy": float(np.mean(-np.sum(class_mass * np.log(np.maximum(class_mass, 1e-12)), axis=0) / math.log(len(LABELS))))}


def apply_ot(values: np.ndarray, transform: dict) -> np.ndarray:
    out = np.c_[np.asarray(values, dtype=float), np.ones(len(values))] @ transform["affine"]
    if not np.isfinite(out).all(): raise FloatingPointError("OT affine map emitted NaN/Inf")
    return out


def ot_retention(source: pd.DataFrame, target: pd.DataFrame, names: list[str], epochs: int) -> tuple[pd.DataFrame, dict]:
    rows = []
    for load in sorted(source.load.unique()):
        tr, te = source[source.load != load], source[source.load == load]
        model, scaler = train_encoder(tr, names, 9000 + int(load), epochs)
        ztr, _ = infer(model, scaler.transform(tr[names].to_numpy(float)))
        zte, _ = infer(model, scaler.transform(te[names].to_numpy(float)))
        transform = fit_class_regularized_ot(ztr, tr, zte, te)
        head = fit_head(apply_ot(ztr, transform), tr, 9100 + int(load))
        table = aggregate(te, probabilities(head, zte), "class_regularized_ot")
        truth = te.groupby("file_id").label.first()
        table = table.set_index("file_id").loc[truth.index]
        for fid, label in truth.items(): rows.append({"load": int(load), "method": "class_regularized_ot", "file_id": fid, "true_label": label, "predicted_label": table.loc[fid, "predicted_label"]})
    raw = pd.DataFrame(rows); y, pred = raw.true_label.to_numpy(), raw.predicted_label.to_numpy()
    return raw, {"method": "class_regularized_ot", "macro_f1": f1_score(y, pred, labels=LABELS, average="macro", zero_division=0),
                 "balanced_accuracy": balanced_accuracy_score(y, pred),
                 **{f"recall_{x}": v for x, v in zip(LABELS, recall_score(y, pred, labels=LABELS, average=None, zero_division=0))},
                 "unit": "held-out source raw MAT file"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/q3_refined"))
    parser.add_argument("--data-root", type=Path, default=Path("数据集") / "数据集")
    parser.add_argument("--epochs", type=int, default=45)
    parser.add_argument("--retention-epochs", type=int, default=30)
    args = parser.parse_args()
    if args.epochs < 1 or args.retention_epochs < 1: raise ValueError("epochs must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "models").mkdir(parents=True, exist_ok=True)
    # Source is extracted once; RPM variants only regenerate unlabelled target features.
    if not (args.output_dir / "features_source_transfer_v2.csv").exists():
        v2.build(args.data_root, args.output_dir, 600, write_source=True)
    for rpm in (570, 600, 630):
        if not (args.output_dir / f"features_target_transfer_v2_rpm{rpm}.csv").exists():
            v2.build(args.data_root, args.output_dir, rpm, write_source=False)
    source, target, names = load_inputs(args.output_dir, 600)
    audit = {"input": "Transfer-v2 only", "frozen_q1_q2_modified": False, "target_label_column_present": False,
             "target_labels_used_for_fit_loss_selection_or_metric": False, "source_files": 56, "target_files": 16,
             "formal_method_predeclared": "S56 class/file-balanced CORAL; fixed seed 2025", "features": names}
    write_json(args.output_dir / "input_audit.json", audit)

    # The five encoders are retained for an instability audit.  Seed 2025 is a fixed reference, never the best seed.
    seed_rows, ot_seed_rows, seed_tables, reference = [], [], [], None
    for seed in SEEDS:
        model, scaler = train_encoder(source, names, seed, args.epochs)
        zs, _ = infer(model, scaler.transform(source[names].to_numpy(float)))
        zt, _ = infer(model, scaler.transform(target[names].to_numpy(float)))
        transform = fit_coral(zs, source, zt, target, FORMAL_VARIANT)
        head = fit_head(apply_coral(zs, transform), source, seed)
        p = probabilities(head, zt); table = aggregate(target, p, FORMAL_VARIANT); table["seed"] = seed
        seed_tables.append(table)
        for _, row in table.iterrows(): seed_rows.append({"file_id": row.file_id, "seed": seed, "predicted_label": row.predicted_label,
                                                           "confidence": row.confidence, "margin": row.probability_margin, "entropy": row.normalized_entropy})
        ot_tf = fit_class_regularized_ot(zs, source, zt, target)
        ot_head = fit_head(apply_ot(zs, ot_tf), source, seed)
        ot_by_seed = aggregate(target, probabilities(ot_head, zt), "class_regularized_ot")
        for _, row in ot_by_seed.iterrows(): ot_seed_rows.append({"file_id": row.file_id, "seed": seed, "predicted_label": row.predicted_label,
                                                                   "confidence": row.confidence, "margin": row.probability_margin, "entropy": row.normalized_entropy})
        if seed == 2025: reference = (model, scaler, zs, zt, transform, p, table.drop(columns="seed"))
    seed_table = pd.DataFrame(seed_rows); write_csv(args.output_dir / "encoder_seed_predictions.csv", seed_table)
    consensus = []
    for fid, part in seed_table.groupby("file_id", sort=True):
        vote = part.predicted_label.mode().iloc[0]
        consensus.append({"file_id": fid, "majority_label": vote, "agreement_ratio": float((part.predicted_label == vote).mean()),
                          "mean_confidence": float(part.confidence.mean()), "mean_margin": float(part.margin.mean())})
    consensus = pd.DataFrame(consensus); write_csv(args.output_dir / "encoder_seed_consensus.csv", consensus)
    ot_seed_table = pd.DataFrame(ot_seed_rows); write_csv(args.output_dir / "class_regularized_ot_seed_predictions.csv", ot_seed_table)
    ot_consensus = []
    for fid, part in ot_seed_table.groupby("file_id", sort=True):
        vote = part.predicted_label.mode().iloc[0]
        ot_consensus.append({"file_id": fid, "majority_label": vote, "agreement_ratio": float((part.predicted_label == vote).mean()),
                             "mean_confidence": float(part.confidence.mean()), "mean_margin": float(part.margin.mean())})
    ot_consensus = pd.DataFrame(ot_consensus); write_csv(args.output_dir / "class_regularized_ot_seed_consensus.csv", ot_consensus)
    model, scaler, zs, zt, formal_transform, formal_p, formal = reference
    torch.save(model.encoder.state_dict(), args.output_dir / "models" / "transfer_v2_seed2025_encoder.pth")
    torch.save(model.classifier.state_dict(), args.output_dir / "models" / "transfer_v2_seed2025_classifier.pth")

    variants, comparisons = {}, []
    for variant, subset in (("s56_file_balanced", source), ("s56_class_balanced", source), ("s16_balanced", s16(source))):
        # S16 is an explicit sensitivity source; its encoder is refit rather than borrowing S56 weights.
        if variant == FORMAL_VARIANT:
            zsrc, ztarget = zs, zt
        else:
            vm, vs = train_encoder(subset, names, 2025, args.epochs)
            zsrc, _ = infer(vm, vs.transform(subset[names].to_numpy(float)))
            ztarget, _ = infer(vm, vs.transform(target[names].to_numpy(float)))
        tf = fit_coral(zsrc, subset, ztarget, target, variant)
        h = fit_head(apply_coral(zsrc, tf), subset, 2025)
        p = probabilities(h, ztarget); table = aggregate(target, p, variant)
        variants[variant] = (subset, zsrc, ztarget, tf, p, table)
        dm = old.domain_metrics(subset, target, apply_coral(zsrc, tf), ztarget)
        comparisons.append({"method": variant, "source_files": int(subset.file_id.nunique()), **dm,
                            **old.collapse(table), "mean_confidence": float(table.confidence.mean()), "mean_margin": float(table.probability_margin.mean())})
        write_csv(args.output_dir / f"{variant}_predictions.csv", table)
    write_csv(args.output_dir / "domain_metrics.csv", pd.DataFrame(comparisons))

    raw, retention_summary = retention(source, target, names, args.retention_epochs)
    ot_transform = fit_class_regularized_ot(zs, source, zt, target)
    ot_head = fit_head(apply_ot(zs, ot_transform), source, 2025)
    ot_p = probabilities(ot_head, zt); ot_table = aggregate(target, ot_p, "class_regularized_ot")
    write_csv(args.output_dir / "class_regularized_ot_predictions.csv", ot_table)
    ot_raw, ot_summary = ot_retention(source, target, names, args.retention_epochs)
    s16_raw, s16_summary = s16_retention(s16(source), target, names, args.retention_epochs)
    raw = pd.concat([raw, ot_raw, s16_raw], ignore_index=True)
    retention_summary = pd.concat([retention_summary, pd.DataFrame([ot_summary, s16_summary])], ignore_index=True)
    write_csv(args.output_dir / "source_retention_predictions.csv", raw); write_csv(args.output_dir / "source_retention.csv", retention_summary)
    retention_map = retention_summary.set_index("method").macro_f1.to_dict()
    for row in comparisons:
        row["source_retention_macro_f1"] = retention_map.get(row["method"], np.nan)
    ot_metric = old.domain_metrics(source, target, apply_ot(zs, ot_transform), zt)
    comparisons.append({"method": "class_regularized_ot", "source_files": 56, **ot_metric, **old.collapse(ot_table),
                        "mean_confidence": float(ot_table.confidence.mean()), "mean_margin": float(ot_table.probability_margin.mean()),
                        "source_retention_macro_f1": retention_map["class_regularized_ot"],
                        "source_class_mass_entropy": ot_transform["source_class_mass_entropy"]})
    write_csv(args.output_dir / "domain_metrics.csv", pd.DataFrame(comparisons))
    write_csv(args.output_dir / "coral_source_variant_comparison.csv", pd.DataFrame(comparisons))

    raw_table = raw_feature_coral(source, target, names); write_csv(args.output_dir / "raw_feature_coral_predictions.csv", raw_table)
    write_csv(args.output_dir / "transfer_v2_ablation.csv", feature_ablation(source, names))
    # CORAL crossed the predeclared -0.05 source-retention boundary.  OT becomes
    # the main candidate only if it clears that same source-only gate.
    ot_selected = retention_map["class_regularized_ot"] - retention_map["source_mlp"] >= -.05
    final_method = "Class-Regularized OT Transfer-v2 S56" if ot_selected else "CORAL Transfer-v2 S56 class/file-balanced"
    chosen_p, chosen_table = (ot_p, ot_table) if ot_selected else (formal_p, formal)
    final_consensus = ot_consensus if ot_selected else consensus
    sub = stability(target, chosen_p)
    coral_loto = loto(zs, source, zt, target, FORMAL_VARIANT); write_csv(args.output_dir / "coral_leave_one_target_out.csv", coral_loto)
    if ot_selected:
        loto_rows = []
        for file_id, idx in target.groupby("file_id", sort=True).indices.items():
            keep = target.file_id.to_numpy() != file_id
            tmp = fit_class_regularized_ot(zs, source, zt[keep], target.loc[keep])
            head = fit_head(apply_ot(zs, tmp), source, 2025)
            loto_rows.append(aggregate(target.iloc[np.asarray(idx)], probabilities(head, zt[np.asarray(idx)]), "class_regularized_ot_loto"))
        loto_table = pd.concat(loto_rows, ignore_index=True)
    else:
        loto_table = coral_loto
    write_csv(args.output_dir / "final_method_leave_one_target_out.csv", loto_table)
    loto_map = loto_table.set_index("file_id").predicted_label.to_dict()

    # RPM sensitivity reuses the same fixed source encoder and formal CORAL rule; RPM is not tuned.
    rpm_tables = {}
    for rpm in (570, 600, 630):
        _, tr, _ = load_inputs(args.output_dir, rpm)
        zr, _ = infer(model, scaler.transform(tr[names].to_numpy(float)))
        if ot_selected:
            tf = fit_class_regularized_ot(zs, source, zr, tr); hp = fit_head(apply_ot(zs, tf), source, 2025)
            rpm_tables[rpm] = aggregate(tr, probabilities(hp, zr), "class_regularized_ot").set_index("file_id")
        else:
            tf = fit_coral(zs, source, zr, tr, FORMAL_VARIANT); hp = fit_head(apply_coral(zs, tf), source, 2025)
            rpm_tables[rpm] = aggregate(tr, probabilities(hp, zr), FORMAL_VARIANT).set_index("file_id")
    rpm_rows = []
    for fid in list("ABCDEFGHIJKLMNOP"):
        labels = [rpm_tables[x].loc[fid, "predicted_label"] for x in (570, 600, 630)]
        conf = [rpm_tables[x].loc[fid, "confidence"] for x in (570, 600, 630)]
        margin = [rpm_tables[x].loc[fid, "probability_margin"] for x in (570, 600, 630)]
        rpm_rows.append({"file_id": fid, "prediction_570": labels[0], "prediction_600": labels[1], "prediction_630": labels[2],
                         "agreement_ratio": float(np.mean(np.asarray(labels) == labels[1])), "confidence_variation": float(np.ptp(conf)), "margin_variation": float(np.ptp(margin))})
    rpm_sensitivity = pd.DataFrame(rpm_rows); write_csv(args.output_dir / "rpm_sensitivity.csv", rpm_sensitivity)

    final = chosen_table.rename(columns={"predicted_label": "candidate_label"}).copy()
    final.insert(1, "final_method", final_method)
    final["encoder_agreement"] = final.file_id.map(final_consensus.set_index("file_id").agreement_ratio)
    final["rpm_agreement"] = final.file_id.map(rpm_sensitivity.set_index("file_id").agreement_ratio)
    final["subsample_agreement"] = final.file_id.map(sub)
    final["loto_agreement"] = final.apply(lambda row: row.candidate_label == loto_map[row.file_id], axis=1)
    final["review_required"] = ((final.encoder_agreement < .8) | (final.rpm_agreement < 1.0) | (final.subsample_agreement < .75) |
                                (~final.loto_agreement) | (final.probability_margin < .05) | (final.normalized_entropy > .8))
    final["review_reason"] = final.apply(lambda r: ";".join(name for name, failed in (("encoder", r.encoder_agreement < .8), ("rpm", r.rpm_agreement < 1), ("subsample", r.subsample_agreement < .75), ("loto", not r.loto_agreement), ("low_margin", r.probability_margin < .05), ("high_entropy", r.normalized_entropy > .8)) if failed) or "stable", axis=1)
    expected = np.asarray(LABELS)[final[[f"prob_{x}" for x in LABELS]].to_numpy().argmax(1)]
    if not np.array_equal(expected, final.candidate_label.to_numpy()): raise AssertionError("final candidate does not match final probabilities")
    write_csv(args.output_dir / "target_predictions_final.csv", final)
    coral_geo_input = formal.rename(columns={"predicted_label": "candidate_label"})
    geo, movement = geometry(source, target, zs, zt, formal_transform, coral_geo_input)
    geo = geo.rename(columns={"candidate_label": "coral_predicted_label"})
    write_csv(args.output_dir / "target_class_geometry.csv", geo); write_csv(args.output_dir / "source_class_center_movement.csv", movement)

    old_coral = pd.read_csv("outputs/q3/coral_predictions.csv", encoding="utf-8-sig")[["file_id", "predicted_label"]].rename(columns={"predicted_label": "old_transfer20_coral"})
    changes = old_coral.merge(final[["file_id", "candidate_label", "encoder_agreement", "rpm_agreement", "review_required"]], on="file_id")
    changes["changed"] = changes.old_transfer20_coral != changes.candidate_label
    write_csv(args.output_dir / "transfer20_vs_v2_label_changes.csv", changes)
    verification = {"tests_preconditions": {"target_label_absent": True, "finite_features": True, "final_argmax_consistent": True, "target_file_count": 16},
                    "target_accuracy_reported": False, "target_labels_used": False, "formal_variant": FORMAL_VARIANT, "final_method": final_method,
                    "source_retention_negative_transfer_vs_source_mlp": float(retention_map[FORMAL_VARIANT] - retention_map["source_mlp"]),
                    "ot_source_retention_minus_source_mlp": float(retention_map["class_regularized_ot"] - retention_map["source_mlp"]),
                    "notes": "MMD/PAD and stability are diagnostics, not target performance."}
    write_json(args.output_dir / "verification.json", verification)
    dist = final.candidate_label.value_counts().reindex(LABELS, fill_value=0).to_dict()
    comp = pd.DataFrame(comparisons).set_index("method")
    old_metrics = pd.read_csv("outputs/q3/domain_metrics.csv", encoding="utf-8-sig").set_index("method")
    encoder_stability = float(final.encoder_agreement.mean()); rpm_stability = float(final.rpm_agreement.mean())
    table_lines = ["| Method | Features | Source variant | Source retention F1 | MMD | Target class distribution | Encoder stability | RPM stability | Collapse |",
                   "|---|---|---|---:|---:|---|---:|---:|---|"]
    table_lines += [
        f"| Source-only | Transfer20 | S56 | 0.8773 | {old_metrics.loc['source_mlp','mmd_file_level']:.4f} | baseline only | NA | NA | NA |",
        f"| Old CORAL | Transfer20 | S56 | 0.8315 | {old_metrics.loc['coral','mmd_file_level']:.4f} | baseline only | NA | NA | no |",
        f"| CORAL | Transfer-v2 | S56 file-balanced | {retention_map['s56_file_balanced']:.4f} | {comp.loc['s56_file_balanced','mmd_file_level']:.4f} | {comp.loc['s56_file_balanced','class_count']} | NA | NA | {comp.loc['s56_file_balanced','collapse_warning']} |",
        f"| CORAL | Transfer-v2 | S56 class-balanced | {retention_map['s56_class_balanced']:.4f} | {comp.loc['s56_class_balanced','mmd_file_level']:.4f} | {comp.loc['s56_class_balanced','class_count']} | {consensus.agreement_ratio.mean():.3f} | NA | {comp.loc['s56_class_balanced','collapse_warning']} |",
        f"| CORAL | Transfer-v2 | S16 | {retention_map['s16_balanced']:.4f} | {comp.loc['s16_balanced','mmd_file_level']:.4f} | {comp.loc['s16_balanced','class_count']} | NA | NA | {comp.loc['s16_balanced','collapse_warning']} |",
        f"| Raw-feature CORAL | Transfer-v2 | S56 class-balanced | NA | NA | {raw_table.predicted_label.value_counts().reindex(LABELS, fill_value=0).to_dict()} | NA | NA | {old.collapse(raw_table)['collapse_warning']} |",
        f"| Class-Reg OT | Transfer-v2 | S56 class-balanced | {retention_map['class_regularized_ot']:.4f} | {comp.loc['class_regularized_ot','mmd_file_level']:.4f} | {comp.loc['class_regularized_ot','class_count']} | {encoder_stability:.3f} | {rpm_stability:.3f} | {comp.loc['class_regularized_ot','collapse_warning']} |"]
    summary = f"""# 第三问迁移链路定向修正结果\n\n最终唯一候选方法为 **{final_method}**。此选择仅由预先设定的源域 LOLO 保持门槛触发：CORAL 相对 Source-MLP 低于 -0.05 时运行 Class-Regularized OT，OT 只有达到同一门槛才取代 CORAL；没有使用 A–P 真值、PDF 或参考答案。旧 Transfer20 CORAL 保留在 `outputs/q3/` 作为基线。\n\nTransfer-v2 使用每窗 8 个名义转数、256 samples/revolution（2048 点）的近似恒速角域重采样；因没有转速脉冲，这不是严格 order tracking。角域 Welch 分辨率固定为 0.25 order；包络特征采用 500–2000、2000–4000、4000–8000 Hz 三个固定频带。\n\n{chr(10).join(table_lines)}\n\n- CORAL 源域 LOLO Macro-F1：{retention_map[FORMAL_VARIANT]:.4f}；Class-Reg OT：{retention_map['class_regularized_ot']:.4f}；Source-MLP：{retention_map['source_mlp']:.4f}。这些都是源域保持验证，不是目标准确率。\n- A–P 候选类别分布：{dist}；需复核文件数：{int(final.review_required.sum())}/16。\n- 5 encoder、570/600/630 rpm、LOTO 和窗口子采样均已生成；它们只作为可靠性证据。\n\n不得把 MMD/PAD 降低、置信度或候选标签写成目标域 accuracy/F1/recall。最终方法仍有 {int(final.review_required.sum())}/16 个文件需要复核，因此第三问只能交付候选标签，**尚不可声称完全封版或开始第四问**。\n"""
    (args.output_dir / "q3_refined_summary.md").write_text(summary, encoding="utf-8")
    write_json(args.output_dir / "q3_refined_config.json", {"seeds": SEEDS, "epochs": args.epochs, "retention_epochs": args.retention_epochs,
               "formal_variant": FORMAL_VARIANT, "final_method": final_method, "rpm_sensitivity": [570, 600, 630], "coral_epsilon": 1e-4,
               "target_labels_used": False, "statistical_unit": "raw MAT file"})
    print(f"Refined Question 3 outputs written to {args.output_dir}")


if __name__ == "__main__":
    main()
