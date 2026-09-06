"""Leakage-aware refined Question 3 pipeline using Transfer-v2.

This is an independent experiment beside the frozen Transfer20/CORAL results.
It never reads a target label or reference answer.  The fixed formal method is
S56 class/file-balanced CORAL with the reference source encoder seed 2025;
other source variants and seeds are sensitivity evidence, not label voting.
"""
from __future__ import annotations

import argparse
import io
import json
import math
import subprocess
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


def pre_fair_prediction_baseline(output: Path) -> pd.DataFrame | None:
    """Reporting-only snapshot of the committed pre-fair-benchmark candidates.

    It is never supplied to a model, metric, selection rule, or transform.
    """
    backup = output / "target_predictions_pre_fair_benchmark.csv"
    if backup.exists():
        return pd.read_csv(backup)[["file_id", "candidate_label"]].rename(columns={"candidate_label": "before_fair_benchmark"})
    try:
        payload = subprocess.run(["git", "show", "HEAD:outputs/q3_refined/target_predictions_final.csv"], check=True, capture_output=True, text=True, encoding="utf-8").stdout
        frame = pd.read_csv(io.StringIO(payload))
        backup.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(backup, index=False, encoding="utf-8-sig")
        return frame[["file_id", "candidate_label"]].rename(columns={"candidate_label": "before_fair_benchmark"})
    except (OSError, subprocess.CalledProcessError, pd.errors.EmptyDataError):
        return None


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


def ablation_sets(names: list[str]) -> dict[str, list[str]]:
    """Predeclared feature sets; only env_* depends on fixed Hz band edges."""
    sets = {"full": list(names),
            "no_absolute_hz": [x for x in names if not x.startswith("env_")],
            "no_envelope": [x for x in names if not (x.startswith("env_") or x.startswith("envelope_"))],
            "no_order": [x for x in names if "order" not in x]}
    signatures = {name: tuple(columns) for name, columns in sets.items()}
    if len(set(signatures.values())) != len(sets) or any(not columns for columns in sets.values()):
        raise AssertionError("Transfer-v2 ablation sets must be non-empty and distinct")
    return sets


def feature_ablation(source: pd.DataFrame, names: list[str]) -> tuple[pd.DataFrame, dict]:
    """Fixed source-only LOLO probes; A--P is never read by this function."""
    sets = ablation_sets(names)
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
    schema = {"sets": sets, "feature_counts": {name: len(columns) for name, columns in sets.items()},
              "sets_are_distinct": True, "rule": "no_absolute_hz removes fixed-Hz env_* features; no_envelope additionally removes full-band envelope descriptors; no_order removes every order-domain descriptor."}
    return pd.DataFrame(rows), schema


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


def legacy_main() -> None:
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


def _method_probabilities(method: str, source: pd.DataFrame, target: pd.DataFrame, zs: np.ndarray, zt: np.ndarray,
                          source_mlp_probabilities: np.ndarray, seed: int) -> tuple[np.ndarray, dict | None]:
    """Adapt with target features only; target labels are structurally unavailable."""
    if "label" in target.columns:
        raise ValueError("Surrogate/real target labels must not enter adaptation")
    if method == "source_only":
        return source_mlp_probabilities, None
    if method == "coral":
        transform = fit_coral(zs, source, zt, target, FORMAL_VARIANT)
        head = fit_head(apply_coral(zs, transform), source, seed)
        return probabilities(head, zt), transform
    if method == "class_regularized_ot":
        transform = fit_class_regularized_ot(zs, source, zt, target)
        head = fit_head(apply_ot(zs, transform), source, seed)
        return probabilities(head, zt), transform
    raise ValueError(method)


def surrogate_uda_benchmark(source: pd.DataFrame, names: list[str], epochs: int) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Fair LOLO-as-unlabelled-target benchmark, with labels opened only after inference."""
    rows, fold_rows, protocol = [], [], []
    for load in sorted(source.load.unique()):
        train = source[source.load != load].copy()
        held_out_with_labels = source[source.load == load].copy()
        truth = held_out_with_labels.groupby("file_id").label.first().copy()
        pseudo_target = held_out_with_labels.drop(columns="label")
        if "label" in pseudo_target.columns:
            raise AssertionError("held-out labels leaked into adaptation frame")
        model, scaler = train_encoder(train, names, 12000 + int(load), epochs)
        ztrain, _ = infer(model, scaler.transform(train[names].to_numpy(float)))
        ztarget, source_p = infer(model, scaler.transform(pseudo_target[names].to_numpy(float)))
        protocol.append({"held_out_load": int(load), "adaptation_target_has_label": False,
                         "scaler_fit_files": int(train.file_id.nunique()), "pseudo_target_files": int(pseudo_target.file_id.nunique())})
        for method in ("source_only", "coral", "class_regularized_ot"):
            p, _ = _method_probabilities(method, train, pseudo_target, ztrain, ztarget, source_p, 13000 + int(load))
            table = aggregate(pseudo_target, p, method).set_index("file_id").loc[truth.index].reset_index()
            table["held_out_load"] = int(load)
            table["true_label"] = table.file_id.map(truth)
            table["margin"] = table["probability_margin"]; table["entropy"] = table["normalized_entropy"]
            rows.extend(table[["held_out_load", "file_id", "true_label", "method", "predicted_label", *[f"prob_{x}" for x in LABELS], "confidence", "margin", "entropy"]].to_dict("records"))
            y, pred = table.true_label.to_numpy(), table.predicted_label.to_numpy()
            fold_rows.append({"held_out_load": int(load), "method": method,
                              "macro_f1": f1_score(y, pred, labels=LABELS, average="macro", zero_division=0),
                              "balanced_accuracy": balanced_accuracy_score(y, pred),
                              **{f"recall_{x}": v for x, v in zip(LABELS, recall_score(y, pred, labels=LABELS, average=None, zero_division=0))},
                              "collapse_warning": bool(len(set(pred)) == 1)})
    predictions, folds = pd.DataFrame(rows), pd.DataFrame(fold_rows)
    summary = []
    for method, part in folds.groupby("method"):
        oof = predictions[predictions.method == method]
        y, pred = oof.true_label.to_numpy(), oof.predicted_label.to_numpy()
        summary.append({"method": method, "macro_f1_mean": float(part.macro_f1.mean()), "macro_f1_std": float(part.macro_f1.std(ddof=0)),
                        "balanced_accuracy_mean": float(part.balanced_accuracy.mean()), "balanced_accuracy_std": float(part.balanced_accuracy.std(ddof=0)),
                        **{f"recall_{x}": v for x, v in zip(LABELS, recall_score(y, pred, labels=LABELS, average=None, zero_division=0))},
                        "worst_load_f1": float(part.macro_f1.min()), "collapse_warning": bool(part.collapse_warning.any()),
                        "evaluation_unit": "held-out source raw MAT file after unlabelled-target adaptation"})
    audit = {"protocol_fair": True, "held_out_labels_hidden_during_adaptation": True, "folds": protocol,
             "rule": "Labels are copied for scoring only after each method has completed prediction."}
    return predictions, pd.DataFrame(summary), audit


def select_final_method(summary: pd.DataFrame, coral_seed_stability: float, ot_seed_stability: float) -> tuple[str, dict]:
    """Predeclared selection: benchmark first, then stability/simplicity; never target labels."""
    metric = summary.set_index("method")
    source, coral, ot = (metric.loc[name] for name in ("source_only", "coral", "class_regularized_ot"))
    # A source-only win by more than 0.01 is meaningful enough to avoid an unnecessary adaptation step.
    if source.macro_f1_mean >= max(coral.macro_f1_mean, ot.macro_f1_mean) + .01 and not source.collapse_warning:
        chosen, reason = "source_only", "surrogate UDA F1 exceeds both adaptation methods by predeclared 0.01"
    elif (ot.macro_f1_mean >= coral.macro_f1_mean - .005 and ot.balanced_accuracy_mean >= coral.balanced_accuracy_mean - .005 and
          ot.worst_load_f1 >= coral.worst_load_f1 - .02 and not ot.collapse_warning and ot_seed_stability >= coral_seed_stability - .03):
        chosen, reason = "class_regularized_ot", "OT is not materially worse than CORAL on the fair surrogate benchmark and its seed stability is not worse"
    else:
        chosen, reason = "coral", "CORAL is materially better or more stable under the predeclared fair-benchmark rule; choose the simpler alignment"
    return chosen, {"selected_final_method": chosen, "selection_basis": reason, "selection_thresholds": {"f1_tie": .005, "ba_tie": .005, "worst_load_f1_tie": .02, "seed_stability_tie": .03},
                    "target_labels_used": False, "pdf_or_reference_labels_used": False}


def geometry_for_method(source: pd.DataFrame, target: pd.DataFrame, zs: np.ndarray, zt: np.ndarray, method: str,
                        transform: dict | None, final: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Final explanation in the exact representation used to form final predictions."""
    aligned_source = zs if method == "source_only" else (apply_coral(zs, transform) if method == "coral" else apply_ot(zs, transform))
    scaler = StandardScaler().fit(aligned_source, sample_weight=class_file_window_weights(source))
    source_space, target_space = scaler.transform(aligned_source), scaler.transform(zt)
    centers = {label: source_space[source.label.to_numpy() == label].mean(axis=0) for label in LABELS}
    file_centers = {str(fid): source_space[np.asarray(idx)].mean(axis=0) for fid, idx in source.groupby("file_id").indices.items()}
    final_by_file = final.set_index("file_id")
    rows = []
    for fid, indices in target.groupby("file_id", sort=True).indices.items():
        point = target_space[np.asarray(indices)].mean(axis=0)
        row = {"file_id": str(fid), "final_method": final_by_file.loc[str(fid), "final_method"],
               "final_candidate_label": final_by_file.loc[str(fid), "candidate_label"],
               **{f"distance_to_{label}": float(np.linalg.norm(point - centers[label])) for label in LABELS},
               "nearest_source_class": min(centers, key=lambda label: np.linalg.norm(point - centers[label])),
               "nearest_source_file": min(file_centers, key=lambda item: np.linalg.norm(point - file_centers[item]))}
        row.update(final_by_file.loc[str(fid), ["probability_margin", "normalized_entropy", "encoder_agreement", "rpm_agreement", "subsample_agreement", "loto_agreement", "review_required"]].to_dict())
        rows.append(row)
    movement = pd.DataFrame([{"label": label, "aligned_center_norm": float(np.linalg.norm(centers[label])), "method": method} for label in LABELS])
    return pd.DataFrame(rows), movement


def loto_for_method(method: str, source: pd.DataFrame, target: pd.DataFrame, zs: np.ndarray, zt: np.ndarray,
                    source_p: np.ndarray, seed: int) -> pd.DataFrame:
    rows = []
    for file_id, indices in target.groupby("file_id", sort=True).indices.items():
        part = target.iloc[np.asarray(indices)]
        if method == "source_only":
            p = source_p[np.asarray(indices)]
        else:
            keep = target.file_id.to_numpy() != file_id
            if method == "coral":
                transform = fit_coral(zs, source, zt[keep], target.loc[keep], FORMAL_VARIANT); head = fit_head(apply_coral(zs, transform), source, seed)
            else:
                transform = fit_class_regularized_ot(zs, source, zt[keep], target.loc[keep]); head = fit_head(apply_ot(zs, transform), source, seed)
            p = probabilities(head, zt[np.asarray(indices)])
        rows.append(aggregate(part, p, f"{method}_loto"))
    return pd.concat(rows, ignore_index=True)


def target_stability_for_method(method: str, source: pd.DataFrame, target: pd.DataFrame, names: list[str], model: q2.SourceMLP,
                                scaler: StandardScaler, zs: np.ndarray, zt: np.ndarray, source_p: np.ndarray) -> dict:
    """Same 570/600/630, LOTO and subsample rules for every candidate method."""
    p600, transform600 = _method_probabilities(method, source, target, zs, zt, source_p, 2025)
    base = aggregate(target, p600, method).set_index("file_id")
    loto_table = loto_for_method(method, source, target, zs, zt, source_p, 2025)
    loto_map = loto_table.set_index("file_id").predicted_label.to_dict()
    tables = {600: base}
    for rpm in (570, 630):
        _, rpm_target, _ = load_inputs(Path("outputs/q3_refined"), rpm)
        zrpm, rpm_source_p = infer(model, scaler.transform(rpm_target[names].to_numpy(float)))
        p, _ = _method_probabilities(method, source, rpm_target, zs, zrpm, rpm_source_p, 2025)
        tables[rpm] = aggregate(rpm_target, p, method).set_index("file_id")
    rpm_rows = []
    for fid in list("ABCDEFGHIJKLMNOP"):
        labels = [tables[rpm].loc[fid, "predicted_label"] for rpm in (570, 600, 630)]
        conf = [tables[rpm].loc[fid, "confidence"] for rpm in (570, 600, 630)]
        margin = [tables[rpm].loc[fid, "probability_margin"] for rpm in (570, 600, 630)]
        rpm_rows.append({"file_id": fid, "method": method, "prediction_570": labels[0], "prediction_600": labels[1], "prediction_630": labels[2],
                         "agreement_ratio": float(np.mean(np.asarray(labels) == labels[1])), "confidence_variation": float(np.ptp(conf)), "margin_variation": float(np.ptp(margin))})
    rpm_table = pd.DataFrame(rpm_rows)
    return {"probabilities": p600, "transform": transform600, "base": base.reset_index(), "loto": loto_table, "rpm": rpm_table,
            "subsample": stability(target, p600), "rpm_mean": float(rpm_table.agreement_ratio.mean()),
            "loto_mean": float(np.mean([base.loc[fid, "predicted_label"] == loto_map[fid] for fid in base.index]))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/q3_refined"))
    parser.add_argument("--data-root", type=Path, default=Path("数据集") / "数据集")
    parser.add_argument("--epochs", type=int, default=45); parser.add_argument("--surrogate-epochs", type=int, default=30)
    args = parser.parse_args()
    if args.epochs < 1 or args.surrogate_epochs < 1: raise ValueError("epochs must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True); (args.output_dir / "models").mkdir(parents=True, exist_ok=True)
    previous = pre_fair_prediction_baseline(args.output_dir)
    # Rebuild every v2 input after the schema correction; no old feature cache is reused.
    v2.build(args.data_root, args.output_dir, 600, write_source=True)
    for rpm in (570, 630): v2.build(args.data_root, args.output_dir, rpm, write_source=False)
    source, target, names = load_inputs(args.output_dir, 600)
    schema_audit = json.loads((args.output_dir / "transfer_v2_schema_audit.json").read_text(encoding="utf-8"))
    ablation, ablation_schema = feature_ablation(source, names)
    write_csv(args.output_dir / "transfer_v2_ablation.csv", ablation); write_json(args.output_dir / "transfer_v2_ablation_schema.json", ablation_schema)
    audit_lines = ["# 固定 Hz 包络依赖审计", "", "消融只在源域文件级 LOLO Logistic 探针上完成，未读取 A–P 标签或候选结果。", "", "```csv", ablation.to_csv(index=False).strip(), "```", "", "`no_absolute_hz` 删除三段固定 Hz 包络特征；`no_envelope` 还删除不依赖固定 Hz 的全频包络描述；`no_order` 删除所有阶次描述。四组 schema 不同。", "",
                   "固定 Hz 包络可能编码 CWRU 传感器/结构共振，因此其源域增益不能被解释为跨设备正确性。正式 Transfer-v2 暂保留 full schema，以预先固定的源域可辨识性作为工程基线；最终目标结论保持候选级，且 no_absolute_hz 是必须报告的设备依赖敏感性，而非按 A–P 结果选择。"]
    (args.output_dir / "fixed_hz_dependency_audit.md").write_text("\n".join(audit_lines) + "\n", encoding="utf-8")

    # Five source encoders: fixed seeds, no best-seed selection.
    seed_records = {"source_only": [], "coral": [], "class_regularized_ot": []}; reference = None
    for seed in SEEDS:
        model, scaler = train_encoder(source, names, seed, args.epochs)
        zs, _ = infer(model, scaler.transform(source[names].to_numpy(float)))
        zt, source_p = infer(model, scaler.transform(target[names].to_numpy(float)))
        for method in seed_records:
            p, transform = _method_probabilities(method, source, target, zs, zt, source_p, seed)
            table = aggregate(target, p, method)
            for _, row in table.iterrows(): seed_records[method].append({"file_id": row.file_id, "seed": seed, "predicted_label": row.predicted_label, "confidence": row.confidence, "margin": row.probability_margin, "entropy": row.normalized_entropy})
            if seed == 2025 and method == "coral": reference = (model, scaler, zs, zt, source_p, transform)
    seed_consensus = {}
    for method, records in seed_records.items():
        table = pd.DataFrame(records); write_csv(args.output_dir / ("encoder_seed_predictions.csv" if method == "coral" else f"{method}_seed_predictions.csv"), table)
        rows = []
        for fid, part in table.groupby("file_id", sort=True):
            vote = part.predicted_label.mode().iloc[0]
            rows.append({"file_id": fid, "majority_label": vote, "agreement_ratio": float((part.predicted_label == vote).mean()), "mean_confidence": float(part.confidence.mean()), "mean_margin": float(part.margin.mean())})
        seed_consensus[method] = pd.DataFrame(rows)
        write_csv(args.output_dir / ("encoder_seed_consensus.csv" if method == "coral" else f"{method}_seed_consensus.csv"), seed_consensus[method])
    model, scaler, zs, zt, source_p, coral_transform = reference
    torch.save(model.encoder.state_dict(), args.output_dir / "models" / "transfer_v2_seed2025_encoder.pth"); torch.save(model.classifier.state_dict(), args.output_dir / "models" / "transfer_v2_seed2025_classifier.pth")

    surrogate_predictions, surrogate_summary, surrogate_audit = surrogate_uda_benchmark(source, names, args.surrogate_epochs)
    write_csv(args.output_dir / "surrogate_uda_predictions.csv", surrogate_predictions); write_csv(args.output_dir / "surrogate_uda_summary.csv", surrogate_summary)
    candidate_stability = {method: target_stability_for_method(method, source, target, names, model, scaler, zs, zt, source_p)
                           for method in ("source_only", "coral", "class_regularized_ot")}
    stability_rows = [{"method": method, "seed_agreement_mean": float(seed_consensus[method].agreement_ratio.mean()),
                       "rpm_agreement_mean": values["rpm_mean"], "loto_agreement_mean": values["loto_mean"],
                       "subsample_agreement_mean": float(np.mean(list(values["subsample"].values())))}
                      for method, values in candidate_stability.items()]
    write_csv(args.output_dir / "method_target_stability.csv", pd.DataFrame(stability_rows))
    selected, selection = select_final_method(surrogate_summary, float(seed_consensus["coral"].agreement_ratio.mean()), float(seed_consensus["class_regularized_ot"].agreement_ratio.mean()))
    chosen_stability = candidate_stability[selected]
    final_p, final_transform = chosen_stability["probabilities"], chosen_stability["transform"]
    final_table = chosen_stability["base"].rename(columns={"predicted_label": "candidate_label"})
    final_method_name = {"source_only": "Source-only Transfer-v2 S56", "coral": "CORAL Transfer-v2 S56 class/file-balanced", "class_regularized_ot": "Class-Regularized OT Transfer-v2 S56"}[selected]
    final_table.insert(1, "final_method", final_method_name)
    chosen_consensus = seed_consensus[selected].set_index("file_id")
    final_table["encoder_agreement"] = final_table.file_id.map(chosen_consensus.agreement_ratio)

    loto_table = chosen_stability["loto"]; write_csv(args.output_dir / "final_method_leave_one_target_out.csv", loto_table)
    loto_map = loto_table.set_index("file_id").predicted_label.to_dict()
    rpm_sensitivity = chosen_stability["rpm"].drop(columns="method"); write_csv(args.output_dir / "rpm_sensitivity.csv", rpm_sensitivity)
    final_table["rpm_agreement"] = final_table.file_id.map(rpm_sensitivity.set_index("file_id").agreement_ratio)
    final_table["subsample_agreement"] = final_table.file_id.map(chosen_stability["subsample"])
    final_table["loto_agreement"] = final_table.apply(lambda row: row.candidate_label == loto_map[row.file_id], axis=1)
    final_table["review_required"] = ((final_table.encoder_agreement < .8) | (final_table.rpm_agreement < 1.0) | (final_table.subsample_agreement < .75) | (~final_table.loto_agreement) | (final_table.probability_margin < .05) | (final_table.normalized_entropy > .8))
    final_table["review_reason"] = final_table.apply(lambda row: ";".join(name for name, bad in (("encoder", row.encoder_agreement < .8), ("rpm", row.rpm_agreement < 1), ("subsample", row.subsample_agreement < .75), ("loto", not row.loto_agreement), ("low_margin", row.probability_margin < .05), ("high_entropy", row.normalized_entropy > .8)) if bad) or "stable", axis=1)
    expected = np.asarray(LABELS)[final_table[[f"prob_{x}" for x in LABELS]].to_numpy().argmax(1)]
    if not np.array_equal(expected, final_table.candidate_label.to_numpy()) or len(final_table) != 16: raise AssertionError("final output must be 16 argmax-consistent target candidates")
    write_csv(args.output_dir / "target_predictions_final.csv", final_table)

    geometry_final, geometry_movement = geometry_for_method(source, target, zs, zt, selected, final_transform, final_table)
    write_csv(args.output_dir / "target_class_geometry_final.csv", geometry_final); write_csv(args.output_dir / "target_class_geometry.csv", geometry_final); write_csv(args.output_dir / "source_class_center_movement.csv", geometry_movement)
    for method in ("coral", "class_regularized_ot"):
        p, transform = _method_probabilities(method, source, target, zs, zt, source_p, 2025)
        local = aggregate(target, p, method).rename(columns={"predicted_label": "candidate_label"}); local.insert(1, "final_method", method)
        for column in ("encoder_agreement", "rpm_agreement", "subsample_agreement", "loto_agreement", "review_required"):
            local[column] = final_table.set_index("file_id")[column].reindex(local.file_id).to_numpy()
        local["probability_margin"] = local.probability_margin; local["normalized_entropy"] = local.normalized_entropy
        geo, _ = geometry_for_method(source, target, zs, zt, method, transform, local)
        write_csv(args.output_dir / f"target_class_geometry_{method}.csv", geo)

    change_count = 0
    if previous is not None:
        changes = previous.merge(final_table[["file_id", "candidate_label", "encoder_agreement", "rpm_agreement", "review_required"]], on="file_id")
        changes["changed"] = changes.before_fair_benchmark != changes.candidate_label
        write_csv(args.output_dir / "target_label_changes_fair_benchmark.csv", changes)
        change_count = int(changes.changed.sum())
    metrics_rows = []
    for method in ("source_only", "coral", "class_regularized_ot"):
        p, transform = _method_probabilities(method, source, target, zs, zt, source_p, 2025)
        aligned = zs if method == "source_only" else (apply_coral(zs, transform) if method == "coral" else apply_ot(zs, transform))
        metrics_rows.append({"method": method, **old.domain_metrics(source, target, aligned, zt), **old.collapse(aggregate(target, p, method))})
    write_csv(args.output_dir / "domain_metrics.csv", pd.DataFrame(metrics_rows))
    # Retained only as a semantic-preservation diagnostic; not used for method selection.
    retention = surrogate_summary.rename(columns={"macro_f1_mean": "macro_f1", "balanced_accuracy_mean": "balanced_accuracy"}).copy()
    retention["interpretation"] = "Surrogate unlabelled-target OOF diagnostic; not the primary method-selection claim outside its fair benchmark."
    write_csv(args.output_dir / "source_retention.csv", retention)
    write_csv(args.output_dir / "source_retention_predictions.csv", surrogate_predictions.assign(interpretation="Fair surrogate UDA OOF prediction; labels were joined only after adaptation."))
    verification = {"target_labels_used": False, "pdf_or_reference_labels_used": False, "duplicate_transfer_feature_removed": True,
                    "ablation_sets_are_distinct": True, "surrogate_uda_protocol_fair": True, "surrogate_target_labels_hidden_during_adaptation": True,
                    "final_geometry_matches_final_method": True, "final_argmax_consistent": True, "target_file_count": 16,
                    **selection, "surrogate_uda_metrics": surrogate_summary.to_dict("records"), "method_target_stability": stability_rows, "schema_audit": schema_audit}
    write_json(args.output_dir / "verification.json", verification)
    old_labels = previous if previous is not None else pd.DataFrame(columns=["file_id", "before_fair_benchmark"])
    comparison = surrogate_summary.set_index("method")
    domain = pd.DataFrame(metrics_rows).set_index("method")
    stability_summary = pd.DataFrame(stability_rows).set_index("method")
    lines = ["# 第三问封版前公平验证与解释链修正", "", f"最终唯一主方法：**{final_method_name}**。{selection['selection_basis']}。该结论只使用统一 surrogate UDA、无标签稳定性和类别塌缩规则；未读取 A–P 真值或 PDF/参考答案。", "",
             "| Method | Surrogate UDA F1 | BA | Worst-load F1 | MMD | Seed stability | RPM stability | Collapse |", "|---|---:|---:|---:|---:|---:|---:|---|"]
    for method in ("source_only", "coral", "class_regularized_ot"):
        seed_value = stability_summary.loc[method, "seed_agreement_mean"]; rpm_value = stability_summary.loc[method, "rpm_agreement_mean"]
        lines.append(f"| {method} | {comparison.loc[method,'macro_f1_mean']:.4f}±{comparison.loc[method,'macro_f1_std']:.4f} | {comparison.loc[method,'balanced_accuracy_mean']:.4f} | {comparison.loc[method,'worst_load_f1']:.4f} | {domain.loc[method,'mmd_file_level']:.4f} | {seed_value:.3f} | {rpm_value:.3f} | {comparison.loc[method,'collapse_warning']} |")
    lines += ["", "## 解释边界", "", "`target_class_geometry_final.csv` 在最终方法空间计算：CORAL 使用 CORAL 变换后的 source，OT 使用 OT affine 变换后的 source，source-only 不作对齐；target embedding 从不以标签参与变换。距离与最近类仅是模型几何证据，不是目标真值。", "", "## 固定 Hz 风险", "", "见 `fixed_hz_dependency_audit.md`。full Transfer-v2 用作预先固定的工程基线；固定 Hz 包络的设备依赖风险不能由本数据的无标签目标验证排除，因而 A–P 仍只能称候选诊断标签。", "", f"相对提交版本的 pre-fair 候选，有 {change_count}/16 个文件发生变化；逐文件表见 `target_label_changes_fair_benchmark.csv`，没有按外部答案回改。", f"A–P 中需复核：{int(final_table.review_required.sum())}/16。即使公平 benchmark 完成，也只有在稳定性复核解释充分后才能宣称第三问封版；当前不报告任何目标 accuracy/F1/recall。"]
    (args.output_dir / "q3_refined_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(args.output_dir / "q3_refined_config.json", {"seeds": SEEDS, "epochs": args.epochs, "surrogate_epochs": args.surrogate_epochs, "features": names, "selected_final_method": selected, "selection_basis": selection["selection_basis"], "target_labels_used": False, "statistical_unit": "raw MAT file"})
    print(f"Fair refined Question 3 outputs written to {args.output_dir}")


if __name__ == "__main__":
    main()
