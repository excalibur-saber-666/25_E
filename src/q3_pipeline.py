"""Question 3: leakage-aware unsupervised transfer on the frozen Transfer20 schema.

The A--P files are never assigned target labels.  Metrics involving them are
domain-discrepancy and stability diagnostics, not target-domain accuracy.

Run from the repository root:
    python src/q3_pipeline.py
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

import q2_pipeline as q2
from q2_transfer_pretrain import load_interface

LABELS = q2.LABELS
SEEDS = (2025, 2026, 2027, 2028, 2029)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")


def _json_default(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def write_csv(path: Path, table: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False, encoding="utf-8-sig")


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def load_data(q1_dir: Path, model_dir: Path):
    """Read exactly the frozen Transfer20 inputs and reject a target label."""
    source = pd.read_csv(q1_dir / "features_source_transfer.csv")
    target = pd.read_csv(q1_dir / "features_target_transfer.csv")
    names = json.loads((q1_dir / "feature_names_transfer.json").read_text(encoding="utf-8"))["features"]
    saved = json.loads((model_dir / "q2_transfer20_feature_names.json").read_text(encoding="utf-8"))["features"]
    if "label" in target.columns:
        raise ValueError("Target labels are forbidden: refusing to run Question 3 with a target label column")
    if len(names) != 20 or names != saved:
        raise ValueError("Frozen Transfer20 schema does not match the Question 2 interface")
    for frame, domain in ((source, "source"), (target, "target")):
        missing = {"file_id", "window_id", *names} - set(frame.columns)
        if missing:
            raise ValueError(f"{domain} input missing columns: {sorted(missing)}")
        values = frame[names].to_numpy(float)
        if not np.isfinite(values).all():
            raise ValueError(f"{domain} Transfer20 contains NaN or Inf")
        if frame.file_id.isna().any() or frame.window_id.isna().any():
            raise ValueError(f"{domain} has missing file/window identifiers")
    if set(source.label.unique()) != set(LABELS):
        raise ValueError("Source labels must be exactly N/B/IR/OR")
    if source.file_id.nunique() != 56 or target.file_id.nunique() != 16:
        raise ValueError("Expected 56 source and 16 target raw files")
    target_ids = sorted(target.file_id.astype(str).unique())
    if target_ids != list("ABCDEFGHIJKLMNOP"):
        raise ValueError("Target files must be A--P exactly once")
    metadata = pd.read_csv(q1_dir / "source_metadata.csv")
    source = source.merge(metadata[["file_id", "load"]], on="file_id", how="left", validate="many_to_one")
    if source.load.isna().any():
        raise ValueError("Source load metadata cannot be joined")
    audit = {
        "input": "frozen Transfer20 only", "ordered_features": names,
        "source_windows": int(len(source)), "target_windows": int(len(target)),
        "source_files": int(source.file_id.nunique()), "target_files": int(target.file_id.nunique()),
        "source_windows_per_file": source.groupby("file_id").size().to_dict(),
        "target_windows_per_file": target.groupby("file_id").size().to_dict(),
        "target_label_column_present": False,
        "target_labels_used_for_fit_loss_selection_or_metric": False,
    }
    return source, target, names, audit


def infer(encoder: nn.Module, classifier: nn.Module, values: np.ndarray, batch_size: int = 256):
    encoder.eval(); classifier.eval()
    embeddings, probabilities = [], []
    with torch.no_grad():
        for start in range(0, len(values), batch_size):
            x = torch.tensor(values[start:start + batch_size], dtype=torch.float32)
            z = encoder(x)
            embeddings.append(z.cpu().numpy())
            probabilities.append(torch.softmax(classifier(z), dim=1).cpu().numpy())
    return np.vstack(embeddings), np.vstack(probabilities)


def group_embeddings(frame: pd.DataFrame, embeddings: np.ndarray) -> tuple[list[str], np.ndarray]:
    ids, rows = [], []
    for file_id, indices in frame.groupby("file_id", sort=True).indices.items():
        z = embeddings[np.asarray(indices)]
        ids.append(str(file_id)); rows.append(np.r_[z.mean(axis=0), z.std(axis=0, ddof=0)])
    return ids, np.asarray(rows)


def rbf_mmd(x: np.ndarray, y: np.ndarray) -> float:
    """Biased multi-kernel RBF MMD on file-level observations."""
    both = np.vstack([x, y])
    sq = np.sum((both[:, None, :] - both[None, :, :]) ** 2, axis=2)
    positive = sq[np.triu_indices_from(sq, k=1)]
    median = float(np.median(positive[positive > 0])) if np.any(positive > 0) else 1.0
    values = []
    for factor in (0.5, 1.0, 2.0):
        gamma = 1.0 / max(factor * median, 1e-12)
        kxx = np.exp(-gamma * np.sum((x[:, None] - x[None, :]) ** 2, axis=2)).mean()
        kyy = np.exp(-gamma * np.sum((y[:, None] - y[None, :]) ** 2, axis=2)).mean()
        kxy = np.exp(-gamma * np.sum((x[:, None] - y[None, :]) ** 2, axis=2)).mean()
        values.append(kxx + kyy - 2 * kxy)
    return float(np.mean(values))


def proxy_a_distance(source_file_z: np.ndarray, target_file_z: np.ndarray, seed: int = 2025) -> dict:
    x = np.vstack([source_file_z, target_file_z])
    y = np.r_[np.ones(len(source_file_z), dtype=int), np.zeros(len(target_file_z), dtype=int)]
    cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=seed)
    pred = np.zeros(len(y), dtype=int)
    for train, test in cv.split(x, y):
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed))
        clf.fit(x[train], y[train]); pred[test] = clf.predict(x[test])
    ba = float(balanced_accuracy_score(y, pred))
    return {"domain_balanced_accuracy": ba, "proxy_a_distance": float(np.clip(4 * ba - 2, 0, 2)),
            "unit": "file embedding mean+std", "n_source_files": len(source_file_z), "n_target_files": len(target_file_z)}


def domain_metrics(source_frame: pd.DataFrame, target_frame: pd.DataFrame, zs: np.ndarray, zt: np.ndarray) -> dict:
    _, fs = group_embeddings(source_frame, zs); _, ft = group_embeddings(target_frame, zt)
    out = proxy_a_distance(fs, ft)
    out["mmd_file_level"] = rbf_mmd(fs, ft)
    return out


def aggregate(frame: pd.DataFrame, probabilities: np.ndarray, method: str) -> pd.DataFrame:
    rows = []
    for file_id, indices in frame.groupby("file_id", sort=True).indices.items():
        p = probabilities[np.asarray(indices)]
        mean = p.mean(axis=0); label_index = int(np.argmax(mean)); window_labels = p.argmax(axis=1)
        entropy = -float(np.sum(mean * np.log(np.clip(mean, 1e-12, 1))) / math.log(len(LABELS)))
        sorted_p = np.sort(mean)
        row = {"file_id": str(file_id), "method": method, "predicted_label": LABELS[label_index],
               "confidence": float(mean[label_index]), "window_vote_ratio": float(np.mean(window_labels == label_index)),
               "normalized_entropy": entropy, "probability_margin": float(sorted_p[-1] - sorted_p[-2]),
               "window_probability_std": float(p[:, label_index].std(ddof=0))}
        row.update({f"prob_{name}": float(mean[i]) for i, name in enumerate(LABELS)})
        rows.append(row)
    return pd.DataFrame(rows)


def file_balanced_weights(file_ids: np.ndarray | pd.Series) -> np.ndarray:
    """Weights summing to one, with every raw MAT file contributing equally."""
    file_ids = np.asarray(file_ids, dtype=str)
    if file_ids.ndim != 1 or len(file_ids) == 0:
        raise ValueError("file identifiers must be a non-empty vector")
    _, inverse, counts = np.unique(file_ids, return_inverse=True, return_counts=True)
    return 1.0 / (len(counts) * counts[inverse])


def weighted_mean_covariance(values: np.ndarray, file_ids: np.ndarray | pd.Series, epsilon: float = 0.0):
    """Window-level covariance with equal aggregate contribution per MAT file."""
    values = np.asarray(values, dtype=float)
    weights = file_balanced_weights(file_ids)
    if values.ndim != 2 or len(values) != len(weights) or not np.isfinite(values).all():
        raise ValueError("weighted covariance needs a finite matrix aligned to file identifiers")
    mean = np.sum(values * weights[:, None], axis=0)
    centered = values - mean
    covariance = centered.T @ (centered * weights[:, None]) + epsilon * np.eye(values.shape[1])
    if not np.allclose(covariance, covariance.T) or not np.isfinite(covariance).all():
        raise ValueError("weighted covariance is not finite and symmetric")
    return mean, covariance, weights


def _matrix_power(matrix: np.ndarray, exponent: float, epsilon: float) -> np.ndarray:
    values, vectors = np.linalg.eigh(matrix)
    return (vectors * np.maximum(values, epsilon) ** exponent) @ vectors.T


def fit_coral_transform(source_z: np.ndarray, source_file_ids, target_z: np.ndarray, target_file_ids, epsilon: float = 1e-4) -> dict:
    """Fit file-balanced CORAL; target data are used only without labels."""
    if source_z.ndim != 2 or source_z.shape[1] != target_z.shape[1] or epsilon <= 0:
        raise ValueError("CORAL needs equal-dimensional inputs and positive epsilon")
    sm, cs, source_weights = weighted_mean_covariance(source_z, source_file_ids, epsilon)
    tm, ct, target_weights = weighted_mean_covariance(target_z, target_file_ids, epsilon)
    return {"source_mean": sm, "target_mean": tm,
            "map": _matrix_power(cs, -0.5, epsilon) @ _matrix_power(ct, 0.5, epsilon),
            "source_covariance": cs, "target_covariance": ct,
            "source_weights": source_weights, "target_weights": target_weights, "epsilon": epsilon}


def apply_coral(values: np.ndarray, transform: dict) -> np.ndarray:
    aligned = (np.asarray(values, dtype=float) - transform["source_mean"]) @ transform["map"] + transform["target_mean"]
    if not np.isfinite(aligned).all():
        raise FloatingPointError("CORAL produced non-finite embedding")
    return aligned


def coral_align(source_z: np.ndarray, target_z: np.ndarray, source_file_ids=None, target_file_ids=None, epsilon: float = 1e-4) -> np.ndarray:
    """Compatibility wrapper for file-balanced CORAL source alignment."""
    source_file_ids = np.arange(len(source_z)) if source_file_ids is None else source_file_ids
    target_file_ids = np.arange(len(target_z)) if target_file_ids is None else target_file_ids
    return apply_coral(source_z, fit_coral_transform(source_z, source_file_ids, target_z, target_file_ids, epsilon))


def window_weights(frame: pd.DataFrame, labels: np.ndarray | None = None) -> np.ndarray:
    counts = frame.groupby("file_id").size().to_dict()
    if labels is None:
        return np.asarray([1.0 / counts[row.file_id] for _, row in frame.iterrows()], dtype=float)
    file_label = frame.groupby("file_id").label.first().to_dict()
    class_files = pd.Series(file_label).value_counts().to_dict()
    return np.asarray([1.0 / (len(LABELS) * class_files[file_label[row.file_id]] * counts[row.file_id])
                       for _, row in frame.iterrows()], dtype=float)


def fit_source_linear(embeddings: np.ndarray, source: pd.DataFrame, seed: int = 2025) -> LogisticRegression:
    """One and only one class/file balancing mechanism: sample_weight."""
    head = LogisticRegression(max_iter=3000, class_weight=None, random_state=seed)
    head.fit(embeddings, source.label.to_numpy(), sample_weight=window_weights(source, source.label.to_numpy()))
    return head


def linear_probabilities(head: LogisticRegression, embeddings: np.ndarray) -> np.ndarray:
    probabilities = np.zeros((len(embeddings), len(LABELS)))
    probabilities[:, [LABELS.index(str(label)) for label in head.classes_]] = head.predict_proba(embeddings)
    if not np.isfinite(probabilities).all() or not np.allclose(probabilities.sum(axis=1), 1.0):
        raise FloatingPointError("linear classifier probabilities are invalid")
    return probabilities


class GradientReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values, coefficient):
        ctx.coefficient = coefficient
        return values.view_as(values)
    @staticmethod
    def backward(ctx, grad):
        return grad.neg().mul(ctx.coefficient), None


class DomainDiscriminator(nn.Module):
    def __init__(self, dropout: float = 0.10):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(32, 32), nn.GELU(), nn.Dropout(dropout), nn.Linear(32, 16), nn.GELU(), nn.Linear(16, 1))
    def forward(self, z, coefficient):
        return self.layers(GradientReverse.apply(z, coefficient))


class DANN(nn.Module):
    def __init__(self, dropout: float = 0.10):
        super().__init__()
        self.encoder = q2.FeatureEncoder(20, dropout)
        self.classifier = q2.SourceClassifier()
        self.domain = DomainDiscriminator(dropout)


def train_dann(source: pd.DataFrame, target: pd.DataFrame, names: list[str], seed: int, epochs: int,
               domain_weight: float, initial: tuple[nn.Module, nn.Module, object] | None = None):
    """Train with target inputs only; no target labels or selection metric exist here."""
    set_seed(seed)
    if initial is None:
        scaler = q2.fit_weighted_scaler(source, names)
        dropout = 0.10
    else:
        initial_encoder, initial_head, scaler = initial
        dropout = next((m.p for m in initial_encoder.modules() if isinstance(m, nn.Dropout)), 0.10)
    sx = scaler.transform(source[names].to_numpy(float)).astype(np.float32)
    tx = scaler.transform(target[names].to_numpy(float)).astype(np.float32)
    sy = np.asarray([LABELS.index(v) for v in source.label], dtype=np.int64)
    model = DANN(float(dropout))
    if initial is not None:
        model.encoder.load_state_dict(copy.deepcopy(initial_encoder.state_dict()))
        model.classifier.load_state_dict(copy.deepcopy(initial_head.state_dict()))
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4)
    src_loader = DataLoader(TensorDataset(torch.tensor(sx), torch.tensor(sy)), batch_size=64,
                            sampler=WeightedRandomSampler(torch.tensor(window_weights(source, sy), dtype=torch.double),
                                                          num_samples=max(len(source), len(target)), replacement=True,
                                                          generator=torch.Generator().manual_seed(seed)))
    tgt_loader = DataLoader(TensorDataset(torch.tensor(tx)), batch_size=64,
                            sampler=WeightedRandomSampler(torch.tensor(window_weights(target), dtype=torch.double),
                                                          num_samples=max(len(source), len(target)), replacement=True,
                                                          generator=torch.Generator().manual_seed(seed + 41)))
    total = max(1, epochs * len(src_loader)); step = 0; history = []
    for epoch in range(1, epochs + 1):
        model.train(); sums = {"source_classification_loss": 0.0, "domain_loss": 0.0, "domain_accuracy": 0.0}
        for (xb, yb), (tb,) in zip(src_loader, tgt_loader):
            progress = step / max(total - 1, 1)
            coefficient = float(2 / (1 + math.exp(-10 * progress)) - 1)
            zs, zt = model.encoder(xb), model.encoder(tb)
            cls = F.cross_entropy(model.classifier(zs), yb)
            ds, dt = model.domain(zs, coefficient), model.domain(zt, coefficient)
            dloss = .5 * (F.binary_cross_entropy_with_logits(ds, torch.ones_like(ds)) + F.binary_cross_entropy_with_logits(dt, torch.zeros_like(dt)))
            loss = cls + domain_weight * dloss
            if not torch.isfinite(loss):
                raise FloatingPointError("DANN loss is non-finite")
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
            with torch.no_grad():
                pred = torch.cat((ds, dt)).sigmoid() >= .5
                truth = torch.cat((torch.ones_like(ds), torch.zeros_like(dt))) >= .5
                sums["source_classification_loss"] += float(cls); sums["domain_loss"] += float(dloss)
                sums["domain_accuracy"] += float((pred == truth).float().mean())
            step += 1
        history.append({"seed": seed, "epoch": epoch, **{k: v / len(src_loader) for k, v in sums.items()}})
    # A constant discriminator is an implementation failure, not evidence of adaptation.
    with torch.no_grad():
        test_logits = model.domain(model.encoder(torch.tensor(sx[:min(128, len(sx))])), 1.0)
    if float(test_logits.std()) < 1e-7:
        raise RuntimeError("Domain discriminator has degenerated to a constant")
    return model.eval(), scaler, history


def train_source_fold(source: pd.DataFrame, names: list[str], seed: int, epochs: int):
    set_seed(seed); scaler = q2.fit_weighted_scaler(source, names)
    x = scaler.transform(source[names].to_numpy(float)).astype(np.float32)
    y = np.asarray([LABELS.index(v) for v in source.label], dtype=np.int64)
    model = q2.SourceMLP(20, dropout=.10); optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4)
    loader = DataLoader(TensorDataset(torch.tensor(x), torch.tensor(y)), batch_size=64,
                        sampler=WeightedRandomSampler(torch.tensor(window_weights(source, y), dtype=torch.double), len(source), replacement=True,
                                                      generator=torch.Generator().manual_seed(seed)))
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            loss = F.cross_entropy(model(xb), yb); optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
    return model.eval(), scaler


def coral_source_retention(source: pd.DataFrame, target: pd.DataFrame, names: list[str], epochs: int):
    """LOLO for Source-MLP, Source-linear and CORAL without held-out source leakage."""
    rows = []
    for load in sorted(source.load.unique()):
        tr, te = source[source.load != load].copy(), source[source.load == load].copy()
        source_model, scaler = train_source_fold(tr, names, 5000 + int(load), epochs)
        ztr, _ = infer(source_model.encoder, source_model.classifier, scaler.transform(tr[names].to_numpy(float)))
        zte, mlp_p = infer(source_model.encoder, source_model.classifier, scaler.transform(te[names].to_numpy(float)))
        zt, _ = infer(source_model.encoder, source_model.classifier, scaler.transform(target[names].to_numpy(float)))
        source_linear_head = fit_source_linear(ztr, tr, seed=7000 + int(load))
        linear = aggregate(te, linear_probabilities(source_linear_head, zte), "source_linear_fold")
        coral_transform = fit_coral_transform(ztr, tr.file_id, zt, target.file_id, epsilon=1e-4)
        coral_head = fit_source_linear(apply_coral(ztr, coral_transform), tr, seed=8000 + int(load))
        coral = aggregate(te, linear_probabilities(coral_head, apply_coral(zte, coral_transform)), "coral_fold")
        source_mlp = aggregate(te, mlp_p, "source_mlp_fold")
        for method, table in (("source_mlp", source_mlp), ("source_linear", linear), ("coral", coral)):
            table = table.merge(te.groupby("file_id").label.first().rename("true_label"), on="file_id")
            for _, row in table.iterrows():
                rows.append({"load": int(load), "method": method, "file_id": row.file_id, "true_label": row.true_label,
                             "predicted_label": row.predicted_label})
    result = pd.DataFrame(rows)
    summary = []
    for method, block in result.groupby("method"):
        y, p = block.true_label.to_numpy(), block.predicted_label.to_numpy()
        summary.append({"method": method, "macro_f1": f1_score(y, p, labels=LABELS, average="macro", zero_division=0),
                        "balanced_accuracy": balanced_accuracy_score(y, p),
                        **{f"recall_{k}": v for k, v in zip(LABELS, recall_score(y, p, labels=LABELS, average=None, zero_division=0))}})
    table = pd.DataFrame(summary).set_index("method")
    against_linear = float(table.loc["coral", "macro_f1"] - table.loc["source_linear", "macro_f1"])
    against_mlp = float(table.loc["coral", "macro_f1"] - table.loc["source_mlp", "macro_f1"])
    return result, pd.DataFrame(summary), {
        "coral_minus_source_linear_macro_f1": against_linear,
        "coral_minus_source_mlp_macro_f1": against_mlp,
        "negative_transfer_flag": bool(against_linear < -0.05 or against_mlp < -0.05),
        "threshold": -0.05, "evaluation_unit": "held-out source raw MAT file",
        "target_labels_used": False,
    }


def subsample_agreement(frame: pd.DataFrame, probabilities: np.ndarray, repeats: int = 20, fraction: float = .8, seed: int = 2025) -> dict[str, float]:
    rng = np.random.default_rng(seed); out = {}
    for file_id, idx in frame.groupby("file_id", sort=True).indices.items():
        p = probabilities[np.asarray(idx)]; full = int(p.mean(axis=0).argmax()); keep = max(1, round(len(p) * fraction))
        out[str(file_id)] = float(np.mean([int(p[rng.choice(len(p), keep, replace=False)].mean(axis=0).argmax()) == full for _ in range(repeats)]))
    return out


def collapse(table: pd.DataFrame) -> dict:
    counts = table.predicted_label.value_counts().reindex(LABELS, fill_value=0)
    fractions = counts / len(table)
    return {"class_count": counts.to_dict(), "max_class_fraction": float(fractions.max()),
            "effective_class_count": float(math.exp(-np.sum(fractions[fractions > 0] * np.log(fractions[fractions > 0])))),
            "collapse_warning": bool(fractions.max() > .75)}


def make_figures(output: Path, source_z: np.ndarray, target_z: np.ndarray, dann_source_z: np.ndarray, dann_target_z: np.ndarray,
                 metric_table: pd.DataFrame, history: pd.DataFrame, final: pd.DataFrame):
    figdir = output / "figures"; figdir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "svg.fonttype": "none"})
    pca = PCA(n_components=2, random_state=2025).fit(np.vstack([source_z, target_z, dann_source_z, dann_target_z]))
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.4))
    for ax, zs, zt, title in zip(axes, (source_z, dann_source_z), (target_z, dann_target_z), ("Frozen Transfer20", "After DANN")):
        ax.scatter(*pca.transform(zs).T, s=6, alpha=.35, label="source"); ax.scatter(*pca.transform(zt).T, s=8, alpha=.45, label="target")
        ax.set_title(title); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(figdir / "embedding_before_after.png", dpi=250); fig.savefig(figdir / "embedding_before_after.svg"); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.2))
    axes[0].bar(metric_table.method, metric_table.mmd_file_level); axes[0].set_title("File-level MMD")
    axes[1].bar(metric_table.method, metric_table.proxy_a_distance); axes[1].set_title("Proxy A-distance")
    fig.tight_layout(); fig.savefig(figdir / "domain_metrics.png", dpi=250); fig.savefig(figdir / "domain_metrics.svg"); plt.close(fig)
    mean_hist = history.groupby("epoch")[["source_classification_loss", "domain_loss", "domain_accuracy"]].mean()
    fig, ax = plt.subplots(figsize=(6, 3.2)); mean_hist.plot(ax=ax); ax.set_title("DANN five-seed mean training history"); fig.tight_layout(); fig.savefig(figdir / "dann_training.png", dpi=250); fig.savefig(figdir / "dann_training.svg"); plt.close(fig)
    label_map = {v: i for i, v in enumerate(LABELS)}
    labels = ["source_mlp_label", "source_linear_label", "coral_label", "dann_majority_vote_label", "final_candidate_label"]
    fig, ax = plt.subplots(figsize=(9, 3)); matrix = final[labels].apply(lambda column: column.map(label_map)).T.to_numpy()
    im = ax.imshow(matrix, vmin=0, vmax=3, aspect="auto", cmap="tab10"); ax.set_yticks(range(5), ["Source-MLP", "Source-linear", "CORAL", "DANN vote", "Final"]); ax.set_xticks(range(16), final.file_id); fig.colorbar(im, ax=ax, ticks=range(4)); fig.tight_layout(); fig.savefig(figdir / "method_comparison.png", dpi=250); fig.savefig(figdir / "method_comparison.svg"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 3)); final.set_index("file_id")[["window_vote_ratio", "subsample_agreement", "coral_epsilon_agreement", "coral_loto_agreement"]].plot.bar(ax=ax); ax.set_ylim(0, 1.05); fig.tight_layout(); fig.savefig(figdir / "target_stability.png", dpi=250); fig.savefig(figdir / "target_stability.svg"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 2.8)); colours = final.reliability_level.map({"High": "#2ca02c", "Medium": "#ffbf00", "Review": "#d62728"}); ax.scatter(final.file_id, final.final_candidate_label, c=colours, s=70); ax.set_title("Final candidate and internal stability level"); fig.tight_layout(); fig.savefig(figdir / "final_candidates.png", dpi=250); fig.savefig(figdir / "final_candidates.svg"); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--q1-dir", type=Path, default=Path("outputs/q1")); parser.add_argument("--model-dir", type=Path, default=Path("outputs/q2_refined/models"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/q3")); parser.add_argument("--epochs", type=int, default=30); parser.add_argument("--retention-epochs", type=int, default=24)
    parser.add_argument("--domain-weight", type=float, default=.10); args = parser.parse_args()
    if args.epochs < 1 or args.retention_epochs < 1 or args.domain_weight <= 0: raise ValueError("positive epochs and domain weight required")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "models").mkdir(parents=True, exist_ok=True)
    old_coral_path = args.output_dir / "coral_predictions.csv"
    old_coral = pd.read_csv(old_coral_path, encoding="utf-8-sig")[["file_id", "predicted_label"]].rename(columns={"predicted_label": "coral_label_before_revision"}) if old_coral_path.exists() else None
    source, target, names, audit = load_data(args.q1_dir, args.model_dir); write_json(args.output_dir / "input_audit.json", audit)
    encoder, classifier, scaler, schema = load_interface(args.model_dir, expected_features=names)
    zs, ps = infer(encoder, classifier, scaler.transform(source[names].to_numpy(float))); zt, pt = infer(encoder, classifier, scaler.transform(target[names].to_numpy(float)))
    source_mlp = aggregate(target, pt, "source_mlp")
    write_csv(args.output_dir / "source_only_predictions.csv", source_mlp)
    write_csv(args.output_dir / "source_mlp_predictions.csv", source_mlp)
    source_linear_head = fit_source_linear(zs, source)
    source_linear = aggregate(target, linear_probabilities(source_linear_head, zt), "source_linear")
    write_csv(args.output_dir / "source_linear_predictions.csv", source_linear)
    epsilon_tables = []
    for epsilon in (1e-5, 1e-4, 1e-3):
        transform = fit_coral_transform(zs, source.file_id, zt, target.file_id, epsilon)
        head = fit_source_linear(apply_coral(zs, transform), source)
        table = aggregate(target, linear_probabilities(head, zt), "coral")
        table["epsilon"] = epsilon
        epsilon_tables.append(table)
    epsilon_sensitivity = pd.concat(epsilon_tables, ignore_index=True)
    write_csv(args.output_dir / "coral_epsilon_sensitivity.csv", epsilon_sensitivity)
    coral = epsilon_sensitivity[np.isclose(epsilon_sensitivity.epsilon, 1e-4)].drop(columns="epsilon").reset_index(drop=True)
    write_csv(args.output_dir / "coral_predictions.csv", coral)
    coral_transform = fit_coral_transform(zs, source.file_id, zt, target.file_id, 1e-4)
    zsc = apply_coral(zs, coral_transform)
    formal_coral_probabilities = linear_probabilities(fit_source_linear(zsc, source), zt)
    if not np.allclose(coral[[f"prob_{label}" for label in LABELS]].to_numpy(), aggregate(target, formal_coral_probabilities, "check")[[f"prob_{label}" for label in LABELS]].to_numpy()):
        raise AssertionError("formal CORAL probabilities are not reproducible")
    epsilon_votes = epsilon_sensitivity.pivot(index="file_id", columns="epsilon", values="predicted_label").reindex(coral.file_id)
    coral_epsilon_agreement = {str(fid): float((row == coral.set_index("file_id").loc[str(fid), "predicted_label"]).mean()) for fid, row in epsilon_votes.iterrows()}
    loto_rows = []
    for file_id, indices in target.groupby("file_id", sort=True).indices.items():
        keep = target.file_id.astype(str).to_numpy() != str(file_id)
        transform = fit_coral_transform(zs, source.file_id, zt[keep], target.loc[keep, "file_id"], 1e-4)
        head = fit_source_linear(apply_coral(zs, transform), source)
        file_frame = target.iloc[np.asarray(indices)]
        loto_rows.append(aggregate(file_frame, linear_probabilities(head, zt[np.asarray(indices)]), "coral_loto"))
    coral_loto = pd.concat(loto_rows, ignore_index=True)
    write_csv(args.output_dir / "coral_leave_one_target_out.csv", coral_loto)
    metric_rows = [{"method": "source_mlp", **domain_metrics(source, target, zs, zt)},
                   {"method": "source_linear", **domain_metrics(source, target, zs, zt)},
                   {"method": "coral", **domain_metrics(source, target, zsc, zt)}]
    seed_tables, seed_probabilities, seed_embeddings_source, seed_embeddings_target, histories = [], [], [], [], []
    seed_window_tables = []
    for seed in SEEDS:
        dann, ds, history = train_dann(source, target, names, seed, args.epochs, args.domain_weight, initial=(encoder, classifier, scaler))
        zsd, _ = infer(dann.encoder, dann.classifier, ds.transform(source[names].to_numpy(float))); ztd, pdann = infer(dann.encoder, dann.classifier, ds.transform(target[names].to_numpy(float)))
        table = aggregate(target, pdann, f"dann_seed_{seed}"); table["seed"] = seed; seed_tables.append(table); seed_probabilities.append(pdann); seed_embeddings_source.append(zsd); seed_embeddings_target.append(ztd); histories.extend(history)
        window = target[["file_id", "window_id"]].copy(); window["seed"] = seed
        window["predicted_label"] = np.asarray(LABELS)[pdann.argmax(axis=1)]
        for i, label in enumerate(LABELS): window[f"prob_{label}"] = pdann[:, i]
        seed_window_tables.append(window)
        torch.save(dann.encoder.state_dict(), args.output_dir / "models" / f"dann_seed_{seed}_encoder.pth"); torch.save(dann.classifier.state_dict(), args.output_dir / "models" / f"dann_seed_{seed}_classifier.pth")
    histories = pd.DataFrame(histories); write_csv(args.output_dir / "dann_training_history.csv", histories)
    per_seed = pd.concat(seed_tables, ignore_index=True); write_csv(args.output_dir / "dann_seed_predictions.csv", per_seed)
    write_csv(args.output_dir / "dann_window_predictions_by_seed.csv", pd.concat(seed_window_tables, ignore_index=True))
    mean_prob = np.mean(seed_probabilities, axis=0); dann_consensus = aggregate(target, mean_prob, "dann_mean_probability")
    consensus_window = target[["file_id", "window_id"]].copy(); consensus_window["predicted_label"] = np.asarray(LABELS)[mean_prob.argmax(axis=1)]
    for i, label in enumerate(LABELS): consensus_window[f"prob_{label}"] = mean_prob[:, i]
    write_csv(args.output_dir / "dann_window_predictions_mean_probability.csv", consensus_window)
    votes = per_seed.pivot(index="file_id", columns="seed", values="predicted_label").reindex(dann_consensus.file_id)
    dann_consensus["dann_seed_agreement"] = [float((row == row.mode().iloc[0]).mean()) for _, row in votes.iterrows()]
    dann_consensus["dann_majority_vote_label"] = [row.mode().iloc[0] for _, row in votes.iterrows()]
    dann_consensus["dann_mean_probability_label"] = dann_consensus.predicted_label
    write_csv(args.output_dir / "dann_mean_probability_predictions.csv", dann_consensus)
    write_csv(args.output_dir / "dann_majority_vote_predictions.csv", dann_consensus[["file_id", "dann_majority_vote_label", "dann_seed_agreement"]])
    mean_zsd, mean_ztd = np.mean(seed_embeddings_source, axis=0), np.mean(seed_embeddings_target, axis=0)
    metric_rows.append({"method": "dann", **domain_metrics(source, target, mean_zsd, mean_ztd)})
    metrics = pd.DataFrame(metric_rows); write_csv(args.output_dir / "domain_metrics.csv", metrics)
    retention_rows, retention_summary, retention = coral_source_retention(source, target, names, args.retention_epochs)
    write_csv(args.output_dir / "coral_source_retention_predictions.csv", retention_rows); write_csv(args.output_dir / "coral_source_retention_summary.csv", retention_summary); write_json(args.output_dir / "coral_source_retention.json", retention)
    seed_collapse = {str(seed): collapse(block) for seed, block in per_seed.groupby("seed")}
    mean_collapse = collapse(dann_consensus.assign(predicted_label=dann_consensus.dann_mean_probability_label))
    majority_collapse = collapse(dann_consensus.assign(predicted_label=dann_consensus.dann_majority_vote_label))
    write_json(args.output_dir / "collapse_diagnostics.json", {"dann_by_seed": seed_collapse, "dann_mean_probability": mean_collapse, "dann_majority_vote": majority_collapse, "source_mlp": collapse(source_mlp), "source_linear": collapse(source_linear), "coral": collapse(coral)})
    # CORAL is the pre-specified formal method; DANN is never a selection gate.
    final = coral.copy()
    final = final.merge(source_mlp[["file_id", "predicted_label"]].rename(columns={"predicted_label": "source_mlp_label"}), on="file_id")
    final = final.merge(source_linear[["file_id", "predicted_label"]].rename(columns={"predicted_label": "source_linear_label"}), on="file_id")
    final = final.merge(dann_consensus[["file_id", "dann_mean_probability_label", "dann_majority_vote_label", "dann_seed_agreement"]], on="file_id")
    final = final.merge(coral_loto[["file_id", "predicted_label"]].rename(columns={"predicted_label": "coral_loto_label"}), on="file_id")
    final["coral_label"] = final.predicted_label
    final["final_candidate_label"] = final.coral_label
    final["final_method"] = "CORAL"
    final["subsample_agreement"] = final.file_id.map(subsample_agreement(target, formal_coral_probabilities, seed=2025))
    final["coral_epsilon_agreement"] = final.file_id.map(coral_epsilon_agreement)
    final["coral_loto_agreement"] = (final.coral_label == final.coral_loto_label).astype(float)
    final["source_linear_coral_agree"] = final.source_linear_label == final.coral_label
    final["method_agreement_count"] = final.apply(lambda r: int(pd.Series([r.source_mlp_label, r.source_linear_label, r.coral_label]).value_counts().max()), axis=1)
    high = (final.window_vote_ratio >= .75) & (final.subsample_agreement >= .85) & (final.coral_epsilon_agreement == 1.0) & (final.coral_loto_agreement == 1.0) & final.source_linear_coral_agree & (final.probability_margin >= .10)
    medium = (final.window_vote_ratio >= .60) & (final.subsample_agreement >= .75) & (final.coral_epsilon_agreement >= 2/3) & (final.coral_loto_agreement == 1.0) & (final.probability_margin >= .05)
    final["reliability_level"] = np.where(high, "High", np.where(medium, "Medium", "Review"))
    final["review_required"] = final.reliability_level.eq("Review") | retention["negative_transfer_flag"]
    final = final[["file_id", "source_mlp_label", "source_linear_label", "coral_label", "dann_mean_probability_label", "dann_majority_vote_label", "final_candidate_label", "final_method", "confidence", *[f"prob_{x}" for x in LABELS], "window_vote_ratio", "window_probability_std", "subsample_agreement", "coral_epsilon_agreement", "coral_loto_agreement", "source_linear_coral_agree", "method_agreement_count", "normalized_entropy", "probability_margin", "reliability_level", "review_required"]]
    if len(final) != 16 or sorted(final.file_id) != list("ABCDEFGHIJKLMNOP"): raise AssertionError("Final output must contain A--P exactly once")
    if not np.array_equal(final.final_candidate_label.to_numpy(), np.asarray(LABELS)[final[[f"prob_{x}" for x in LABELS]].to_numpy().argmax(axis=1)]): raise AssertionError("Final label/probability mismatch")
    write_csv(args.output_dir / "target_predictions_final.csv", final)
    changes = final[["file_id", "coral_label"]].merge(old_coral, on="file_id", how="left") if old_coral is not None else final[["file_id", "coral_label"]].assign(coral_label_before_revision=np.nan)
    changes["label_changed"] = changes.coral_label != changes.coral_label_before_revision
    write_csv(args.output_dir / "coral_label_changes.csv", changes)
    write_csv(args.output_dir / "embeddings_source_before.csv", pd.DataFrame(zs)); write_csv(args.output_dir / "embeddings_target_before.csv", pd.DataFrame(zt)); write_csv(args.output_dir / "embeddings_source_dann.csv", pd.DataFrame(mean_zsd)); write_csv(args.output_dir / "embeddings_target_dann.csv", pd.DataFrame(mean_ztd))
    joblib.dump(scaler, args.output_dir / "models" / "q3_input_scaler.pkl")
    write_json(args.output_dir / "models" / "q3_transfer_schema.json", {"features": names, "labels": list(LABELS), "input_dim": 20, "embedding_dim": 32})
    make_figures(args.output_dir, zs, zt, mean_zsd, mean_ztd, metrics, histories, final)
    metric_lines = ["| method | MMD | PAD | domain BA |", "|---|---:|---:|---:|"]
    metric_lines += [f"| {r.method} | {r.mmd_file_level:.4f} | {r.proxy_a_distance:.4f} | {r.domain_balanced_accuracy:.4f} |" for r in metrics.itertuples()]
    retention_lines = ["| method | Macro-F1 | BA | Recall N | Recall B | Recall IR | Recall OR |", "|---|---:|---:|---:|---:|---:|---:|"]
    retention_lines += [f"| {r.method} | {r.macro_f1:.4f} | {r.balanced_accuracy:.4f} | {r.recall_N:.4f} | {r.recall_B:.4f} | {r.recall_IR:.4f} | {r.recall_OR:.4f} |" for r in retention_summary.itertuples()]
    summary = ["# 第三问：CORAL 无监督跨域候选诊断结果", "", "目标 A–P 没有真值标签；本目录不含、也不报告目标域准确率。所有类别均是模型候选。", "", "## 正式方法", "", f"正式方法固定为 `CORAL`。其协方差采用窗口级、文件等权统计；线性头仅通过 sample_weight 进行一次类别/文件平衡。CORAL source-retention 负迁移标记为 `{retention['negative_transfer_flag']}`。DANN 仅为对照，其 mean-probability 与 majority-vote 标签均独立保存。", "", "## 域差异（文件级 embedding）", "", *metric_lines, "", "## CORAL 源域 LOLO 保持", "", *retention_lines, "", "MMD/PAD 的下降只说明域表征更难区分，不能证明目标类别正确。epsilon 与 LOTO 检查只用于稳定性评估，未用于调参或选标签。", "", "## 可复核输出", "", "- `target_predictions_final.csv`：最终 CORAL 候选及一致性、稳定性、复核标记；", "- `source_linear_predictions.csv`、`coral_source_retention_summary.csv`、`coral_epsilon_sensitivity.csv`、`coral_leave_one_target_out.csv`；", "- `coral_label_changes.csv`：修正前后标签对照；", "- `dann_*`：失败对照、训练历史和窗口级预测。"]
    (args.output_dir / "q3_summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    write_json(args.output_dir / "q3_config.json", {"seeds": SEEDS, "epochs": args.epochs, "retention_epochs": args.retention_epochs, "domain_weight": args.domain_weight, "final_method": "CORAL", "coral_epsilon": 1e-4, "coral_epsilon_sensitivity": [1e-5, 1e-4, 1e-3], "target_labels_used": False, "statistical_unit": "raw MAT file"})
    print(f"Question 3 outputs written to {args.output_dir}")


if __name__ == "__main__":
    main()
