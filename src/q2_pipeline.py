"""Question 2: source-domain diagnosis with nested, file-grouped validation.

This program reads Question 1 diagnostic features only.  It never reads target
A--P files and reports every metric at raw-MAT-file level.
"""
from __future__ import annotations

import argparse
import copy
import json
import pickle
import time
from collections import Counter
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score, recall_score)
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

LABELS = ("N", "B", "IR", "OR")
SEED = 2025
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
                     "svg.fonttype": "none", "pdf.fonttype": 42, "font.size": 9,
                     "axes.spines.right": False, "axes.spines.top": False})


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def save_df(path: Path, data: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(path, index=False, encoding="utf-8-sig")


def savefig(fig, base: Path) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(base.with_suffix(".png"), dpi=400, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


class FeatureEncoder(nn.Module):
    """The encoder that Question 3 can load without any diagnostic head."""
    def __init__(self, input_dim: int, dropout: float = 0.10) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, 64), nn.LayerNorm(64), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(64, 32),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.layers(values)


class SourceClassifier(nn.Module):
    def __init__(self, embedding_dim: int = 32, class_count: int = 4) -> None:
        super().__init__()
        self.linear = nn.Linear(embedding_dim, class_count)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.linear(values)


class SourceMLP(nn.Module):
    def __init__(self, input_dim: int, dropout: float = 0.10) -> None:
        super().__init__()
        self.encoder = FeatureEncoder(input_dim, dropout)
        self.classifier = SourceClassifier()

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encoder(values))


def load_inputs(q1_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, list[str], pd.DataFrame]:
    windows = pd.read_csv(q1_dir / "features_source_diagnostic.csv")
    metadata = pd.read_csv(q1_dir / "source_metadata.csv")
    names = json.loads((q1_dir / "feature_names_diagnostic.json").read_text(encoding="utf-8"))["features"]
    required = {"file_id", "label", "load", *names}
    if not required.issubset(set(windows.columns) | set(metadata.columns)):
        raise ValueError("Question 1 Diagnostic feature schema is incomplete")
    files = metadata[["file_id", "label", "load", "rpm", "fault_size", "fault_position"]].copy()
    if files.file_id.nunique() != 56 or len(files) != 56:
        raise ValueError("Question 2 expects exactly 56 unique formal source files")
    windows = windows.merge(files[["file_id", "load"]], on="file_id", how="left", validate="many_to_one")
    if windows.load.isna().any():
        raise ValueError("Window rows could not be linked to a source-file load")
    means = windows.groupby("file_id")[names].mean().add_suffix("_mean")
    stds = windows.groupby("file_id")[names].std(ddof=0).fillna(0).add_suffix("_std")
    file_features = files.set_index("file_id").join(means).join(stds).reset_index()
    if file_features[list(means.columns) + list(stds.columns)].isna().any().any():
        raise ValueError("Missing file-level diagnostic features")
    return windows, files, names, file_features


def class_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    recalls = recall_score(y_true, y_pred, labels=LABELS, average=None, zero_division=0)
    return {
        "macro_f1": f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": precision_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0),
        **{f"recall_{label}": value for label, value in zip(LABELS, recalls)},
    }


def inner_splits(files: pd.DataFrame, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    splitter = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=seed)
    return list(splitter.split(files, files.label, groups=files.file_id))


def outer_splits(files: pd.DataFrame, scheme: str) -> list[tuple[str, np.ndarray, np.ndarray]]:
    if scheme == "lolo":
        return [(f"load_{load}", np.flatnonzero(files.load.astype(str).to_numpy() != str(load)),
                 np.flatnonzero(files.load.astype(str).to_numpy() == str(load))) for load in range(4)]
    splitter = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=SEED)
    return [(f"group_{fold + 1}", train, test) for fold, (train, test) in enumerate(splitter.split(files, files.label, files.file_id))]


def build_sklearn_model(name: str):
    if name == "logistic_regression":
        return Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=5000, class_weight="balanced", random_state=SEED))]), {"model__C": [0.1, 1.0, 10.0]}
    if name == "rbf_svm":
        return Pipeline([("scaler", StandardScaler()), ("model", SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=SEED))]), {"model__C": [0.3, 1.0, 3.0], "model__gamma": ["scale", 0.01]}
    if name == "random_forest":
        return Pipeline([("model", RandomForestClassifier(n_estimators=500, class_weight="balanced", random_state=SEED, n_jobs=1))]), {"model__max_features": ["sqrt", 0.5], "model__min_samples_leaf": [1, 2]}
    raise ValueError(name)


def tune_sklearn(name: str, x: np.ndarray, y: np.ndarray, file_ids: np.ndarray, seed: int):
    model, grid = build_sklearn_model(name)
    file_frame = pd.DataFrame({"file_id": file_ids, "label": y})
    search = GridSearchCV(model, grid, scoring="f1_macro", cv=inner_splits(file_frame, seed), n_jobs=1, refit=True)
    search.fit(x, y, groups=file_ids)
    return search.best_estimator_, search.best_params_


def align_probabilities(probabilities: np.ndarray, classes: np.ndarray) -> np.ndarray:
    aligned = np.zeros((len(probabilities), len(LABELS)))
    for column, label in enumerate(classes):
        aligned[:, LABELS.index(str(label))] = probabilities[:, column]
    return aligned


def window_weights(frame: pd.DataFrame) -> np.ndarray:
    counts = frame.groupby("file_id").size()
    labels = frame.groupby("file_id").label.first()
    class_files = labels.value_counts()
    return np.asarray([1.0 / (len(LABELS) * class_files[row.label] * counts[row.file_id]) for _, row in frame.iterrows()], dtype=float)


def fit_weighted_scaler(frame: pd.DataFrame, features: list[str]) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(frame[features].to_numpy(float), sample_weight=window_weights(frame))
    return scaler


def aggregate_window_probabilities(frame: pd.DataFrame, probabilities: np.ndarray) -> pd.DataFrame:
    table = frame[["file_id", "label", "load", "rpm"]].copy()
    for index, label in enumerate(LABELS):
        table[f"prob_{label}"] = probabilities[:, index]
    grouped = table.groupby("file_id", as_index=False).agg({"label": "first", "load": "first", "rpm": "first", **{f"prob_{label}": "mean" for label in LABELS}})
    values = grouped[[f"prob_{label}" for label in LABELS]].to_numpy()
    grouped["predicted_label"] = np.asarray(LABELS)[np.argmax(values, axis=1)]
    grouped["confidence"] = values.max(axis=1)
    return grouped


def mlp_probabilities(model: SourceMLP, values: np.ndarray) -> np.ndarray:
    model.eval()
    result = []
    with torch.no_grad():
        for start in range(0, len(values), 256):
            logits = model(torch.tensor(values[start:start + 256], dtype=torch.float32))
            result.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.vstack(result)


def train_mlp(train: pd.DataFrame, validation: pd.DataFrame | None, features: list[str], config: dict, seed: int, epochs: int, patience: int) -> tuple[SourceMLP, StandardScaler, list[dict], int]:
    set_seed(seed)
    scaler = fit_weighted_scaler(train, features)
    x = scaler.transform(train[features].to_numpy(float)).astype(np.float32)
    y = np.asarray([LABELS.index(label) for label in train.label], dtype=np.int64)
    sampler = WeightedRandomSampler(torch.tensor(window_weights(train), dtype=torch.double), num_samples=len(train), replacement=True, generator=torch.Generator().manual_seed(seed))
    loader = DataLoader(TensorDataset(torch.tensor(x), torch.tensor(y)), batch_size=64, sampler=sampler, num_workers=0)
    model = SourceMLP(len(features), config["dropout"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=config["weight_decay"])
    history, best_state, best_epoch, best_score, best_ba, waiting = [], None, 1, -np.inf, -np.inf, 0
    validation_x = scaler.transform(validation[features].to_numpy(float)).astype(np.float32) if validation is not None else None
    for epoch in range(1, epochs + 1):
        model.train(); total_loss = 0.0
        for values, labels in loader:
            logits = model(values); loss = F.cross_entropy(logits, labels)
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step(); total_loss += float(loss.detach())
        row = {"epoch": epoch, "train_loss": total_loss / max(1, len(loader)), "val_macro_f1": np.nan, "val_balanced_accuracy": np.nan}
        if validation is not None:
            validation_files = aggregate_window_probabilities(validation, mlp_probabilities(model, validation_x))
            score = class_metrics(validation_files.label.to_numpy(), validation_files.predicted_label.to_numpy())
            row.update({"val_macro_f1": score["macro_f1"], "val_balanced_accuracy": score["balanced_accuracy"]})
            if (score["macro_f1"], score["balanced_accuracy"]) > (best_score, best_ba):
                best_state, best_epoch, best_score, best_ba, waiting = copy.deepcopy(model.state_dict()), epoch, score["macro_f1"], score["balanced_accuracy"], 0
            else:
                waiting += 1
                if waiting >= patience:
                    history.append(row); break
        history.append(row)
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, scaler, history, best_epoch


def tune_mlp(train_files: pd.DataFrame, all_windows: pd.DataFrame, features: list[str], seed: int) -> tuple[dict, int, list[dict]]:
    candidates = [{"dropout": 0.10, "weight_decay": 1e-4}, {"dropout": 0.20, "weight_decay": 1e-3}]
    records, choices = [], []
    for candidate_id, candidate in enumerate(candidates):
        fold_scores, epochs = [], []
        for fold, (inner_train, inner_valid) in enumerate(inner_splits(train_files, seed + candidate_id)):
            train_ids = set(train_files.iloc[inner_train].file_id)
            valid_ids = set(train_files.iloc[inner_valid].file_id)
            train = all_windows[all_windows.file_id.isin(train_ids)]
            valid = all_windows[all_windows.file_id.isin(valid_ids)]
            model, scaler, history, best_epoch = train_mlp(train, valid, features, candidate, seed + candidate_id * 20 + fold, epochs=100, patience=15)
            final = max((row for row in history if np.isfinite(row["val_macro_f1"])), key=lambda row: (row["val_macro_f1"], row["val_balanced_accuracy"]))
            for row in history:
                records.append({"stage": "inner_tuning", "candidate": candidate_id, "inner_fold": fold + 1, **candidate, **row})
            fold_scores.append((final["val_macro_f1"], final["val_balanced_accuracy"])); epochs.append(best_epoch)
        choices.append({"candidate": candidate_id, **candidate, "mean_macro_f1": float(np.nanmean([score[0] for score in fold_scores])), "mean_balanced_accuracy": float(np.nanmean([score[1] for score in fold_scores])), "epochs": int(np.median(epochs))})
    selected = sorted(choices, key=lambda item: (item["mean_macro_f1"], item["mean_balanced_accuracy"]), reverse=True)[0]
    return {"dropout": selected["dropout"], "weight_decay": selected["weight_decay"]}, selected["epochs"], records


def model_size_bytes(model) -> int:
    if isinstance(model, SourceMLP):
        return sum(parameter.numel() * parameter.element_size() for parameter in model.parameters())
    return len(pickle.dumps(model))


def evaluate_model(name: str, train_files: pd.DataFrame, test_files: pd.DataFrame, all_windows: pd.DataFrame, file_features: pd.DataFrame, file_feature_columns: list[str], window_feature_columns: list[str], seed: int):
    train_ids, test_ids = set(train_files.file_id), set(test_files.file_id)
    if train_ids & test_ids:
        raise AssertionError("file leakage detected")
    started = time.perf_counter(); histories, importances = [], []
    if name == "mlp":
        config, epochs, tuning_history = tune_mlp(train_files, all_windows, window_feature_columns, seed)
        histories.extend(tuning_history)
        train_windows = all_windows[all_windows.file_id.isin(train_ids)]
        test_windows = all_windows[all_windows.file_id.isin(test_ids)]
        model, scaler, history, _ = train_mlp(train_windows, None, window_feature_columns, config, seed + 500, epochs=epochs, patience=epochs)
        for row in history:
            histories.append({"stage": "outer_fit", "candidate": -1, "inner_fold": -1, **config, **row})
        infer_started = time.perf_counter(); values = scaler.transform(test_windows[window_feature_columns].to_numpy(float)).astype(np.float32)
        predicted = aggregate_window_probabilities(test_windows, mlp_probabilities(model, values)); inference_seconds = time.perf_counter() - infer_started
        params = {**config, "epochs": epochs}
    else:
        train_x = file_features[file_features.file_id.isin(train_ids)][file_feature_columns].to_numpy(float)
        train_y = file_features[file_features.file_id.isin(train_ids)].label.to_numpy()
        train_groups = file_features[file_features.file_id.isin(train_ids)].file_id.to_numpy()
        test_x = file_features[file_features.file_id.isin(test_ids)][file_feature_columns].to_numpy(float)
        test_table = file_features[file_features.file_id.isin(test_ids)][["file_id", "label", "load", "rpm"]].copy()
        model, params = tune_sklearn(name, train_x, train_y, train_groups, seed)
        infer_started = time.perf_counter(); probabilities = align_probabilities(model.predict_proba(test_x), model.named_steps["model"].classes_); inference_seconds = time.perf_counter() - infer_started
        for index, label in enumerate(LABELS): test_table[f"prob_{label}"] = probabilities[:, index]
        test_table["predicted_label"] = np.asarray(LABELS)[np.argmax(probabilities, axis=1)]
        test_table["confidence"] = probabilities.max(axis=1)
        predicted = test_table
        if name == "random_forest":
            values = model.named_steps["model"].feature_importances_
            importances = [{"feature": item, "importance": float(value)} for item, value in zip(file_feature_columns, values)]
    elapsed = time.perf_counter() - started
    metrics = class_metrics(predicted.label.to_numpy(), predicted.predicted_label.to_numpy())
    return predicted, metrics, params, elapsed, inference_seconds, model_size_bytes(model), histories, importances


def evaluate_scheme(scheme: str, files: pd.DataFrame, windows: pd.DataFrame, file_features: pd.DataFrame, file_feature_columns: list[str], window_feature_columns: list[str]):
    results, predictions, histories, importances = [], [], [], []
    for fold, (_, train_index, test_index) in enumerate(outer_splits(files, scheme), start=1):
        train_files, test_files = files.iloc[train_index], files.iloc[test_index]
        for model_index, model_name in enumerate(("logistic_regression", "rbf_svm", "random_forest", "mlp")):
            predicted, metrics, params, train_seconds, infer_seconds, size, history, importance = evaluate_model(model_name, train_files, test_files, windows, file_features, file_feature_columns, window_feature_columns, SEED + fold * 100 + model_index)
            results.append({"validation_scheme": scheme, "fold": fold, "model": model_name, "test_load": ",".join(sorted(test_files.load.astype(str).unique())), "n_train_files": len(train_files), "n_test_files": len(test_files), "file_leakage_count": len(set(train_files.file_id) & set(test_files.file_id)), "best_params": json.dumps(params, sort_keys=True), "training_seconds": train_seconds, "inference_seconds": infer_seconds, "model_size_bytes": size, **metrics})
            for _, row in predicted.iterrows():
                predictions.append({"validation_scheme": scheme, "fold": fold, "model": model_name, "file_id": row.file_id, "true_label": row.label, "predicted_label": row.predicted_label, "test_load": row.load, "confidence": row.confidence, **{f"prob_{label}": row[f"prob_{label}"] for label in LABELS}})
            for item in history: histories.append({"validation_scheme": scheme, "outer_fold": fold, "model": model_name, **item})
            for item in importance: importances.append({"validation_scheme": scheme, "outer_fold": fold, "model": model_name, **item})
    return pd.DataFrame(results), pd.DataFrame(predictions), pd.DataFrame(histories), pd.DataFrame(importances)


def aggregate_results(results: pd.DataFrame, scheme: str) -> pd.DataFrame:
    rows = []
    metrics = ["macro_f1", "balanced_accuracy", "accuracy", "macro_precision", "macro_recall", *[f"recall_{label}" for label in LABELS]]
    for model, group in results.groupby("model", sort=False):
        row = {"model": model, "validation_scheme": scheme, "n_folds": len(group), "mean_training_seconds": group.training_seconds.mean(), "mean_model_size_bytes": group.model_size_bytes.mean()}
        for metric in metrics:
            row[f"mean_{metric}"] = group[metric].mean(); row[f"std_{metric}"] = group[metric].std(ddof=1)
        row["min_mean_class_recall"] = min(row[f"mean_recall_{label}"] for label in LABELS)
        rows.append(row)
    return pd.DataFrame(rows)


def choose_best(lolo_summary: pd.DataFrame) -> str:
    maximum = lolo_summary.mean_macro_f1.max()
    candidates = lolo_summary[lolo_summary.mean_macro_f1 >= maximum - 0.02].copy()
    candidates = candidates.sort_values(["mean_balanced_accuracy", "min_mean_class_recall", "std_macro_f1", "mean_model_size_bytes"], ascending=[False, False, True, True])
    return str(candidates.iloc[0].model)


def modal_params(results: pd.DataFrame, model: str) -> dict:
    selected = results[(results.validation_scheme == "lolo") & (results.model == model)]
    text = Counter(selected.best_params).most_common(1)[0][0]
    return json.loads(text)


def refit_and_save(best_model: str, files: pd.DataFrame, windows: pd.DataFrame, file_features: pd.DataFrame, file_feature_columns: list[str], window_feature_columns: list[str], lolo_results: pd.DataFrame, output_dir: Path):
    model_dir = output_dir / "models"; model_dir.mkdir(parents=True, exist_ok=True)
    model_info = {"best_model": best_model, "label_order": list(LABELS), "file_feature_names": file_feature_columns, "mlp_window_feature_names": window_feature_columns}
    params = modal_params(lolo_results, best_model)
    if best_model != "mlp":
        pipeline, _ = build_sklearn_model(best_model)
        pipeline.set_params(**params)
        pipeline.fit(file_features[file_feature_columns].to_numpy(float), file_features.label.to_numpy())
        path = model_dir / f"q2_best_{best_model}.joblib"; joblib.dump(pipeline, path)
        model_info.update({"best_model_path": path.name, "best_model_params": params})
    mlp_params = modal_params(lolo_results, "mlp")
    mlp_epochs = int(mlp_params.pop("epochs"))
    mlp, scaler, _, _ = train_mlp(windows, None, window_feature_columns, mlp_params, SEED + 900, epochs=mlp_epochs, patience=mlp_epochs)
    torch.save(mlp.encoder.state_dict(), model_dir / "q2_mlp_encoder.pth")
    torch.save(mlp.classifier.state_dict(), model_dir / "q2_mlp_classifier.pth")
    torch.save(mlp.state_dict(), model_dir / "q2_mlp_full_candidate.pth")
    if best_model == "mlp":
        torch.save(mlp.state_dict(), model_dir / "q2_best_mlp.pth")
    joblib.dump(scaler, model_dir / "q2_mlp_scaler.pkl")
    joblib.dump(list(LABELS), model_dir / "q2_label_encoder.pkl")
    (model_dir / "q2_feature_names.json").write_text(json.dumps({"features": window_feature_columns, "n_features": len(window_feature_columns)}, ensure_ascii=False, indent=2), encoding="utf-8")
    (model_dir / "q2_model_config.json").write_text(json.dumps({"input_dim": len(window_feature_columns), "encoder": [128, 64, 32], "activation": "GELU", "normalization": "LayerNorm", "dropout": mlp_params["dropout"], "weight_decay": mlp_params["weight_decay"], "epochs": mlp_epochs, "purpose": "Question 3 initialization candidate; final all-source refit has no independent test metric."}, ensure_ascii=False, indent=2), encoding="utf-8")
    model_info.update({"mlp_encoder_path": "q2_mlp_encoder.pth", "mlp_classifier_path": "q2_mlp_classifier.pth", "mlp_scaler_path": "q2_mlp_scaler.pkl", "mlp_candidate_params": {**mlp_params, "epochs": mlp_epochs}, "best_mlp_path": "q2_best_mlp.pth" if best_model == "mlp" else None})
    return model_info


def draw_figures(output_dir: Path, comparison: pd.DataFrame, lolo_results: pd.DataFrame, predictions: pd.DataFrame, importance: pd.DataFrame, history: pd.DataFrame, best_model: str) -> None:
    figures = output_dir / "figures"
    lolo = comparison[comparison.validation_scheme == "lolo"]
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
    for axis, metric, title in zip(axes, ("mean_macro_f1", "mean_balanced_accuracy"), ("LOLO Macro-F1", "LOLO balanced accuracy")):
        axis.bar(lolo.model, lolo[metric], color=["#4D4D4D", "#0F4D92", "#42949E", "#B64342"])
        axis.set(title=title, ylim=(0, 1), ylabel="Score"); axis.tick_params(axis="x", rotation=25)
    savefig(fig, figures / "model_performance_comparison")
    best_predictions = predictions[(predictions.validation_scheme == "lolo") & (predictions.model == best_model)]
    matrix = confusion_matrix(best_predictions.true_label, best_predictions.predicted_label, labels=LABELS, normalize="true")
    fig, axis = plt.subplots(figsize=(4, 3.6)); image = axis.imshow(matrix, vmin=0, vmax=1, cmap="Blues")
    axis.set(title=f"Best LOLO file-level confusion: {best_model}", xlabel="Predicted", ylabel="True", xticks=range(4), xticklabels=LABELS, yticks=range(4), yticklabels=LABELS)
    for row in range(4):
        for col in range(4): axis.text(col, row, f"{matrix[row, col]:.2f}", ha="center", va="center", color="white" if matrix[row, col] > .55 else "black")
    fig.colorbar(image, ax=axis, label="Recall"); savefig(fig, figures / "best_model_confusion_matrix")
    by_load = lolo_results[lolo_results.model == best_model].sort_values("fold")
    fig, axis = plt.subplots(figsize=(5.5, 3.3)); axis.plot(by_load.test_load, by_load.macro_f1, "o-", label="Macro-F1"); axis.plot(by_load.test_load, by_load.balanced_accuracy, "s-", label="Balanced accuracy"); axis.set(xlabel="Held-out load (hp)", ylabel="Score", ylim=(0, 1), title=f"{best_model}: LOLO by held-out load"); axis.legend(); savefig(fig, figures / "lolo_by_load")
    fig, axis = plt.subplots(figsize=(6.2, 3.3)); data = [lolo_results[lolo_results.model == model].macro_f1 for model in lolo.model]; axis.boxplot(data, tick_labels=lolo.model); axis.set(title="LOLO Macro-F1 stability", ylabel="Macro-F1", ylim=(0, 1)); axis.tick_params(axis="x", rotation=25); savefig(fig, figures / "model_stability_boxplot")
    rf = importance[(importance.model == "random_forest") & (importance.validation_scheme == "lolo")]
    if not rf.empty:
        top = rf.groupby("feature").importance.mean().sort_values(ascending=False).head(15).sort_values()
        fig, axis = plt.subplots(figsize=(6.3, 4.2)); axis.barh(top.index, top.values, color="#42949E"); axis.set(title="Random-forest feature importance (LOLO mean)", xlabel="Mean importance"); savefig(fig, figures / "rf_feature_importance")
    mlp_history = history[(history.model == "mlp") & (history.validation_scheme == "lolo") & (history.stage == "inner_tuning")]
    if not mlp_history.empty:
        curve = mlp_history.groupby("epoch")[["train_loss", "val_macro_f1"]].mean()
        fig, axes = plt.subplots(1, 2, figsize=(7, 3)); axes[0].plot(curve.index, curve.train_loss, color="#B64342"); axes[0].set(title="MLP inner-CV training loss", xlabel="Epoch", ylabel="Loss"); axes[1].plot(curve.index, curve.val_macro_f1, color="#0F4D92"); axes[1].set(title="MLP inner-CV file Macro-F1", xlabel="Epoch", ylabel="Macro-F1", ylim=(0, 1)); savefig(fig, figures / "mlp_training_curve")


def write_summary(output_dir: Path, files: pd.DataFrame, feature_names: list[str], comparison: pd.DataFrame, best_model: str, model_info: dict) -> None:
    lolo = comparison[comparison.validation_scheme == "lolo"].set_index("model")
    best = lolo.loc[best_model]
    text = f"""# 第二问总结

## 数据与无泄漏边界

使用第一问的 {len(feature_names)} 维 Diagnostic 特征。源域共有 {len(files)} 个原始 MAT 文件，类别数为 {files.label.value_counts().to_dict()}。传统模型以每文件 mean+std 的 {2 * len(feature_names)} 维特征输入；MLP 以窗口特征训练但按文件平衡采样、按文件平均概率评价。

LOLO 和辅助 Stratified Group K-Fold 的所有 Scaler、调参与训练均只在各外层训练文件内完成。结果仅说明源域跨载荷/跨文件的泛化能力，不估计 A–P 目标域准确率。

## 最优模型

按预注册的 LOLO 选择规则，最终模型为 **{best_model}**：Macro-F1={best.mean_macro_f1:.3f}±{best.std_macro_f1:.3f}，Balanced Accuracy={best.mean_balanced_accuracy:.3f}±{best.std_balanced_accuracy:.3f}，最低平均类别 Recall={best.min_mean_class_recall:.3f}。

## 第三问接口

保存了最优诊断模型（如适用）以及独立 MLP Encoder、Classifier、Scaler、特征 schema 与标签顺序。MLP 全源域重训权重仅用于第三问初始化，不是新的独立测试结果。

详细表格见 `model_comparison.csv`、`lolo_results.csv`、`group_cv_results.csv` 和 `best_model_summary.json`。
"""
    (output_dir / "q2_summary.md").write_text(text, encoding="utf-8")
    (output_dir / "best_model_summary.json").write_text(json.dumps({"selection_scheme": "LOLO nested group CV", "selection_rule": "highest Macro-F1; within 0.02 compare BA, minimum class recall, stability, complexity", "best_model": best_model, "lolo_summary": best.to_dict(), "third_question_interface": model_info, "scope_boundary": "No target labels or target accuracy were used."}, ensure_ascii=False, indent=2, default=float), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--q1-dir", type=Path, default=Path("outputs") / "q1")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs") / "q2")
    args = parser.parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    windows, files, diagnostic_names, file_features = load_inputs(args.q1_dir)
    file_columns = [f"{name}_{stat}" for name in diagnostic_names for stat in ("mean", "std")]
    config = {"random_seed": SEED, "feature_file": str(args.q1_dir / "features_source_diagnostic.csv"), "n_features": len(diagnostic_names), "file_level_features": len(file_columns), "window": 16384, "hop": 8192, "split_methods": {"primary": "nested LOLO", "auxiliary": "nested StratifiedGroupKFold(4)", "inner": "StratifiedGroupKFold(3)"}, "models": ["logistic_regression", "rbf_svm", "random_forest", "mlp"], "class_balance": "MLP window sampling weight = 1/(4 * files_in_class * windows_in_file); sklearn class_weight=balanced", "selected_metric": "LOLO Macro-F1"}
    (args.output_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    all_results, all_predictions, all_history, all_importance = [], [], [], []
    for scheme in ("lolo", "group_cv"):
        results, predictions, history, importance = evaluate_scheme(scheme, files, windows, file_features, file_columns, diagnostic_names)
        all_results.append(results); all_predictions.append(predictions); all_history.append(history); all_importance.append(importance)
    results = pd.concat(all_results, ignore_index=True); predictions = pd.concat(all_predictions, ignore_index=True); history = pd.concat(all_history, ignore_index=True); importance = pd.concat(all_importance, ignore_index=True)
    lolo_results, group_results = results[results.validation_scheme == "lolo"], results[results.validation_scheme == "group_cv"]
    lolo_summary, group_summary = aggregate_results(lolo_results, "lolo"), aggregate_results(group_results, "group_cv")
    comparison = lolo_summary.merge(group_summary.drop(columns=["validation_scheme"]), on="model", suffixes=("_lolo", "_group_cv"))
    best_model = choose_best(lolo_summary)
    save_df(args.output_dir / "lolo_results.csv", lolo_results); save_df(args.output_dir / "group_cv_results.csv", group_results); save_df(args.output_dir / "predictions_file_level.csv", predictions); save_df(args.output_dir / "training_history.csv", history); save_df(args.output_dir / "feature_importance.csv", importance); save_df(args.output_dir / "model_comparison.csv", comparison)
    best_predictions = predictions[(predictions.validation_scheme == "lolo") & (predictions.model == best_model)]
    report = pd.DataFrame(classification_report(best_predictions.true_label, best_predictions.predicted_label, labels=LABELS, output_dict=True, zero_division=0)).T.reset_index().rename(columns={"index": "class_or_average"})
    save_df(args.output_dir / "classification_report.csv", report)
    matrix = pd.DataFrame(confusion_matrix(best_predictions.true_label, best_predictions.predicted_label, labels=LABELS), index=LABELS, columns=LABELS); matrix.index.name = "true_label"; save_df(args.output_dir / "confusion_matrix.csv", matrix.reset_index())
    model_info = refit_and_save(best_model, files, windows, file_features, file_columns, diagnostic_names, lolo_results, args.output_dir)
    draw_figures(args.output_dir, pd.concat([lolo_summary, group_summary], ignore_index=True), lolo_results, predictions, importance, history, best_model)
    write_summary(args.output_dir, files, diagnostic_names, pd.concat([lolo_summary, group_summary], ignore_index=True), best_model, model_info)
    print(json.dumps({"files": len(files), "windows": len(windows), "diagnostic_features": len(diagnostic_names), "best_model": best_model, "lolo_file_leakage_max": int(lolo_results.file_leakage_count.max())}, ensure_ascii=False))


if __name__ == "__main__":
    main()
