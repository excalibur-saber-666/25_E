"""Question 2: Diagnostic26 source-domain diagnosis (MLP only).

The independent unit is an original MAT file. Windows are used only during
training; their probabilities are averaged once per file for every report.
Neither this module nor its validation reads target-domain files or labels.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             classification_report, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

LABELS = ("N", "B", "IR", "OR")
SEED = 2025


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def save_df(path: Path, table: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False, encoding="utf-8-sig")


def savefig(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


class FeatureEncoder(nn.Module):
    """Frozen 26 -> 128 -> 64 -> 32 encoder for a later Q3 rebuild."""
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
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(32, len(LABELS))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.linear(values)


class SourceMLP(nn.Module):
    def __init__(self, input_dim: int, dropout: float = 0.10) -> None:
        super().__init__()
        self.encoder = FeatureEncoder(input_dim, dropout)
        self.classifier = SourceClassifier()

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encoder(values))


def load_inputs(q1_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Load only Q1 Diagnostic26 source artefacts; never open target data."""
    windows = pd.read_csv(q1_dir / "features_source_diagnostic.csv")
    metadata = pd.read_csv(q1_dir / "source_metadata.csv")
    names = json.loads((q1_dir / "feature_names_diagnostic.json").read_text(encoding="utf-8"))["features"]
    required = {"file_id", "label", *names}
    if not required.issubset(windows.columns) or not {"file_id", "label", "load"}.issubset(metadata.columns):
        raise ValueError("Q1 Diagnostic26 source schema is incomplete")
    files = metadata[["file_id", "label", "load", "rpm", "fault_size", "fault_position"]].copy()
    if len(files) != 56 or files.file_id.nunique() != 56 or set(files.label) != set(LABELS):
        raise ValueError("Expected 56 formal source MAT files in four classes")
    windows = windows.merge(files[["file_id", "load"]], on="file_id", how="left", validate="many_to_one")
    if windows.load.isna().any():
        raise ValueError("A source window could not be linked to its original MAT file")
    return windows, files, names


def class_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    recalls = recall_score(y_true, y_pred, labels=LABELS, average=None, zero_division=0)
    return {
        "macro_f1": f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": precision_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0),
        **{f"recall_{label}": float(value) for label, value in zip(LABELS, recalls)},
    }


def outer_splits(files: pd.DataFrame, scheme: str):
    if scheme == "lolo":
        for load in sorted(files.load.unique()):
            yield f"load_{load}", np.flatnonzero(files.load.to_numpy() != load), np.flatnonzero(files.load.to_numpy() == load)
        return
    splitter = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=SEED)
    for fold, (train, test) in enumerate(splitter.split(files, files.label, groups=files.file_id), 1):
        yield f"group_{fold}", train, test


def inner_splits(files: pd.DataFrame, seed: int):
    splitter = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=seed)
    return list(splitter.split(files, files.label, groups=files.file_id))


def window_weights(frame: pd.DataFrame) -> np.ndarray:
    """Each class and each file within a class has equal total training mass."""
    per_file = frame.groupby(["label", "file_id"]).size().rename("n_windows").reset_index()
    per_class = per_file.groupby("label").size().rename("n_files").reset_index()
    joined = frame.merge(per_file, on=["label", "file_id"], validate="many_to_one").merge(per_class, on="label", validate="many_to_one")
    return 1.0 / (len(LABELS) * joined.n_files.to_numpy(float) * joined.n_windows.to_numpy(float))


def fit_weighted_scaler(frame: pd.DataFrame, features: list[str]) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(frame[features].to_numpy(float), sample_weight=window_weights(frame))
    return scaler


def aggregate_window_probabilities(frame: pd.DataFrame, probabilities: np.ndarray) -> pd.DataFrame:
    columns = ["file_id", "label", "load"] + (["rpm"] if "rpm" in frame else [])
    table = frame[columns].copy()
    for index, label in enumerate(LABELS):
        table[f"prob_{label}"] = probabilities[:, index]
    aggregation = {"label": "first", "load": "first", **({"rpm": "first"} if "rpm" in table else {}), **{f"prob_{label}": "mean" for label in LABELS}}
    grouped = table.groupby("file_id", as_index=False).agg(aggregation)
    probs = grouped[[f"prob_{label}" for label in LABELS]].to_numpy(float)
    grouped["predicted_label"] = np.asarray(LABELS)[probs.argmax(axis=1)]
    grouped["confidence"] = probs.max(axis=1)
    return grouped


def mlp_probabilities(model: SourceMLP, values: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return torch.softmax(model(torch.tensor(values, dtype=torch.float32)), dim=1).cpu().numpy()


def train_mlp(train: pd.DataFrame, validation: pd.DataFrame | None, features: list[str], config: dict, seed: int, epochs: int, patience: int):
    set_seed(seed)
    scaler = fit_weighted_scaler(train, features)
    x = scaler.transform(train[features].to_numpy(float)).astype(np.float32)
    y = train.label.map({label: i for i, label in enumerate(LABELS)}).to_numpy(np.int64)
    sampler = WeightedRandomSampler(torch.as_tensor(window_weights(train), dtype=torch.double), len(train), replacement=True, generator=torch.Generator().manual_seed(seed))
    loader = DataLoader(TensorDataset(torch.from_numpy(x), torch.from_numpy(y)), batch_size=min(64, len(train)), sampler=sampler)
    model = SourceMLP(len(features), config["dropout"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=config["weight_decay"])
    history, best_state, best_score, stale = [], None, -np.inf, 0
    for epoch in range(1, epochs + 1):
        model.train(); losses = []
        for bx, by in loader:
            optimizer.zero_grad(); loss = F.cross_entropy(model(bx), by); loss.backward(); optimizer.step(); losses.append(float(loss.detach()))
        row = {"epoch": epoch, "train_loss": float(np.mean(losses))}
        if validation is not None:
            values = scaler.transform(validation[features].to_numpy(float)).astype(np.float32)
            predicted = aggregate_window_probabilities(validation, mlp_probabilities(model, values))
            score = class_metrics(predicted.label.to_numpy(), predicted.predicted_label.to_numpy())["macro_f1"]
            row["val_macro_f1"] = score
            if score > best_score:
                best_score, best_state, stale = score, {k: v.detach().clone() for k, v in model.state_dict().items()}, 0
            else:
                stale += 1
                if stale >= patience:
                    history.append(row); break
        history.append(row)
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, scaler, history, epoch


def tune_mlp(train_files: pd.DataFrame, windows: pd.DataFrame, features: list[str], seed: int):
    candidates = [{"dropout": d, "weight_decay": wd} for d in (0.05, 0.10) for wd in (1e-4, 1e-3)]
    records = []
    for candidate_id, config in enumerate(candidates):
        scores, histories = [], []
        for fold, (tr, va) in enumerate(inner_splits(train_files, seed + candidate_id), 1):
            tr_ids, va_ids = train_files.iloc[tr].file_id, train_files.iloc[va].file_id
            train_windows, valid_windows = windows[windows.file_id.isin(tr_ids)], windows[windows.file_id.isin(va_ids)]
            model, scaler, history, _ = train_mlp(train_windows, valid_windows, features, config, seed + candidate_id * 20 + fold, 60, 10)
            predicted = aggregate_window_probabilities(valid_windows, mlp_probabilities(model, scaler.transform(valid_windows[features]).astype(np.float32)))
            scores.append(class_metrics(predicted.label.to_numpy(), predicted.predicted_label.to_numpy())["macro_f1"])
            histories.extend([{**item, "candidate": candidate_id, "inner_fold": fold, **config} for item in history])
        records.append((float(np.mean(scores)), config, histories))
    return max(records, key=lambda item: item[0])[1:]


def evaluate_scheme(scheme: str, files: pd.DataFrame, windows: pd.DataFrame, features: list[str]):
    results, predictions, history = [], [], []
    for fold, (name, tr, te) in enumerate(outer_splits(files, scheme), 1):
        train_files, test_files = files.iloc[tr], files.iloc[te]
        config, tuning_history = tune_mlp(train_files, windows, features, SEED + fold * 100)
        history.extend([{**row, "validation_scheme": scheme, "outer_fold": fold, "stage": "inner_tuning"} for row in tuning_history])
        started = time.perf_counter()
        model, scaler, final_history, epochs = train_mlp(windows[windows.file_id.isin(train_files.file_id)], None, features, config, SEED + fold * 100 + 50, 60, 60)
        train_seconds = time.perf_counter() - started
        history.extend([{**row, "validation_scheme": scheme, "outer_fold": fold, "stage": "outer_fit", **config} for row in final_history])
        test_windows = windows[windows.file_id.isin(test_files.file_id)]
        started = time.perf_counter(); predicted = aggregate_window_probabilities(test_windows, mlp_probabilities(model, scaler.transform(test_windows[features]).astype(np.float32))); inference_seconds = time.perf_counter() - started
        results.append({"validation_scheme": scheme, "fold": fold, "model": "mlp", "test_partition": name, "n_train_files": len(train_files), "n_test_files": len(test_files), "file_leakage_count": len(set(train_files.file_id) & set(test_files.file_id)), "best_params": json.dumps({**config, "epochs": epochs}), "training_seconds": train_seconds, "inference_seconds": inference_seconds, **class_metrics(predicted.label.to_numpy(), predicted.predicted_label.to_numpy())})
        predictions.append(predicted.assign(validation_scheme=scheme, fold=fold, model="mlp"))
    return pd.DataFrame(results), pd.concat(predictions, ignore_index=True), pd.DataFrame(history)


def refit_and_save(windows: pd.DataFrame, features: list[str], lolo: pd.DataFrame, output: Path) -> dict:
    model_dir = output / "models"; model_dir.mkdir(parents=True, exist_ok=True)
    params = json.loads(lolo.best_params.mode().iloc[0]); epochs = int(params.pop("epochs"))
    model, scaler, _, _ = train_mlp(windows, None, features, params, SEED + 900, epochs, epochs)
    torch.save(model.encoder.state_dict(), model_dir / "q2_mlp_encoder.pth")
    torch.save(model.classifier.state_dict(), model_dir / "q2_mlp_classifier.pth")
    torch.save(model.state_dict(), model_dir / "q2_mlp_full.pth")
    joblib.dump(scaler, model_dir / "q2_mlp_scaler.pkl")
    (model_dir / "q2_feature_names.json").write_text(json.dumps({"features": features, "n_features": len(features)}, ensure_ascii=False, indent=2), encoding="utf-8")
    config = {"input_dim": len(features), "encoder": [128, 64, 32], "classifier": 4, "activation": "GELU", "normalization": "LayerNorm", "label_order": list(LABELS), **params, "purpose": "Source-domain MLP; encoder is an initialization candidate for a future Q3 rebuild."}
    (model_dir / "q2_model_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    return config


def draw_figures(output: Path, lolo: pd.DataFrame, predictions: pd.DataFrame, history: pd.DataFrame) -> None:
    chosen = predictions[predictions.validation_scheme == "lolo"]
    matrix = confusion_matrix(chosen.label, chosen.predicted_label, labels=LABELS, normalize="true")
    fig, axis = plt.subplots(figsize=(4, 3.6)); image = axis.imshow(matrix, vmin=0, vmax=1, cmap="Blues")
    axis.set(title="MLP LOLO file-level confusion", xlabel="Predicted", ylabel="True", xticks=range(4), xticklabels=LABELS, yticks=range(4), yticklabels=LABELS)
    for row in range(4):
        for col in range(4): axis.text(col, row, f"{matrix[row, col]:.2f}", ha="center", va="center")
    fig.colorbar(image, ax=axis); savefig(fig, output / "figures" / "mlp_lolo_confusion_matrix")
    fig, axis = plt.subplots(figsize=(5, 3)); axis.plot(lolo.test_partition, lolo.macro_f1, "o-", label="Macro-F1"); axis.plot(lolo.test_partition, lolo.balanced_accuracy, "s-", label="Balanced accuracy"); axis.set(ylim=(0, 1), title="MLP LOLO by held-out load"); axis.legend(); savefig(fig, output / "figures" / "mlp_lolo_by_load")
    curve = history[(history.validation_scheme == "lolo") & (history.stage == "inner_tuning")].groupby("epoch")[["train_loss", "val_macro_f1"]].mean()
    if not curve.empty:
        fig, axes = plt.subplots(1, 2, figsize=(7, 3)); axes[0].plot(curve.index, curve.train_loss); axes[0].set(title="Inner-CV train loss"); axes[1].plot(curve.index, curve.val_macro_f1); axes[1].set(title="Inner-CV file Macro-F1", ylim=(0, 1)); savefig(fig, output / "figures" / "mlp_training_curve")


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--q1-dir", type=Path, default=Path("outputs/q1")); parser.add_argument("--output-dir", type=Path, default=Path("outputs/q2")); args = parser.parse_args()
    windows, files, features = load_inputs(args.q1_dir); args.output_dir.mkdir(parents=True, exist_ok=True)
    config = {"random_seed": SEED, "formal_model": "mlp", "features": "Diagnostic26", "n_features": len(features), "architecture": [26, 128, 64, 32, 4], "outer_validation": ["LOLO", "StratifiedGroupKFold(4)"], "inner_validation": "StratifiedGroupKFold(3)", "file_level_evaluation": True, "sampling": "equal class and file total mass"}
    (args.output_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    lolo, lolo_predictions, lolo_history = evaluate_scheme("lolo", files, windows, features); group, group_predictions, group_history = evaluate_scheme("group_cv", files, windows, features)
    predictions, history = pd.concat([lolo_predictions, group_predictions]), pd.concat([lolo_history, group_history])
    metric_columns = ["macro_f1", "balanced_accuracy", "accuracy", "macro_precision", "macro_recall"]
    summary = pd.DataFrame([{ "model": "mlp", "validation_scheme": name, **{f"mean_{column}": float(frame[column].mean()) for column in metric_columns}, **{f"std_{column}": float(frame[column].std(ddof=1)) for column in metric_columns}} for name, frame in [("lolo", lolo), ("group_cv", group)]])
    save_df(args.output_dir / "lolo_results.csv", lolo); save_df(args.output_dir / "group_cv_results.csv", group); save_df(args.output_dir / "predictions_file_level.csv", predictions); save_df(args.output_dir / "training_history.csv", history); save_df(args.output_dir / "model_summary.csv", summary)
    held_out = predictions[predictions.validation_scheme == "lolo"]
    save_df(args.output_dir / "classification_report.csv", pd.DataFrame(classification_report(held_out.label, held_out.predicted_label, labels=LABELS, output_dict=True, zero_division=0)).T.reset_index().rename(columns={"index": "class_or_average"}))
    matrix = pd.DataFrame(confusion_matrix(held_out.label, held_out.predicted_label, labels=LABELS), index=LABELS, columns=LABELS).rename_axis("true_label").reset_index(); save_df(args.output_dir / "confusion_matrix.csv", matrix)
    model_config = refit_and_save(windows, features, lolo, args.output_dir); draw_figures(args.output_dir, lolo, predictions, history)
    (args.output_dir / "q2_summary.md").write_text("# 第二问：Diagnostic26 + MLP\n\n唯一正式模型为 MLP。所有外层与内层划分按原始 MAT 文件分组；窗口仅用于训练，测试概率按文件平均。LOLO 与 Group CV 仅报告源域泛化，不代表无标签目标域准确率。全源重训的 encoder 仅为未来第三问初始化候选，并非独立测试结果。\n", encoding="utf-8")
    print(json.dumps({"files": len(files), "windows": len(windows), "features": len(features), "formal_model": "mlp", "file_leakage_max": int(max(lolo.file_leakage_count.max(), group.file_leakage_count.max())), "model_config": model_config}, ensure_ascii=False))


if __name__ == "__main__":
    main()
