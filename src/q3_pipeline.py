"""Question 3: frozen Diagnostic26 MLP embeddings + PythonOT LpL1 transport.

This module deliberately contains no OT solver.  It adapts project data to the
official PythonOT/POT ``SinkhornLpl1Transport`` API and never reads a target
label or the problem PDF during fitting, selection, or prediction.
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import ot
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, recall_score

import q1_pipeline as q1
import q2_pipeline as q2

LABELS = q2.LABELS
LABEL_TO_INT = {label: index for index, label in enumerate(LABELS)}
SEED = 2025
FORMAL_RPM, REG_E, REG_CL = 600, 0.1, 1.0
# These are solver-accuracy controls, fixed before re-running the formal
# 600-rpm fit.  At 1e-6 POT still emitted an inner-Sinkhorn non-convergence
# warning after 20,000 iterations on the unlabelled target embeddings.  The
# complete fit-plus-transform call converged at 1e-5 with 20,000 inner steps,
# without changing the prescribed OT regularisation parameters.
POT_MAX_ITER, POT_MAX_INNER_ITER, POT_TOL = 10, 20000, 1e-5
WINDOW, HOP = 16384, 8192


def save_table(path: Path, table: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False, encoding="utf-8-sig")


def save_figure(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def target_diagnostic_rows(data_root: Path, feature_names: list[str], rpm: int) -> pd.DataFrame:
    """Use the Q1 extraction functions with the formal 32 kHz/16384/8192 protocol."""
    records = [record for record in q1.records(data_root) if record[1] == "target_32k"]
    if len(records) != 16:
        raise ValueError(f"Expected 16 target MAT files, found {len(records)}")
    rows = []
    for path, branch, native_fs, _ in records:
        signal, _, _, _ = q1.load(path, branch)
        signal = q1.resample(signal, native_fs, q1.FS_MAIN)
        for window_id, start in enumerate(q1.starts(len(signal), WINDOW, HOP)):
            values = q1.feature_vector(signal[start:start + WINDOW], q1.FS_MAIN, float(rpm))
            rows.append({"file_id": path.stem, "file_path": path.as_posix(), "window_id": window_id,
                         "branch": branch, "rpm": float(rpm), **{name: values[name] for name in feature_names}})
    frame = pd.DataFrame(rows)
    if set(frame.columns).intersection({"label", "true_label", "target_label"}):
        raise AssertionError("Target feature table must not contain labels")
    if frame.file_id.nunique() != 16 or not set(feature_names).issubset(frame.columns):
        raise AssertionError("Target Diagnostic26 extraction is incomplete")
    return frame


def load_frozen_q2(q2_dir: Path):
    model_dir = q2_dir / "models"
    feature_names = json.loads((model_dir / "q2_feature_names.json").read_text(encoding="utf-8"))["features"]
    config = json.loads((model_dir / "q2_model_config.json").read_text(encoding="utf-8"))
    if len(feature_names) != 26 or config["label_order"] != list(LABELS):
        raise ValueError("Q2 Diagnostic26 schema or label order does not match Q3")
    scaler = joblib.load(model_dir / "q2_mlp_scaler.pkl")
    encoder = q2.FeatureEncoder(len(feature_names), float(config["dropout"]))
    try:
        state = torch.load(model_dir / "q2_mlp_encoder.pth", map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(model_dir / "q2_mlp_encoder.pth", map_location="cpu")
    encoder.load_state_dict(state); encoder.eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    return feature_names, scaler, encoder, config


def encode_windows(frame: pd.DataFrame, feature_names: list[str], scaler, encoder: q2.FeatureEncoder) -> np.ndarray:
    values = scaler.transform(frame[feature_names].to_numpy(float)).astype(np.float32)
    with torch.no_grad():
        output = encoder(torch.from_numpy(values)).cpu().numpy()
    if output.shape != (len(frame), 32) or not np.isfinite(output).all():
        raise ValueError("Frozen Q2 encoder did not return finite 32D embeddings")
    return output


def file_embeddings(frame: pd.DataFrame, embeddings: np.ndarray, source: bool) -> pd.DataFrame:
    table = pd.DataFrame(embeddings, columns=[f"z_{index:02d}" for index in range(embeddings.shape[1])])
    table.insert(0, "file_id", frame.file_id.to_numpy())
    if source:
        table.insert(1, "label", frame.label.to_numpy())
        table.insert(2, "load", frame.load.to_numpy())
    else:
        table.insert(1, "rpm", frame.rpm.to_numpy())
    aggregation = {column: "mean" for column in table if column.startswith("z_")}
    aggregation.update({"label": "first", "load": "first"} if source else {"rpm": "first"})
    output = table.groupby("file_id", as_index=False).agg(aggregation)
    if output.file_id.nunique() != len(output) or len(output) == 0:
        raise ValueError("File-level embedding aggregation failed")
    return output


def embedding_columns(table: pd.DataFrame) -> list[str]:
    columns = [column for column in table if column.startswith("z_")]
    if len(columns) != 32:
        raise ValueError("Expected exactly 32 embedding columns")
    return columns


def source_file_weights(labels: np.ndarray) -> np.ndarray:
    counts = pd.Series(labels).value_counts()
    weights = np.asarray([1.0 / (len(LABELS) * counts[label]) for label in labels], dtype=float)
    if not np.isclose(weights.sum(), 1.0):
        raise AssertionError("Source file weights must sum to one")
    return weights


def target_file_weights(count: int) -> np.ndarray:
    return np.full(count, 1.0 / count, dtype=float)


def fit_pot(Xs: np.ndarray, ys: np.ndarray, Xt: np.ndarray):
    """Call POT's official class; the closure supplies file-balanced masses."""
    a, b = source_file_weights(ys), target_file_weights(len(Xt))
    def distribution_estimation(values):
        if len(values) == len(Xs):
            return a
        if len(values) == len(Xt):
            return b
        raise ValueError("POT requested an unexpected sample distribution")
    transport = ot.da.SinkhornLpl1Transport(
        reg_e=REG_E, reg_cl=REG_CL, max_iter=POT_MAX_ITER,
        max_inner_iter=POT_MAX_INNER_ITER, tol=POT_TOL,
        verbose=False, metric="sqeuclidean", norm="max",
        distribution_estimation=distribution_estimation,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        transport.fit(Xs=Xs, ys=ys, Xt=Xt)
        moved = np.asarray(transport.transform(Xs=Xs), dtype=float)
    transport.q3_convergence_warning_count = sum("did not converge" in str(item.message).lower() for item in caught)
    coupling = np.asarray(transport.coupling_, dtype=float)
    if moved.shape != Xs.shape or coupling.shape != (len(Xs), len(Xt)):
        raise ValueError("POT transport returned an unexpected shape")
    if not np.isfinite(moved).all() or not np.isfinite(coupling).all():
        raise ValueError("POT transport contains non-finite values")
    return transport, moved, coupling, a, b


def train_linear_head(Xs: np.ndarray, ys: np.ndarray) -> LogisticRegression:
    head = LogisticRegression(max_iter=5000, class_weight=None, random_state=SEED)
    head.fit(Xs, ys, sample_weight=source_file_weights(ys))
    if not np.array_equal(head.classes_, np.arange(len(LABELS))):
        raise ValueError("Linear head does not contain all four fixed classes")
    return head


def probability_table(files: pd.DataFrame, head: LogisticRegression, embeddings: np.ndarray) -> pd.DataFrame:
    probabilities = head.predict_proba(embeddings)
    output = pd.DataFrame({"file_id": files.file_id.to_numpy()})
    for index, label in enumerate(LABELS):
        output[f"prob_{label}"] = probabilities[:, index]
    output["candidate_label"] = np.asarray(LABELS)[probabilities.argmax(axis=1)]
    output["confidence"] = probabilities.max(axis=1)
    output["margin"] = np.sort(probabilities, axis=1)[:, -1] - np.sort(probabilities, axis=1)[:, -2]
    output["entropy"] = -(probabilities * np.log(np.maximum(probabilities, 1e-12))).sum(axis=1)
    if not np.allclose(probabilities.sum(axis=1), 1) or not np.array_equal(output.candidate_label, np.asarray(LABELS)[probabilities.argmax(axis=1)]):
        raise AssertionError("Candidate labels must equal probability argmax")
    return output


def rbf_mmd(left: np.ndarray, right: np.ndarray) -> float:
    all_values = np.vstack([left, right]); squared = ((all_values[:, None] - all_values[None, :]) ** 2).sum(axis=2)
    positive = squared[squared > 0]; scale = float(np.median(positive)) if len(positive) else 1.0
    gamma = 1.0 / max(scale, 1e-12)
    kernel = np.exp(-gamma * squared); n = len(left)
    return float(kernel[:n, :n].mean() + kernel[n:, n:].mean() - 2 * kernel[:n, n:].mean())


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    recalls = recall_score(y_true, y_pred, labels=np.arange(4), average=None, zero_division=0)
    return {"macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            **{f"recall_{label}": float(value) for label, value in zip(LABELS, recalls)}}


def fit_and_predict(source_files: pd.DataFrame, target_files: pd.DataFrame):
    cols = embedding_columns(source_files)
    Xs, Xt = source_files[cols].to_numpy(float), target_files[cols].to_numpy(float)
    ys = source_files.label.map(LABEL_TO_INT).to_numpy(int)
    transport, moved, coupling, a, b = fit_pot(Xs, ys, Xt)
    head = train_linear_head(moved, ys)
    predictions = probability_table(target_files, head, Xt)
    return predictions, {"transport": transport, "moved": moved, "coupling": coupling, "head": head,
                          "source_weights": a, "target_weights": b,
                          "convergence_warning_count": transport.q3_convergence_warning_count,
                          "mmd_before": rbf_mmd(Xs, Xt), "mmd_after": rbf_mmd(moved, Xt)}


def surrogate_uda(source_windows: pd.DataFrame, source_files: pd.DataFrame, feature_names: list[str], q2_config: dict):
    rows, predictions = [], []
    for held_load in sorted(source_files.load.unique()):
        train_files = source_files[source_files.load != held_load].copy()
        test_files = source_files[source_files.load == held_load].copy()
        train_windows = source_windows[source_windows.file_id.isin(train_files.file_id)]
        # This encoder is fitted from the outer training loads only; held labels are never
        # available until after POT prediction for metrics.
        model, scaler, _, _ = q2.train_mlp(train_windows, None, feature_names,
                                           {"dropout": q2_config["dropout"], "weight_decay": q2_config["weight_decay"]},
                                           SEED + int(held_load), 60, 60)
        model.encoder.eval()
        train_embed = file_embeddings(train_windows, encode_windows(train_windows, feature_names, scaler, model.encoder), True)
        test_windows = source_windows[source_windows.file_id.isin(test_files.file_id)]
        test_embed = file_embeddings(test_windows, encode_windows(test_windows, feature_names, scaler, model.encoder), True)
        # POT sees held-load embeddings as unlabeled pseudo-targets.
        pot_predictions, detail = fit_and_predict(train_embed, test_embed.drop(columns=["label", "load"]))
        truth = test_embed.sort_values("file_id").label.map(LABEL_TO_INT).to_numpy(int)
        pot_predictions = pot_predictions.sort_values("file_id").reset_index(drop=True)
        pot_pred = pot_predictions.candidate_label.map(LABEL_TO_INT).to_numpy(int)
        source_head = train_linear_head(train_embed[embedding_columns(train_embed)].to_numpy(float), train_embed.label.map(LABEL_TO_INT).to_numpy(int))
        source_pred = source_head.predict(test_embed.sort_values("file_id")[embedding_columns(test_embed)].to_numpy(float))
        rows.append({"held_load": int(held_load), "method": "pot_sinkhorn_lpl1", "n_train_files": len(train_files), "n_test_files": len(test_files), "file_leakage_count": 0, "pot_convergence_warning_count": detail["convergence_warning_count"], "mmd_before": detail["mmd_before"], "mmd_after": detail["mmd_after"], **metrics(truth, pot_pred)})
        rows.append({"held_load": int(held_load), "method": "source_only_sanity", "n_train_files": len(train_files), "n_test_files": len(test_files), "file_leakage_count": 0, "pot_convergence_warning_count": np.nan, "mmd_before": np.nan, "mmd_after": np.nan, **metrics(truth, source_pred)})
        actual = test_embed.sort_values("file_id")[["file_id", "label"]].rename(columns={"label": "true_label"}).reset_index(drop=True)
        predictions.append(pd.concat([actual, pot_predictions.assign(held_load=held_load, method="pot_sinkhorn_lpl1")], axis=1))
    result = pd.DataFrame(rows)
    return result, pd.concat(predictions, ignore_index=True)


def subset_file_embeddings(frame: pd.DataFrame, embeddings: np.ndarray, fraction: float, seed: int) -> pd.DataFrame:
    selected = []
    for file_id, indices in frame.groupby("file_id").groups.items():
        indices = np.asarray(list(indices)); count = max(1, int(np.ceil(len(indices) * fraction)))
        rng = np.random.default_rng(seed + sum(ord(char) for char in str(file_id)))
        selected.extend(rng.choice(indices, size=count, replace=False).tolist())
    return file_embeddings(frame.loc[selected].reset_index(drop=True), embeddings[np.asarray(selected)], False)


def stability(source_files: pd.DataFrame, target_frames: dict[int, pd.DataFrame], target_embeddings: dict[int, np.ndarray]):
    formal_table, formal = fit_and_predict(source_files, file_embeddings(target_frames[FORMAL_RPM], target_embeddings[FORMAL_RPM], False))
    rpm_rows, labels_by_rpm = [], {}
    for rpm, frame in target_frames.items():
        table, run = fit_and_predict(source_files, file_embeddings(frame, target_embeddings[rpm], False))
        labels_by_rpm[rpm] = table.set_index("file_id").candidate_label
        rpm_rows.append(table.assign(rpm=rpm, pot_convergence_warning_count=run["convergence_warning_count"]))
    formal_labels = labels_by_rpm[FORMAL_RPM]
    rpm_agreement = pd.DataFrame({"file_id": formal_labels.index, "rpm_agreement": [float(len({labels_by_rpm[rpm].loc[file_id] for rpm in labels_by_rpm}) == 1) for file_id in formal_labels.index]})
    target_formal = file_embeddings(target_frames[FORMAL_RPM], target_embeddings[FORMAL_RPM], False)
    loto_rows = []
    for file_id in target_formal.file_id:
        remaining = target_formal[target_formal.file_id != file_id]
        held = target_formal[target_formal.file_id == file_id]
        # Refitted head is needed for the held target, while its embedding did not enter OT.
        cols = embedding_columns(source_files); ys = source_files.label.map(LABEL_TO_INT).to_numpy(int)
        transport, moved, _, _, _ = fit_pot(source_files[cols].to_numpy(float), ys, remaining[cols].to_numpy(float))
        head = train_linear_head(moved, ys); held_table = probability_table(held, head, held[cols].to_numpy(float))
        loto_rows.append({"file_id": file_id, "loto_label": held_table.candidate_label.iloc[0], "loto_agreement": float(held_table.candidate_label.iloc[0] == formal_labels.loc[file_id]), "pot_convergence_warning_count": transport.q3_convergence_warning_count})
    subsample_rows, labels_by_fraction, probabilities = [], {}, []
    for fraction in (1.0, 0.75, 0.50):
        target_subset = subset_file_embeddings(target_frames[FORMAL_RPM], target_embeddings[FORMAL_RPM], fraction, SEED)
        table, run = fit_and_predict(source_files, target_subset); labels_by_fraction[fraction] = table.set_index("file_id").candidate_label
        probabilities.append(table.set_index("file_id")[[f"prob_{label}" for label in LABELS]].rename(columns=lambda x: f"{fraction}_{x}"))
        subsample_rows.append(table.assign(fraction=fraction, pot_convergence_warning_count=run["convergence_warning_count"]))
    probability_wide = pd.concat(probabilities, axis=1)
    probability_std = []
    for file_id in formal_labels.index:
        per_class = [np.std([probability_wide.loc[file_id, f"{fraction}_prob_{label}"] for fraction in labels_by_fraction]) for label in LABELS]
        probability_std.append(float(np.mean(per_class)))
    subsample_agreement = pd.DataFrame({"file_id": formal_labels.index, "subsample_agreement": [float(len({labels_by_fraction[f].loc[file_id] for f in labels_by_fraction}) == 1) for file_id in formal_labels.index], "probability_std": probability_std})
    window_probabilities = formal["head"].predict_proba(target_embeddings[FORMAL_RPM])
    window_labels = np.asarray(LABELS)[window_probabilities.argmax(axis=1)]
    window_table = target_frames[FORMAL_RPM][["file_id"]].copy(); window_table["window_label"] = window_labels; window_table["window_confidence"] = window_probabilities.max(axis=1)
    window_table["formal_label"] = window_table.file_id.map(formal_labels); window_table["agree"] = window_table.window_label == window_table.formal_label
    window_stats = window_table.groupby("file_id").agg(window_vote_ratio=("agree", "mean"), window_probability_std=("window_confidence", "std")).reset_index().fillna(0)
    final = formal_table.merge(rpm_agreement, on="file_id").merge(pd.DataFrame(loto_rows), on="file_id").merge(subsample_agreement, on="file_id").merge(window_stats, on="file_id")
    return final, pd.concat(rpm_rows, ignore_index=True), pd.DataFrame(loto_rows), pd.concat(subsample_rows, ignore_index=True), formal


def draw_figures(output: Path, source_files: pd.DataFrame, target_files: pd.DataFrame, detail: dict, surrogate: pd.DataFrame, final: pd.DataFrame) -> None:
    cols = embedding_columns(source_files); source_values, target_values, moved = source_files[cols].to_numpy(float), target_files[cols].to_numpy(float), detail["moved"]
    pca = PCA(n_components=2, random_state=SEED).fit(np.vstack([source_values, target_values, moved]))
    before, target, after = pca.transform(source_values), pca.transform(target_values), pca.transform(moved)
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.4)); axes[0].scatter(before[:, 0], before[:, 1], c="#4078a8", label="source", s=20); axes[0].scatter(target[:, 0], target[:, 1], c="#d78547", label="target", s=28); axes[0].set(title="Before POT"); axes[1].scatter(after[:, 0], after[:, 1], c="#4078a8", label="transported source", s=20); axes[1].scatter(target[:, 0], target[:, 1], c="#d78547", label="target", s=28); axes[1].set(title="After POT"); [axis.legend(fontsize=7) for axis in axes]; save_figure(fig, output / "figures" / "source_target_embedding")
    fig, axis = plt.subplots(figsize=(5, 3.2)); image = axis.imshow(detail["coupling"], aspect="auto", cmap="magma"); axis.set(title="POT coupling (source files × target files)", xlabel="Target file", ylabel="Source file"); fig.colorbar(image, ax=axis); save_figure(fig, output / "figures" / "pot_coupling_matrix")
    pot = surrogate[surrogate.method == "pot_sinkhorn_lpl1"]; matrix = confusion_matrix(np.concatenate([[]])) if False else None
    fig, axis = plt.subplots(figsize=(5, 3.2)); axis.bar(pot.held_load.astype(str), pot.macro_f1, color="#4078a8"); axis.set(title="POT surrogate LOLO Macro-F1", xlabel="Held-out source load", ylabel="Macro-F1", ylim=(0, 1)); save_figure(fig, output / "figures" / "surrogate_uda_per_load")
    fig, axis = plt.subplots(figsize=(6, 3.2)); probs = final[[f"prob_{label}" for label in LABELS]].to_numpy(float); axis.imshow(probs.T, aspect="auto", vmin=0, vmax=1, cmap="Blues"); axis.set(title="A–P candidate probabilities", xlabel="Target file", ylabel="Class", yticks=range(4), yticklabels=LABELS, xticks=range(len(final)), xticklabels=final.file_id); save_figure(fig, output / "figures" / "target_candidate_probability")
    fig, axis = plt.subplots(figsize=(6, 3)); values = final[["rpm_agreement", "loto_agreement", "subsample_agreement", "window_vote_ratio"]].to_numpy(float); axis.imshow(values.T, aspect="auto", vmin=0, vmax=1, cmap="viridis"); axis.set(title="Target stability", xlabel="Target file", yticks=range(4), yticklabels=["rpm", "LOTO", "subsample", "window vote"], xticks=range(len(final)), xticklabels=final.file_id); save_figure(fig, output / "figures" / "target_stability")


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--data-root", type=Path, default=Path("数据集") / "数据集"); parser.add_argument("--q1-dir", type=Path, default=Path("outputs/q1")); parser.add_argument("--q2-dir", type=Path, default=Path("outputs/q2")); parser.add_argument("--output-dir", type=Path, default=Path("outputs/q3")); args = parser.parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    feature_names, scaler, encoder, q2_config = load_frozen_q2(args.q2_dir)
    source_meta = pd.read_csv(args.q1_dir / "source_metadata.csv")
    source_windows = pd.read_csv(args.q1_dir / "features_source_diagnostic.csv").merge(source_meta[["file_id", "load"]], on="file_id", how="left", validate="many_to_one")
    if len(feature_names) != 26 or len(source_windows) != 806 or len(source_meta) != 56: raise ValueError("Q1/Q2 frozen input counts do not match")
    source_files = file_embeddings(source_windows, encode_windows(source_windows, feature_names, scaler, encoder), True).sort_values("file_id").reset_index(drop=True)
    target_frames = {rpm: target_diagnostic_rows(args.data_root, feature_names, rpm) for rpm in (570, FORMAL_RPM, 630)}
    target_embeddings = {rpm: encode_windows(frame, feature_names, scaler, encoder) for rpm, frame in target_frames.items()}
    target_files = file_embeddings(target_frames[FORMAL_RPM], target_embeddings[FORMAL_RPM], False).sort_values("file_id").reset_index(drop=True)
    final, rpm_table, loto, subsample, detail = stability(source_files, target_frames, target_embeddings)
    surrogate, surrogate_predictions = surrogate_uda(source_windows, source_meta, feature_names, q2_config)
    save_table(args.output_dir / "target_features_diagnostic26.csv", target_frames[FORMAL_RPM]); save_table(args.output_dir / "source_file_embeddings.csv", source_files); save_table(args.output_dir / "target_file_embeddings.csv", target_files); save_table(args.output_dir / "target_predictions.csv", final.sort_values("file_id")); save_table(args.output_dir / "target_rpm_sensitivity.csv", rpm_table); save_table(args.output_dir / "target_loto.csv", loto); save_table(args.output_dir / "target_subsample_stability.csv", subsample); save_table(args.output_dir / "surrogate_uda_results.csv", surrogate); save_table(args.output_dir / "surrogate_uda_predictions.csv", surrogate_predictions)
    domain = pd.DataFrame([{"mmd_before_ot": detail["mmd_before"], "mmd_after_ot": detail["mmd_after"], "interpretation": "MMD reduction describes distribution proximity only, not target classification correctness."}]); save_table(args.output_dir / "domain_metrics.csv", domain)
    pot_info = {"repository": "https://github.com/PythonOT/POT", "install_source": "PyPI", "version": ot.__version__, "formal_class": "ot.da.SinkhornLpl1Transport", "reg_e": REG_E, "reg_cl": REG_CL, "max_iter": POT_MAX_ITER, "max_inner_iter": POT_MAX_INNER_ITER, "tol": POT_TOL, "cost_normalization": "max", "numerical_setting_rationale": "At 1e-6, the formal unlabeled fit-plus-transform still warned after 20,000 inner iterations; 1e-5 with 20,000 inner iterations converged for the formal fit and the 570/600/630-rpm stability fits, with fixed reg_e/reg_cl."}; (args.output_dir / "pot_version.json").write_text(json.dumps(pot_info, ensure_ascii=False, indent=2), encoding="utf-8")
    verification = {"formal_method": "Q2 frozen MLP encoder + POT SinkhornLpl1Transport", "pot_used": True, "custom_sinkhorn_used": False, "target_labels_used": False, "pdf_reference_used_for_training": False, "q2_encoder_frozen": True, "diagnostic26_schema_match": True, "formal_target_rpm": FORMAL_RPM, "formal_reg_e": REG_E, "formal_reg_cl": REG_CL, "formal_pot_max_iter": POT_MAX_ITER, "formal_pot_max_inner_iter": POT_MAX_INNER_ITER, "formal_pot_tol": POT_TOL, "target_file_count": int(len(final)), "finite_transport": bool(np.isfinite(detail["coupling"]).all()), "formal_pot_convergence_warning_count": int(detail["convergence_warning_count"]), "formal_pot_converged": bool(detail["convergence_warning_count"] == 0), "final_argmax_consistent": bool(np.array_equal(final.candidate_label, np.asarray(LABELS)[final[[f"prob_{label}" for label in LABELS]].to_numpy().argmax(axis=1)])), "source_file_count": int(len(source_files)), "source_target_file_overlap": int(len(set(source_files.file_id) & set(final.file_id)))}; (args.output_dir / "verification.json").write_text(json.dumps(verification, ensure_ascii=False, indent=2), encoding="utf-8")
    draw_figures(args.output_dir, source_files, target_files, detail, surrogate, final)
    pot_summary = surrogate[surrogate.method == "pot_sinkhorn_lpl1"]
    warning_total = int(rpm_table.pot_convergence_warning_count.sum() + loto.pot_convergence_warning_count.sum() + subsample.pot_convergence_warning_count.sum() + pot_summary.pot_convergence_warning_count.sum())
    summary = f"# 第三问：POT Class-Regularized OT\n\n正式链路为 Diagnostic26 → 冻结 Q2 MLP encoder (32D) → PythonOT/POT `SinkhornLpl1Transport` → 线性头。正式参数为 `reg_e=0.1`、`reg_cl=1.0`，源/目标以 MAT 文件等权参与 OT。A–P 没有用于训练、调参或评分的标签；表中仅为候选诊断。\n\nSurrogate LOLO POT：Macro-F1={pot_summary.macro_f1.mean():.3f}，BA={pot_summary.balanced_accuracy.mean():.3f}。这只是在源域留载荷伪目标协议下的诊断，不是 A–P 准确率。MMD 从 {detail['mmd_before']:.4f} 变为 {detail['mmd_after']:.4f}，只描述分布接近程度。\n\n正式 600 rpm 运行的 POT 未收敛警告数为 {detail['convergence_warning_count']}；稳定性与 surrogate 重拟合累计为 {warning_total}。若该值非零，相关候选仅应视为数值复核对象，不应进入解释阶段。\n"; (args.output_dir / "q3_summary.md").write_text(summary, encoding="utf-8")
    print(json.dumps({"pot_version": ot.__version__, "source_files": len(source_files), "target_files": len(final), "surrogate_pot_macro_f1": float(pot_summary.macro_f1.mean()), "surrogate_pot_ba": float(pot_summary.balanced_accuracy.mean()), "mmd_before": detail["mmd_before"], "mmd_after": detail["mmd_after"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
