"""Optional, leakage-aware feature-level DANN baseline for E problem.

The target domain is unlabelled.  This program reports source-file validation
and target candidate labels only; it never computes or prints target accuracy.
It uses the same controlled source records and manually engineered, order-aware
features as bearing_mvp.py, then learns a small feature extractor and domain
adversary.  The gradient reversal design is independently implemented from the
MIT-licensed Transfer-Learning-Library mechanism; see THIRD_PARTY_NOTICES.md.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from itertools import cycle
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score
from sklearn.preprocessing import StandardScaler
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from bearing_mvp import LABELS, FileWindows, controlled_source_records, extract_windows, target_records


class GradientReverse(torch.autograd.Function):
    """Identity forward; multiply feature gradients by -coefficient backward."""

    @staticmethod
    def forward(ctx: torch.autograd.function.FunctionCtx, values: Tensor, coefficient: float) -> Tensor:
        ctx.coefficient = coefficient
        return values.view_as(values)

    @staticmethod
    def backward(ctx: torch.autograd.function.FunctionCtx, gradient: Tensor) -> tuple[Tensor, None]:
        return gradient.neg().mul(ctx.coefficient), None


def reverse_gradient(values: Tensor, coefficient: float) -> Tensor:
    return GradientReverse.apply(values, coefficient)


class FeatureDANN(nn.Module):
    def __init__(self, input_dimension: int, class_count: int) -> None:
        super().__init__()
        self.extractor = nn.Sequential(
            nn.Linear(input_dimension, 64),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(32, class_count)
        self.domain_discriminator = nn.Sequential(
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(32, 1),
        )

    def features(self, values: Tensor) -> Tensor:
        return self.extractor(values)

    def class_logits(self, features: Tensor) -> Tensor:
        return self.classifier(features)

    def domain_logits(self, features: Tensor, coefficient: float) -> Tensor:
        return self.domain_discriminator(reverse_gradient(features, coefficient))


@dataclass
class TrainResult:
    model: FeatureDANN
    history: list[dict[str, float]]


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _stack_source(files: list[FileWindows]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features = np.vstack([item.features for item in files])
    labels = np.concatenate(
        [np.repeat(LABELS.index(str(item.record.label)), len(item.features)) for item in files]
    ).astype(np.int64)
    groups = np.concatenate([np.repeat(item.record.path.as_posix(), len(item.features)) for item in files])
    return features, labels, groups


def _stack_target(files: list[FileWindows]) -> np.ndarray:
    return np.vstack([item.features for item in files])


def _train_dann(
    source_x: np.ndarray,
    source_y: np.ndarray,
    target_x: np.ndarray,
    seed: int,
    epochs: int,
    batch_size: int,
    domain_weight: float,
) -> TrainResult:
    _set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FeatureDANN(source_x.shape[1], len(LABELS)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    class_count = np.bincount(source_y, minlength=len(LABELS))
    sample_weights = 1.0 / class_count[source_y]
    source_dataset = TensorDataset(torch.tensor(source_x, dtype=torch.float32), torch.tensor(source_y, dtype=torch.long))
    target_dataset = TensorDataset(torch.tensor(target_x, dtype=torch.float32))
    source_loader = DataLoader(
        source_dataset,
        batch_size=batch_size,
        sampler=WeightedRandomSampler(torch.tensor(sample_weights, dtype=torch.double), len(target_dataset), replacement=True),
    )
    target_loader = DataLoader(target_dataset, batch_size=batch_size, shuffle=True)
    steps_per_epoch = len(source_loader)
    total_steps = max(epochs * steps_per_epoch, 1)
    history: list[dict[str, float]] = []
    global_step = 0

    for epoch in range(1, epochs + 1):
        model.train()
        sums = {"classification_loss": 0.0, "domain_loss": 0.0, "domain_accuracy": 0.0}
        for (source_batch, labels), (target_batch,) in zip(source_loader, cycle(target_loader)):
            progress = global_step / max(total_steps - 1, 1)
            coefficient = float(2.0 / (1.0 + np.exp(-10.0 * progress)) - 1.0)
            source_batch, labels, target_batch = source_batch.to(device), labels.to(device), target_batch.to(device)
            source_features = model.features(source_batch)
            target_features = model.features(target_batch)
            classification_loss = F.cross_entropy(model.class_logits(source_features), labels)
            source_domain = model.domain_logits(source_features, coefficient)
            target_domain = model.domain_logits(target_features, coefficient)
            domain_loss = 0.5 * (
                F.binary_cross_entropy_with_logits(source_domain, torch.ones_like(source_domain))
                + F.binary_cross_entropy_with_logits(target_domain, torch.zeros_like(target_domain))
            )
            loss = classification_loss + domain_weight * domain_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                domain_predictions = torch.cat([source_domain, target_domain]).sigmoid() >= 0.5
                domain_truth = torch.cat([torch.ones_like(source_domain), torch.zeros_like(target_domain)]) >= 0.5
                sums["classification_loss"] += float(classification_loss)
                sums["domain_loss"] += float(domain_loss)
                sums["domain_accuracy"] += float((domain_predictions == domain_truth).float().mean())
            global_step += 1
        history.append({"epoch": float(epoch), **{name: value / steps_per_epoch for name, value in sums.items()}})
    return TrainResult(model=model, history=history)


def _probabilities(model: FeatureDANN, values: np.ndarray) -> np.ndarray:
    device = next(model.parameters()).device
    model.eval()
    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(values), 256):
            batch = torch.tensor(values[start : start + 256], dtype=torch.float32, device=device)
            outputs.append(torch.softmax(model.class_logits(model.features(batch)), dim=1).cpu().numpy())
    return np.vstack(outputs)


def _fold_test_groups(source_files: list[FileWindows], seed: int) -> list[set[str]]:
    rng = np.random.default_rng(seed)
    by_class = {
        label: sorted(item.record.path.as_posix() for item in source_files if item.record.label == label)
        for label in LABELS
    }
    orders = {label: rng.permutation(by_class[label]) for label in LABELS}
    return [{str(orders[label][fold]) for label in LABELS} for fold in range(4)]


def _file_predictions(files: list[FileWindows], probabilities: np.ndarray, method: str, review_threshold: float) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    offset = 0
    for item in files:
        count = len(item.features)
        file_probability = probabilities[offset : offset + count]
        offset += count
        mean_probability = file_probability.mean(axis=0)
        predicted_index = int(np.argmax(mean_probability))
        window_labels = np.argmax(file_probability, axis=1)
        rows.append(
            {
                "method": method,
                "file": item.record.path.stem,
                "predicted_label": LABELS[predicted_index],
                "mean_confidence": float(mean_probability[predicted_index]),
                "window_vote_ratio": float(np.mean(window_labels == predicted_index)),
                "window_probability_p05": float(np.quantile(file_probability[:, predicted_index], 0.05)),
                "window_probability_p95": float(np.quantile(file_probability[:, predicted_index], 0.95)),
                "review_required": bool(
                    mean_probability[predicted_index] < review_threshold
                    or np.mean(window_labels == predicted_index) < review_threshold
                ),
                "interpretation_boundary": "Unsupervised DANN candidate only; not target truth or target-domain accuracy.",
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if rows:
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def _three_method_comparison(dann_rows: list[dict[str, object]], mvp_dir: Path) -> list[dict[str, object]]:
    """Merge optional DANN candidates with existing source-only/CORAL evidence.

    A DANN confidence must never erase an OOD or review flag from another
    method.  If MVP outputs are absent, no merged table is produced.
    """
    source_path = mvp_dir / "target_predictions_source_only.csv"
    coral_path = mvp_dir / "target_predictions_coral.csv"
    if not source_path.exists() or not coral_path.exists():
        return []
    with source_path.open(encoding="utf-8-sig", newline="") as handle:
        source_rows = {row["file"]: row for row in csv.DictReader(handle)}
    with coral_path.open(encoding="utf-8-sig", newline="") as handle:
        coral_rows = {row["file"]: row for row in csv.DictReader(handle)}
    rows = []
    for dann in dann_rows:
        source = source_rows[dann["file"]]
        coral = coral_rows[dann["file"]]
        labels = [source["predicted_label"], coral["predicted_label"], str(dann["predicted_label"])]
        source_review = source["review_required"] == "True"
        coral_review = coral["review_required"] == "True"
        dann_review = bool(dann["review_required"])
        rows.append(
            {
                "file": dann["file"],
                "source_only_label": labels[0],
                "coral_label": labels[1],
                "dann_label": labels[2],
                "all_methods_agree": len(set(labels)) == 1,
                "source_only_review_required": source_review,
                "coral_review_required": coral_review,
                "dann_review_required": dann_review,
                "final_review_required": source_review or coral_review or dann_review or len(set(labels)) != 1,
                "interpretation_boundary": "A candidate agreement is not target ground truth; any review flag must be retained.",
            }
        )
    return rows


def _source_cv(
    source_files: list[FileWindows],
    target_x: np.ndarray,
    seed: int,
    epochs: int,
    batch_size: int,
    domain_weight: float,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    x, y, groups = _stack_source(source_files)
    rows: list[dict[str, object]] = []
    for fold, held_out_groups in enumerate(_fold_test_groups(source_files, seed), start=1):
        test_mask = np.isin(groups, list(held_out_groups))
        scaler = StandardScaler().fit(x[~test_mask])
        training = _train_dann(
            scaler.transform(x[~test_mask]), y[~test_mask], scaler.transform(target_x), seed + fold, epochs, batch_size, domain_weight
        )
        probabilities = _probabilities(training.model, scaler.transform(x[test_mask]))
        for group in sorted(held_out_groups):
            mask = groups[test_mask] == group
            mean_probability = probabilities[mask].mean(axis=0)
            predicted_index = int(np.argmax(mean_probability))
            rows.append(
                {
                    "fold": fold,
                    "file": group,
                    "true_label": LABELS[int(y[test_mask][mask][0])],
                    "predicted_label": LABELS[predicted_index],
                    "mean_confidence": float(mean_probability[predicted_index]),
                    "window_vote_ratio": float(np.mean(np.argmax(probabilities[mask], axis=1) == predicted_index)),
                }
            )
    truth = [str(row["true_label"]) for row in rows]
    prediction = [str(row["predicted_label"]) for row in rows]
    recalls = recall_score(truth, prediction, labels=list(LABELS), average=None, zero_division=0)
    return rows, {
        "evaluation_unit": "raw_mat_file",
        "macro_f1": float(f1_score(truth, prediction, labels=list(LABELS), average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, prediction)),
        "recall_by_class": {label: float(score) for label, score in zip(LABELS, recalls)},
        "warning": "Target windows participated only without labels during each fold's DANN training. This remains source-file validation, not target accuracy.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("数据集") / "数据集")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs") / "dann")
    parser.add_argument("--mvp-dir", type=Path, default=Path("outputs") / "mvp")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--domain-weight", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--review-threshold", type=float, default=0.60)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 2 or not 0.0 <= args.domain_weight <= 1.0:
        raise ValueError("epochs >= 1, batch-size >= 2, and domain-weight in [0, 1] are required")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    source_files = [extract_windows(record, "DE") for record in controlled_source_records(args.data_root)]
    target_files = [extract_windows(record, "DE") for record in target_records(args.data_root)]
    source_x, source_y, _ = _stack_source(source_files)
    target_x = _stack_target(target_files)

    source_cv_rows, source_cv_summary = _source_cv(
        source_files, target_x, args.seed, args.epochs, args.batch_size, args.domain_weight
    )
    final_scaler = StandardScaler().fit(source_x)
    final_training = _train_dann(
        final_scaler.transform(source_x), source_y, final_scaler.transform(target_x), args.seed, args.epochs, args.batch_size, args.domain_weight
    )
    target_rows = _file_predictions(
        target_files, _probabilities(final_training.model, final_scaler.transform(target_x)), "DANN", args.review_threshold
    )
    _write_csv(args.output_dir / "source_file_cv_predictions.csv", source_cv_rows)
    _write_csv(args.output_dir / "target_candidates.csv", target_rows)
    _write_csv(args.output_dir / "three_method_comparison.csv", _three_method_comparison(target_rows, args.mvp_dir))
    _write_csv(args.output_dir / "training_history.csv", final_training.history)
    summary = {
        "method": "feature-level DANN with a warm-start gradient reversal schedule",
        "source_selection": "same controlled 16-file DE source subset as bearing_mvp.py",
        "split_rule": "all windows from a raw source .mat file stay in one source-validation fold",
        "target_evaluation": "No target accuracy is computed or claimed.",
        "source_validation": source_cv_summary,
        "epochs": args.epochs,
        "domain_weight": args.domain_weight,
    }
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
