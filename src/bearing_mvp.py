"""Leakage-aware MVP for the 2025 Huawei Cup E problem.

This program intentionally implements a conservative baseline rather than
claiming target-domain accuracy.  It uses the 16 controlled CWRU source files
(four files per class), keeps every window from a raw file in the same split,
and reports only predictions plus uncertainty flags for target files A-P.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy import linalg, signal, stats
from scipy.io import loadmat
from sklearn.covariance import LedoitWolf
from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


LABELS = ("N", "OR", "IR", "B")
SOURCE_FS = 48_000
TARGET_FS = 32_000
TARGET_RPM = 600.0
WINDOW = 16_384
HOP = 8_192


@dataclass(frozen=True)
class Record:
    path: Path
    domain: str
    label: str | None
    rpm: float | None


@dataclass
class FileWindows:
    record: Record
    features: np.ndarray
    diagnostics: np.ndarray


FEATURE_NAMES = [
    "abs_mean_norm",
    "skew_norm",
    "kurtosis_pearson_norm",
    "crest_factor_norm",
    "impulse_factor_norm",
    "margin_factor_norm",
    "shape_factor_norm",
    "zero_crossing_rate",
    "envelope_kurtosis",
    "envelope_crest_factor",
    "order_band_0p5_2",
    "order_band_2_4",
    "order_band_4_8",
    "order_band_8_16",
    "order_band_16_32",
    "order_band_32_64",
]


def controlled_source_records(data_root: Path) -> list[Record]:
    """Return exactly the 16 files specified by the controlled MVP design."""
    records: list[Record] = []
    for path in sorted(data_root.rglob("*.mat")):
        name = path.name
        text = path.as_posix()
        label: str | None = None
        if "48kHz_Normal_data" in text and re.fullmatch(r"N_[0-3](?:_\(\d{4}rpm\))?\.mat", name):
            label = "N"
        elif "48kHz_DE_data" in text and re.fullmatch(r"B007_[0-3]\.mat", name):
            label = "B"
        elif "48kHz_DE_data" in text and re.fullmatch(r"IR007_[0-3]\.mat", name):
            label = "IR"
        elif "48kHz_DE_data" in text and re.fullmatch(r"OR007@6_[0-3]\.mat", name):
            label = "OR"
        if label is not None:
            records.append(Record(path=path, domain="source", label=label, rpm=None))

    counts = {label: sum(record.label == label for record in records) for label in LABELS}
    if counts != {label: 4 for label in LABELS}:
        raise RuntimeError(f"Controlled source selection is incomplete: {counts}")
    return records


def target_records(data_root: Path) -> list[Record]:
    records = [
        Record(path=path, domain="target", label=None, rpm=TARGET_RPM)
        for path in sorted(data_root.rglob("*.mat"))
        if path.stem in set("ABCDEFGHIJKLMNOP")
        and "目标域数据集" in path.as_posix()
    ]
    if len(records) != 16:
        raise RuntimeError(f"Expected target files A-P, found {len(records)}")
    return records


def _read_signal_and_rpm(record: Record, channel: str) -> tuple[np.ndarray, float]:
    contents = loadmat(record.path)
    if record.domain == "target":
        key = record.path.stem
        if key not in contents:
            raise RuntimeError(f"Target variable {key!r} missing in {record.path}")
        return np.asarray(contents[key], dtype=float).reshape(-1), TARGET_RPM

    suffix = f"_{channel}_time"
    candidates = [key for key in contents if key.endswith(suffix)]
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one {suffix} variable in {record.path}, got {candidates}")
    signal_values = np.asarray(contents[candidates[0]], dtype=float).reshape(-1)
    rpm_keys = [key for key in contents if key.endswith("RPM")]
    if rpm_keys:
        return signal_values, float(np.asarray(contents[rpm_keys[0]]).reshape(-1)[0])

    match = re.search(r"\((\d{4})rpm\)", record.path.name)
    if match:
        return signal_values, float(match.group(1))
    raise RuntimeError(f"RPM is absent and cannot be parsed from {record.path}")


def _robust_normalize(values: np.ndarray) -> np.ndarray:
    centre = np.median(values)
    mad = np.median(np.abs(values - centre))
    scale = max(1.4826 * mad, np.finfo(float).eps)
    return (values - centre) / scale


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / max(denominator, np.finfo(float).eps))


def _feature_vector(window: np.ndarray, fs: int, rpm: float) -> tuple[np.ndarray, np.ndarray]:
    detrended = signal.detrend(window, type="constant")
    normalized = _robust_normalize(detrended)
    abs_values = np.abs(normalized)
    rms = float(np.sqrt(np.mean(detrended**2)))
    peak = float(np.max(abs_values))
    mean_abs = float(np.mean(abs_values))
    sqrt_abs_mean = float(np.mean(np.sqrt(abs_values))) ** 2
    envelope = np.abs(signal.hilbert(normalized))
    envelope_centered = envelope - np.mean(envelope)
    envelope_peak = float(np.max(np.abs(envelope_centered)))
    envelope_rms = float(np.sqrt(np.mean(envelope_centered**2)))

    spectrum = np.abs(np.fft.rfft(envelope_centered)) ** 2
    frequencies = np.fft.rfftfreq(len(window), d=1.0 / fs)
    order = frequencies / max(rpm / 60.0, np.finfo(float).eps)
    total_power = float(np.sum(spectrum[(order >= 0.5) & (order < 64.0)]))
    band_energies = []
    for lower, upper in ((0.5, 2), (2, 4), (4, 8), (8, 16), (16, 32), (32, 64)):
        band_energies.append(_safe_ratio(float(np.sum(spectrum[(order >= lower) & (order < upper)])), total_power))

    vector = np.array(
        [
            mean_abs,
            float(stats.skew(normalized, bias=False)),
            float(stats.kurtosis(normalized, fisher=False, bias=False)),
            _safe_ratio(peak, float(np.sqrt(np.mean(normalized**2)))),
            _safe_ratio(peak, mean_abs),
            _safe_ratio(peak, sqrt_abs_mean),
            _safe_ratio(float(np.sqrt(np.mean(normalized**2))), mean_abs),
            float(np.mean(np.diff(np.signbit(normalized)) != 0)),
            float(stats.kurtosis(envelope, fisher=False, bias=False)),
            _safe_ratio(envelope_peak, envelope_rms),
            *band_energies,
        ],
        dtype=float,
    )
    diagnostic = np.array(
        [
            rms,
            float(stats.kurtosis(detrended, fisher=False, bias=False)),
            _safe_ratio(float(np.max(np.abs(detrended))), rms),
        ],
        dtype=float,
    )
    return np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0), diagnostic


def extract_windows(record: Record, channel: str) -> FileWindows:
    values, rpm = _read_signal_and_rpm(record, channel)
    if record.domain == "source":
        values = signal.resample_poly(values, up=2, down=3)
        fs = TARGET_FS
    else:
        fs = TARGET_FS
    if len(values) < WINDOW:
        raise RuntimeError(f"Signal is shorter than one window: {record.path}")

    vectors: list[np.ndarray] = []
    diagnostics: list[np.ndarray] = []
    for start in range(0, len(values) - WINDOW + 1, HOP):
        vector, diagnostic = _feature_vector(values[start : start + WINDOW], fs, rpm)
        vectors.append(vector)
        diagnostics.append(diagnostic)
    return FileWindows(record=record, features=np.vstack(vectors), diagnostics=np.vstack(diagnostics))


def _fit_classifier(features: np.ndarray, labels: np.ndarray, groups: np.ndarray, seed: int) -> Pipeline:
    # Each raw file contributes total weight one, so long recordings cannot dominate.
    group_sizes = {group: int(np.sum(groups == group)) for group in np.unique(groups)}
    weights = np.array([1.0 / group_sizes[group] for group in groups])
    classifier = Pipeline(
        [
            ("scale", StandardScaler()),
            ("svc", SVC(C=2.0, gamma="scale", class_weight="balanced", probability=True, random_state=seed)),
        ]
    )
    classifier.fit(features, labels, svc__sample_weight=weights)
    return classifier


def evaluate_grouped(source_files: Iterable[FileWindows], seed: int) -> tuple[list[dict[str, object]], dict[str, object]]:
    source_files = list(source_files)
    x = np.vstack([item.features for item in source_files])
    y = np.concatenate([np.repeat(item.record.label, len(item.features)) for item in source_files])
    groups = np.concatenate([np.repeat(item.record.path.as_posix(), len(item.features)) for item in source_files])
    rows: list[dict[str, object]] = []

    # This controlled data set has four files per class.  Constructing folds
    # explicitly guarantees that each test fold holds out one raw file from
    # each class; generic grouped splitters can violate this with unequal
    # numbers of windows per file.
    group_by_label = {
        label: sorted(item.record.path.as_posix() for item in source_files if item.record.label == label)
        for label in LABELS
    }
    rng = np.random.default_rng(seed)
    orders = {label: rng.permutation(file_names) for label, file_names in group_by_label.items()}
    for fold in range(4):
        test_groups = {str(orders[label][fold]) for label in LABELS}
        test_index = np.flatnonzero(np.isin(groups, list(test_groups)))
        train_index = np.flatnonzero(~np.isin(groups, list(test_groups)))
        model = _fit_classifier(x[train_index], y[train_index], groups[train_index], seed + fold)
        probabilities = model.predict_proba(x[test_index])
        classes = model.classes_
        for group in np.unique(groups[test_index]):
            mask = groups[test_index] == group
            expected = y[test_index][mask][0]
            mean_probability = probabilities[mask].mean(axis=0)
            predicted = str(classes[int(np.argmax(mean_probability))])
            probability_labels = classes[np.argmax(probabilities[mask], axis=1)]
            rows.append(
                {
                    "fold": fold + 1,
                    "file": group,
                    "true_label": str(expected),
                    "predicted_label": predicted,
                    "mean_confidence": float(np.max(mean_probability)),
                    "window_vote_ratio": float(np.mean(probability_labels == predicted)),
                }
            )

    truth = [str(row["true_label"]) for row in rows]
    prediction = [str(row["predicted_label"]) for row in rows]
    recalls = recall_score(truth, prediction, labels=list(LABELS), average=None, zero_division=0)
    summary = {
        "evaluation_unit": "raw_mat_file",
        "n_source_files": len(rows),
        "macro_f1": float(f1_score(truth, prediction, labels=list(LABELS), average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, prediction)),
        "recall_by_class": {label: float(value) for label, value in zip(LABELS, recalls)},
        "warning": "This is source-domain, file-grouped validation only; it is not a target-domain accuracy estimate.",
    }
    return rows, summary


def _coral_source_to_target(source_features: np.ndarray, target_features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Align source covariance to target covariance without using target labels."""
    pre_scale = StandardScaler().fit(source_features)
    source = pre_scale.transform(source_features)
    target = pre_scale.transform(target_features)
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    dimension = source.shape[1]
    regularizer = 1e-5
    source_covariance = np.cov(source - source_mean, rowvar=False) + regularizer * np.eye(dimension)
    target_covariance = np.cov(target - target_mean, rowvar=False) + regularizer * np.eye(dimension)

    def matrix_power(matrix: np.ndarray, exponent: float) -> np.ndarray:
        values, vectors = linalg.eigh(matrix)
        return (vectors * np.maximum(values, regularizer) ** exponent) @ vectors.T

    aligned_source = (source - source_mean) @ matrix_power(source_covariance, -0.5) @ matrix_power(target_covariance, 0.5) + target_mean
    return aligned_source, target


def _ood_reference(model: Pipeline, source_features: np.ndarray) -> tuple[LedoitWolf, float]:
    standard_features = model.named_steps["scale"].transform(source_features)
    detector = LedoitWolf().fit(standard_features)
    source_distance = detector.mahalanobis(standard_features)
    return detector, float(np.quantile(source_distance, 0.99))


def predict_target(
    model: Pipeline,
    target_files: Iterable[FileWindows],
    review_threshold: float,
    method: str,
    ood_detector: LedoitWolf | None = None,
    ood_threshold: float | None = None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    classes = model.classes_
    for item in target_files:
        probabilities = model.predict_proba(item.features)
        mean_probability = probabilities.mean(axis=0)
        index = int(np.argmax(mean_probability))
        predicted = str(classes[index])
        window_predictions = classes[np.argmax(probabilities, axis=1)]
        average_diagnostics = item.diagnostics.mean(axis=0)
        if ood_detector is not None and ood_threshold is not None:
            standardized = model.named_steps["scale"].transform(item.features)
            mean_ood_distance = float(np.mean(ood_detector.mahalanobis(standardized)))
            out_of_distribution = bool(mean_ood_distance > ood_threshold)
        else:
            mean_ood_distance = float("nan")
            out_of_distribution = False
        rows.append(
            {
                "method": method,
                "file": item.record.path.stem,
                "predicted_label": predicted,
                "mean_confidence": float(mean_probability[index]),
                "window_vote_ratio": float(np.mean(window_predictions == predicted)),
                "window_probability_p05": float(np.quantile(probabilities[:, index], 0.05)),
                "window_probability_p95": float(np.quantile(probabilities[:, index], 0.95)),
                "mean_rms": float(average_diagnostics[0]),
                "mean_kurtosis": float(average_diagnostics[1]),
                "mean_crest_factor": float(average_diagnostics[2]),
                "mean_ood_distance": mean_ood_distance,
                "ood_threshold": ood_threshold if ood_threshold is not None else "not_applicable",
                "out_of_distribution": out_of_distribution,
                "review_required": bool(
                    mean_probability[index] < review_threshold
                    or np.mean(window_predictions == predicted) < review_threshold
                    or out_of_distribution
                ),
                "interpretation_boundary": "Similarity to a CWRU source class, not verified target truth or a target bearing fault-frequency attribution.",
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("数据集") / "数据集")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs") / "mvp")
    parser.add_argument("--channel", choices=("DE", "FE"), default="DE")
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--review-threshold", type=float, default=0.60)
    args = parser.parse_args()

    if not 0.0 < args.review_threshold <= 1.0:
        raise ValueError("--review-threshold must be in (0, 1]")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    source_records = controlled_source_records(args.data_root)
    targets = target_records(args.data_root)
    source_files = [extract_windows(record, args.channel) for record in source_records]
    target_files = [extract_windows(record, args.channel) for record in targets]
    source_rows, source_summary = evaluate_grouped(source_files, args.seed)

    source_x = np.vstack([item.features for item in source_files])
    source_y = np.concatenate([np.repeat(item.record.label, len(item.features)) for item in source_files])
    source_groups = np.concatenate([np.repeat(item.record.path.as_posix(), len(item.features)) for item in source_files])
    final_model = _fit_classifier(source_x, source_y, source_groups, args.seed)
    source_ood_detector, source_ood_threshold = _ood_reference(final_model, source_x)
    target_x = np.vstack([item.features for item in target_files])
    source_rows_target = predict_target(
        final_model,
        target_files,
        args.review_threshold,
        method="source_only",
        ood_detector=source_ood_detector,
        ood_threshold=source_ood_threshold,
    )
    coral_source_x, coral_target_x = _coral_source_to_target(source_x, target_x)
    coral_model = _fit_classifier(coral_source_x, source_y, source_groups, args.seed)
    coral_files = []
    offset = 0
    for item in target_files:
        count = len(item.features)
        coral_files.append(FileWindows(record=item.record, features=coral_target_x[offset : offset + count], diagnostics=item.diagnostics))
        offset += count
    coral_rows_target = predict_target(coral_model, coral_files, args.review_threshold, method="CORAL")
    comparison_rows = []
    for source_row, coral_row in zip(source_rows_target, coral_rows_target):
        comparison_rows.append(
            {
                "file": source_row["file"],
                "source_only_label": source_row["predicted_label"],
                "coral_label": coral_row["predicted_label"],
                "models_agree": source_row["predicted_label"] == coral_row["predicted_label"],
                "source_only_review_required": source_row["review_required"],
                "coral_review_required": coral_row["review_required"],
                "final_review_required": bool(
                    source_row["review_required"]
                    or coral_row["review_required"]
                    or source_row["predicted_label"] != coral_row["predicted_label"]
                ),
            }
        )

    manifest_rows = [
        {"domain": item.record.domain, "file": item.record.path.as_posix(), "label": item.record.label or "", "windows": len(item.features)}
        for item in [*source_files, *target_files]
    ]
    write_csv(args.output_dir / "source_file_cv_predictions.csv", source_rows)
    write_csv(args.output_dir / "target_predictions_source_only.csv", source_rows_target)
    write_csv(args.output_dir / "target_predictions_coral.csv", coral_rows_target)
    write_csv(args.output_dir / "target_prediction_comparison.csv", comparison_rows)
    write_csv(args.output_dir / "data_manifest.csv", manifest_rows)
    summary = {
        "method": "controlled-source, order-aware handcrafted features plus RBF-SVM",
        "channel": args.channel,
        "source_selection": "16 files: 4 each of N, OR007@6, IR007, B007; source 48 kHz signals resampled to 32 kHz",
        "split_rule": "all windows from one raw .mat file stay in one split",
        "feature_names": FEATURE_NAMES,
        "source_validation": source_summary,
        "target_evaluation": "No target accuracy is reported because A-P labels are unavailable. Source-only and CORAL outputs are presented separately; disagreements or OOD flags require review.",
        "target_review_threshold": args.review_threshold,
    }
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
