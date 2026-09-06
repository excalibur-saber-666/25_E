"""Question 3 only: physically matched Transfer-v2 features.

This module deliberately does not alter Question 1 outputs.  Each window spans
a fixed number of nominal shaft revolutions and is resampled to a common
angle-like grid.  With only one RPM value per recording this is *approximate
constant-speed angle-domain resampling*, not tachometer-based order tracking.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal, stats

import q1_pipeline as q1

FS = 32000
REVOLUTIONS = 8
SAMPLES_PER_REVOLUTION = 256
ANGLE_SAMPLES = REVOLUTIONS * SAMPLES_PER_REVOLUTION
BANDS = ((500, 2000), (2000, 4000), (4000, 8000))
LABELS = ("N", "B", "IR", "OR")


def _entropy(power: np.ndarray) -> float:
    power = np.maximum(np.asarray(power, dtype=float), 0.0)
    power = power / max(float(power.sum()), np.finfo(float).eps)
    valid = power[power > 0]
    return float(-(valid * np.log(valid)).sum() / np.log(max(2, len(power))))


def _order_power(angle_signal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return an order spectrum, whose bin width is fixed across all RPMs."""
    nperseg = min(1024, len(angle_signal))
    order, power = signal.welch(signal.detrend(angle_signal), fs=SAMPLES_PER_REVOLUTION,
                                nperseg=nperseg, scaling="spectrum")
    return order, power


def _relative_energy(order: np.ndarray, power: np.ndarray, lo: float, hi: float) -> float:
    return float(power[(order >= lo) & (order < hi)].sum() / max(float(power.sum()), np.finfo(float).eps))


def _band_envelope(x: np.ndarray, lo: int, hi: int) -> np.ndarray:
    # All requested bands lie below the common 16 kHz Nyquist frequency.
    sos = signal.butter(4, (lo, hi), btype="bandpass", fs=FS, output="sos")
    return np.abs(signal.hilbert(signal.sosfiltfilt(sos, x)))


def feature_names() -> list[str]:
    names = ["shape_skew", "shape_kurtosis", "shape_crest", "zcr",
             "angle_psd_peak_ratio", "angle_order_entropy",
             "angle_order_0_5_2", "angle_order_2_4", "angle_order_4_8", "angle_order_8_16",
             "envelope_kurtosis", "envelope_order_entropy"]
    for lo, hi in BANDS:
        prefix = f"env_{lo}_{hi}"
        names += [f"{prefix}_kurtosis", f"{prefix}_order_entropy",
                  f"{prefix}_order_0_5_2", f"{prefix}_order_2_4",
                  f"{prefix}_order_4_8", f"{prefix}_order_8_16"]
    return names


def transfer_v2_feature(x: np.ndarray) -> dict[str, float]:
    """Features on a fixed-eight-revolution segment resampled to 2048 points."""
    x = signal.detrend(np.asarray(x, dtype=float))
    mad = max(1.4826 * np.median(np.abs(x - np.median(x))), np.finfo(float).eps)
    z = (x - np.median(x)) / mad
    angle = signal.resample(z, ANGLE_SAMPLES)
    order, power = _order_power(angle)
    total = max(float(power.sum()), np.finfo(float).eps)
    out = {
        "shape_skew": float(stats.skew(angle)),
        "shape_kurtosis": float(stats.kurtosis(angle, fisher=False)),
        "shape_crest": float(np.max(np.abs(angle)) / max(np.sqrt(np.mean(angle ** 2)), 1e-12)),
        "zcr": float(np.mean(np.diff(np.signbit(angle)) != 0)),
        "angle_psd_peak_ratio": float(power.max() / total),
        "angle_order_entropy": _entropy(power),
    }
    for lo, hi, name in ((.5, 2, "0_5_2"), (2, 4, "2_4"), (4, 8, "4_8"), (8, 16, "8_16")):
        out[f"angle_order_{name}"] = _relative_energy(order, power, lo, hi)
    # These full-band envelope descriptors are deliberately separate from the
    # fixed-Hz bands.  They make the no_absolute_hz and no_envelope ablations
    # distinct without introducing another device-specific frequency boundary.
    full_env = signal.resample(np.abs(signal.hilbert(z)), ANGLE_SAMPLES)
    _, full_env_power = _order_power(full_env)
    out["envelope_kurtosis"] = float(stats.kurtosis(full_env, fisher=False))
    out["envelope_order_entropy"] = _entropy(full_env_power)
    for lo, hi in BANDS:
        env_angle = signal.resample(_band_envelope(x, lo, hi), ANGLE_SAMPLES)
        env_order, env_power = _order_power(env_angle)
        prefix = f"env_{lo}_{hi}"
        out[f"{prefix}_kurtosis"] = float(stats.kurtosis(env_angle, fisher=False))
        out[f"{prefix}_order_entropy"] = _entropy(env_power)
        for low, high, name in ((.5, 2, "0_5_2"), (2, 4, "2_4"), (4, 8, "4_8"), (8, 16, "8_16")):
            out[f"{prefix}_order_{name}"] = _relative_energy(env_order, env_power, low, high)
    out = {key: float(np.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)) for key, value in out.items()}
    if list(out) != feature_names() or not np.isfinite(list(out.values())).all():
        raise FloatingPointError("Transfer-v2 emitted an invalid schema or non-finite feature")
    return out


def _window_rows(records, target_rpm: float | None) -> list[dict]:
    rows: list[dict] = []
    for path, branch, native_fs, item_label in records:
        x, source_rpm, _, _ = q1.load(path, branch)
        is_target = branch == "target_32k"
        rpm = float(target_rpm) if is_target else float(source_rpm)
        if not np.isfinite(rpm) or rpm <= 0:
            raise ValueError(f"Missing/invalid RPM for {path}")
        x = q1.resample(x, native_fs, FS)
        window = int(round(REVOLUTIONS * FS / (rpm / 60.0)))
        hop = int(round(window / 2))
        if len(x) < window:
            raise ValueError(f"{path.name} is shorter than one eight-revolution window")
        for window_id, start in enumerate(range(0, len(x) - window + 1, hop)):
            row = {"file_id": path.stem, "file_path": path.as_posix(), "window_id": window_id,
                   "branch": branch, "rpm": rpm, "window_samples": window,
                   "window_revolutions": REVOLUTIONS, "angle_samples": ANGLE_SAMPLES,
                   "angle_order_resolution": SAMPLES_PER_REVOLUTION / min(1024, ANGLE_SAMPLES)}
            if not is_target:
                row["label"] = item_label
            row.update(transfer_v2_feature(x[start:start + window]))
            rows.append(row)
    return rows


def build(data_root: Path, output: Path, target_rpm: float = 600.0, write_source: bool = True) -> tuple[pd.DataFrame | None, pd.DataFrame]:
    """Build v2 source once and target at one nominal RPM; never opens target labels."""
    records = q1.records(data_root)
    source_records = [r for r in records if r[1] in ("48k_DE", "48k_normal")]
    target_records = [r for r in records if r[1] == "target_32k"]
    output.mkdir(parents=True, exist_ok=True)
    source = None
    if write_source:
        source = pd.DataFrame(_window_rows(source_records, None))
        if set(source.label.unique()) != set(LABELS) or source.file_id.nunique() != 56:
            raise AssertionError("Transfer-v2 source must contain the 56 formal source files")
        source.to_csv(output / "features_source_transfer_v2.csv", index=False, encoding="utf-8-sig")
    target = pd.DataFrame(_window_rows(target_records, target_rpm))
    if "label" in target or target.file_id.nunique() != 16 or sorted(target.file_id.unique()) != list("ABCDEFGHIJKLMNOP"):
        raise AssertionError("Target Transfer-v2 must be unlabelled A--P")
    target.to_csv(output / f"features_target_transfer_v2_rpm{int(target_rpm)}.csv", index=False, encoding="utf-8-sig")
    schema = {"features": feature_names(), "n_features": len(feature_names()), "sample_rate_hz": FS,
              "revolutions_per_window": REVOLUTIONS, "samples_per_revolution": SAMPLES_PER_REVOLUTION,
              "angle_samples_per_window": ANGLE_SAMPLES, "order_resolution": 0.25,
              "envelope_bands_hz": [list(band) for band in BANDS],
              "angle_domain_note": "Approximate constant-speed angle-domain resampling from recording-level RPM; no tachometer or instantaneous speed trace is available.",
              "target_labels_used": False}
    (output / "transfer_v2_schema.json").write_text(json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8")
    if source is not None:
        duplicates = []
        values = source[feature_names()].to_numpy(float)
        for left in range(values.shape[1]):
            for right in range(left + 1, values.shape[1]):
                if np.array_equal(values[:, left], values[:, right]):
                    duplicates.append([feature_names()[left], feature_names()[right]])
        audit = {"old_feature_count": 29, "new_feature_count": len(feature_names()),
                 "removed_duplicate_feature": "angle_psd_entropy",
                 "added_nonfixed_fullband_envelope_features": ["envelope_kurtosis", "envelope_order_entropy"],
                 "duplicate_feature_pairs": duplicates, "duplicate_feature_check_passed": not duplicates,
                 "check_domain": "source Transfer-v2 numeric feature matrix only"}
        (output / "transfer_v2_schema_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {"source_written": bool(write_source), "source_windows": int(len(source)) if source is not None else None,
               "target_windows": int(len(target)), "source_files": int(source.file_id.nunique()) if source is not None else None,
               "target_files": int(target.file_id.nunique()), "target_nominal_rpm": target_rpm,
               "source_order_resolution": 0.25, "target_order_resolution": 0.25,
               "target_labels_used": False}
    (output / f"transfer_v2_summary_rpm{int(target_rpm)}.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return source, target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("数据集") / "数据集")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/q3_refined"))
    parser.add_argument("--target-rpm", type=float, default=600.0)
    parser.add_argument("--target-only", action="store_true")
    args = parser.parse_args()
    build(args.data_root, args.output_dir, args.target_rpm, write_source=not args.target_only)
    print(f"Transfer-v2 written to {args.output_dir} at target nominal RPM={args.target_rpm:g}")


if __name__ == "__main__":
    main()
