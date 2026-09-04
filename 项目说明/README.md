# 2025 Huawei Cup E — bearing-fault transfer analysis

This repository contains the competition materials, data audit, and a
reproducible first-question pipeline for a bearing-fault domain-transfer task.
The current main deliverable is deliberately conservative: files A–P have no
true labels, so this repository does **not** claim a target-domain accuracy.

## Current main pipeline: Question 1

`src/q1_pipeline.py` is the current entry point. It performs the work that
Question 1 requires before any target-domain classifier is trusted:

- audits every `.mat` file and writes file-level metadata;
- compares candidate source domains on a shared 12 kHz / 0–6 kHz view;
- selects the full `48kHz_DE_data` fault set plus `48kHz_Normal_data` as the
  formal source domain for its DE measurement location, label coverage and
  bandwidth (MMD/PAD are diagnostic evidence only);
- anti-alias resamples 48 kHz source signals to 32 kHz, extracts time,
  spectral, envelope and order-aware features, and removes near-zero-variance
  or highly correlated features;
- produces mechanism, correlation and descriptive PCA figures;
- keeps the historical controlled 16-file set only for mechanism illustration,
  not as the formal training source domain.

Run from the repository root:

```powershell
python .\src\q1_pipeline.py
```

Results are written to `outputs\q1\`. The key files are:

- `data_audit.csv` and `data_audit_summary.json`: all scanned MAT files and
  branch counts;
- `source_domain_selection.csv`: candidate-domain MMD and proxy A-distance;
- `source_metadata.csv`: formal source-file metadata;
- `features_source_raw.csv` and `features_source_selected.csv`: window-level
  feature datasets;
- `feature_quality.csv` and `feature_names.json`: feature-screening evidence;
- `figures\`: publication-ready mechanism and feature-quality figures;
- `q1_summary.md`: compact machine-generated run summary.

For the latest executed run and the LOLO/ablation/window-validation conclusion,
see `当前运行结果与待反馈问题.md` and `第一问封版验证结果.md` in this directory.

## Older diagnostic baselines (not Question 1 conclusions)

- `src/bearing_mvp.py` is the earlier controlled 16-file, source-only/CORAL
  diagnostic baseline. Its A–P output is only a candidate plus review signal.
- `src/dann_feature_baseline.py` is an optional feature-level DANN diagnostic
  baseline. Its gradient-reversal idea is documented in
  `THIRD_PARTY_NOTICES.md`; it is not evidence of target classification.

Their old outputs are intentionally not part of the current Question 1 result
package. Do not use source-only, CORAL or DANN candidate labels as A–P truth.

## Scientific boundary

- Split, validation and reporting must be grouped by raw `.mat` file; overlapping
  windows are not independent samples.
- MMD, proxy A-distance, PCA/t-SNE/UMAP, confidence and pseudo-label agreement
  do not establish target-domain accuracy.
- Without target bearing geometry, a target spectral peak must not be called
  BPFO, BPFI or BSF.

The complete current evidence, limitations and questions for external review
are in `当前运行结果与待反馈问题.md`.

## Question 2 source-domain diagnosis

Run the nested file-level diagnostic benchmark with:

```powershell
python .\src\q2_pipeline.py
```

It reads only `outputs\q1\features_source_diagnostic.csv`, writes results to
`outputs\q2\`, and saves an MLP encoder/scaler/schema interface for Question 3.
All reported metrics are source-domain, file-level metrics; no target A–P label
or target-domain accuracy is used.
