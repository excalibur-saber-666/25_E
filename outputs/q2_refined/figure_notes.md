# Figure contract and interpretation

Python quantitative comparisons, editable SVG/PDF and 400 dpi PNG. Models are
compared on 56 unique source MAT files, four load folds, five seeds. Repeated
seeds reuse the same files; they are not extra independent observations.
Seed scatter and ablation error bars: mean over four load scores within each
seed, then seed mean and sample SD (not a 95% CI). All five seeds retained.
Confusion: cell counts averaged over seeds, sums to 56. Persistent errors:
all files with at least one wrong seed for the displayed model.
Reliability: ten fixed equal-width confidence bins, pooled seed-file predictions
for descriptive visualization only, empty bins absent. Unique-file counts and
prediction counts are exported separately. No independence or p-value claim.
PCA is explanatory, fitted on loads 1–3 only. Feature shift uses all 26 file-mean
features; scaling uses training-file overall SD, not held-out SD. Associations
are not causal evidence. Models, errors, calibration and ablation each have
machine-readable CSV source data beside this file. No target data used.
Static QA warnings reviewed: TIFF is not required by this task, PNG is a 400 dpi
preview with SVG/PDF as vector masters. Random draws are file-bootstrap indices,
not synthetic observations. Log probabilities are clipped at 1e-12 and the
temperature search uses strictly positive bounds. These are not unguarded logs.
