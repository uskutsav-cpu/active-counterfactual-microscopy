# Active Counterfactual Microscopy — research engine

An additive, executable research extension to Utsav Sunil Kumar's original discrete-evidence prototype.

**Status: tested computational implementation. No real-microscope validation or causal-identification result is claimed.** Existing modules and fixes are preserved; new code lives under `counterfactual_microscopy.research`.

## Start here

From this repository, after installing `requirements-research.txt` and the editable package:

```bash
python -m counterfactual_microscopy.research smoke --out results/research/smoke
python -m counterfactual_microscopy.research verify-run --run results/research/smoke
```

The smoke workflow does not download anything or connect to a device. It creates paired simulated images, trains a predictor, learns evidence distributions on a separate calibration split, evaluates seven policies under two acquisition-count caps, runs paired group bootstraps, and writes tables, full decision traces, five figures, and an HTML report.

For an existing, identical run directory, append `--resume`. A different configuration or source hash is rejected; start a new directory for a changed experiment.

## What is implemented

| Component | Implementation |
| --- | --- |
| Image formation | Fluorescence-like cells, blur, Poisson noise, Gaussian read noise, gain, exposure, clipping and shading; temporal/unknown-nuisance stress tests |
| Predictors | Train-only logistic regression, random forest, optional small PyTorch CNN |
| Calibration | Baseline-context-conditional, discretized image evidence; smoothed empirical joint acquisition law |
| Sequential inference | Full-history conditioning rather than independent multiplication of correlated views |
| Actions | EIG with costs; random, fixed, expected uncertainty, expected OOD, expected image quality; no reacquisition |
| Fair access | Unchosen test images are hidden behind a provider; baseline action-score predictors use only the initial image at selection time |
| Safety and abstention | Hard dose/time/count caps, charge-before-exposure, initial OOD and registration gates, sparse-history abstention |
| Statistical outputs | Explicit false-support denominators, coverage, selective risk, Brier/ECE/AUC, paired group bootstrap |
| Data | Official-source BBBC005/006 plans/downloads, safe extraction, metadata manifests, source hashes, group-disjoint NPZ panels |
| Hardware | Explicitly armed Java-backend Pycro-Manager adapter, fake-core tests, safe-state restoration and fault latch |
| Reproducibility | Validated YAML, source/config/data hashes, checkpoint receipts, resume, reports, sweeps, environment records |

## The most important scientific distinction

The default target is **whether the original prediction agrees with a reference task label**. `SUPPORTED` means supported for that target under this calibration distribution and action family. It does **not** mean proven biology-driven. A stable prediction may be wrong; a correct prediction may use a shortcut; a focus change may remove genuine task information.

`target: annotated_hypothesis` is available only when independent binary hypothesis annotations and their provenance are supplied. Real causal interpretation requires a defensible annotation/intervention design, not merely changing the target's name. See [implemented methods](docs/research/METHODS_IMPLEMENTED.md).

The expected-uncertainty/OOD/quality policies are transparent baseline implementations, not reproductions of named published papers. Their action-selection regressors are fitted on calibration panels and do not inspect candidate test views before selection.

## Outputs

A successful run has `status.json` with `state: COMPLETE` **and** `biological_validation: false`, `hardware_validation: false`.

`test_episodes.json` contains every selected action, candidate score, actual revealed image index, probability update, evidence state, gate failure and reference label (attached after inference). `summary.csv` defines endpoint values; `paired_comparisons.json` contains confidence intervals. `REPORT.md` and `report.html` interpret the same saved rows. `trusted_model.joblib` is executable Python serialization: do not load models from untrusted sources.

## Execution guide

Read [RUNBOOK](docs/research/RUNBOOK.md) for actual runnable commands, [DATA_GUIDE](docs/research/DATA_GUIDE.md) before multi-gigabyte downloads, and [HARDWARE_GATE](docs/research/HARDWARE_GATE.md) before any physical experiment.

[COMPLETION_GATES](docs/research/COMPLETION_GATES.md) distinguishes software completion from empirical research completion. The [preregistration template](docs/research/PREREGISTRATION_TEMPLATE.md) is a draft, not a registered study.
