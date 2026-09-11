# BBBC006 Real-Image Replay Protocol

## Status

Frozen before policy evaluation.

This is a retrospective real-image replay benchmark.
It is not physical microscope validation and is not independent biological ground truth.

## Dataset

Broad Bioimage Benchmark Collection BBBC006 v1.

Selected focal planes:

- z08: initial degraded acquisition
- z16: optimal-focus candidate acquisition
- z24: out-of-focus candidate acquisition

Channel:

- w1 / Hoechst nuclear fluorescence

Pairing:

- same well + site identifies the same field across focal planes
- wells are the independent split groups

## Reference task

Binary nucleus-count task using the official BBBC006 CellProfiler count associated with z16.

Frozen threshold:

- count >= 50 -> positive
- count < 50 -> negative

The count is an algorithm-generated reference quantity, not independent manual biological truth.

## Split

Seed: 42

Group split occurs at the well level before model training:

- training: 40%
- calibration: 25%
- development: 15%
- test: 20%

No well may appear in more than one split.

## Replay

Stored starting view:

- z08

Hidden candidate reacquisitions:

- z16
- z24

The policy may not inspect candidate images before selecting an acquisition.

## Predictor

- logistic regression

## Policies

- EIG
- random
- fixed
- uncertainty
- OOD
- image quality
- no reacquisition

## Acquisition budgets

- 1
- 2

Budget 3 is not used because only two distinct candidate reacquisitions exist.

## Decision target

Prediction correctness.

## Primary interpretation

The primary analysis jointly considers:

1. false-support rate
2. selective verdict risk
3. coverage

No policy will be declared superior solely because it abstains enough to obtain a low false-support rate.

## Statistical comparison

Specimen-paired / group-aware bootstrap comparisons as implemented in the frozen research engine.

## Restrictions

The test split will not be used for:

- threshold tuning
- model selection
- action selection changes
- evidence-model redesign
- hyperparameter tuning

Changes motivated by final test performance require a new untouched validation design.

## Claim boundary

A positive result supports performance on retrospective same-field real-image focus replay.

It does not by itself establish:

- physical microscope intervention validity
- nuisance-only causality
- biological truth
- phototoxicity safety
- prospective closed-loop performance
