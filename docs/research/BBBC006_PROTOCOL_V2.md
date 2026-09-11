# BBBC006 v2 Controlled Focus-Shortcut Study

## Status

Frozen before v2 policy evaluation.

This is a retrospective controlled real-image shortcut benchmark.

It is NOT:
- independent biological validation
- physical microscope validation
- a prospective experiment
- a completely independent external confirmation, because BBBC006 was already used in v1

## Dataset

BBBC006 v1, Hoechst channel.

Selected planes:

- z08: out of focus
- z12: expert-defined in-focus region
- z16: optimal focal plane
- z24: out of focus

## Split

Seed: 20260911

Groups are wells.

Group-disjoint fractions:

- train: 40%
- calibration: 25%
- development: 15%
- test: 20%

The split seed is frozen independently of v2 performance.

## Prediction task

Official z16 CellProfiler nucleus count converted into a binary task.

Count threshold:

108.0

The threshold was chosen as the median count of the TRAINING split only.

## Controlled acquisition shortcut

Possible initial acquisition conditions:

- z08
- z12

Training baseline/label alignment:

0.95

Calibration/development/test alignment:

0.50

Thus focus condition is deliberately predictive of class during training and
the relationship is broken outside training.

## Candidate counterfactual reacquisitions

- z16
- z24

The policy cannot inspect either candidate image before selecting it.

## Predictor

Logistic regression.

## Policies

- EIG
- uncertainty
- OOD
- image quality
- random
- fixed
- no reacquisition

## Budgets

- 1 acquisition
- 2 acquisitions

## Primary outcome

Selective verdict risk at matched acquisition budget.

Secondary outcomes:

- conditional false support
- joint false support
- coverage
- Brier score
- acquisition rate
- dose proxy
- action-selection distribution

False support will never be interpreted without coverage.

## Cost analyses

Two analyses are prespecified.

### Primary: cost-aware EIG

- dose weight: 0.03
- time weight: 0.05
- negative-utility stopping enabled

### Sensitivity: information-only EIG

- dose weight: 0
- time weight: 0
- negative-utility stopping enabled

The sensitivity analysis is included because retrospective panel costs are
unit proxies rather than measured physical photon dose or wall-clock time.

## Interpretation

Evidence for active selection requires more than improvement over no
reacquisition.

If EIG behaves identically to fixed acquisition across all specimens, v2 does
not demonstrate an action-selection advantage.

No policy or threshold will be changed after viewing v2 test results.

## Claim boundary

A positive v2 result establishes only retrospective controlled focus-shortcut
performance on BBBC006.

Prospective microscope and biological validation remain separate future gates.
