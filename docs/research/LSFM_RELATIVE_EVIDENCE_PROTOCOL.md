# LSFM Relative-Action Evidence Diagnostic

## Status

Frozen before inspecting results.

This is optical engineering validation only.

It is not:

- biological validation
- hardware validation
- prospective microscope validation
- evidence for the final biological counterfactual claim

## Dataset

Audited LSFM defocus dataset:

- 42 specimens
- 51 complete planes per specimen
- 2 um axial spacing
- plane 26 is the reference focal plane
- signed stack range: -50 to +50 um

## Hidden baseline states

The initial image is drawn from:

- -30 um
- -24 um
- -18 um
- -12 um
- -6 um
- +6 um
- +12 um
- +18 um
- +24 um
- +30 um

The true starting offset is never supplied as an input feature.

## Relative acquisitions

Candidate moves relative to the initial stage position are:

- -18 um
- -12 um
- -6 um
- +6 um
- +12 um
- +18 um

## Leakage controls

- specimen identity is the grouping unit
- all states/actions from one specimen stay in one fold
- exactly the same folds are used for every action
- each action is evaluated separately
- action identity is not an input feature
- true start offset is not an input feature
- absolute candidate plane is not an input feature
- each action is compared against a baseline-image-only decoder

## Primary engineering question

Does observing the result of a real relative optical intervention add
held-out information about hidden signed defocus beyond the information
already available in the baseline image?

## Metrics

Baseline-only and pair-based:

- signed-defocus ROC AUC
- signed-defocus classification accuracy
- signed-offset mean absolute error

Additional pair diagnostics:

- quality direction accuracy
- registration-valid rate

## Continuation gate

One action is considered incrementally informative if all are true:

1. pair signed-defocus ROC AUC >= 0.70
2. pair AUC improves over baseline-only by >= 0.05
3. signed-offset MAE improves over baseline-only by >= 10%

Proceed to Bayesian/EIG engineering only if:

- at least two of six actions pass all three conditions
- mean pair AUC across all actions >= 0.70

## Statistical boundary

The 2,520 intervention pairs are correlated because multiple pairs arise
from the same specimen.

No naive pair-level significance test will be interpreted as if the pairs
were independent.

## Claim boundary

Passing establishes only that real optical reacquisition adds information
about hidden optical state.

It does not establish EIG superiority or biological verification.
