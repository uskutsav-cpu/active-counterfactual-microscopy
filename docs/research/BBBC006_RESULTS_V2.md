# BBBC006 v2 Controlled Focus-Shortcut Result

## Status

Frozen after completion of the prespecified BBBC006-v2 analyses.

This is a retrospective real-image result.

It is not biological validation, physical microscope validation, or
prospective closed-loop validation.

## Design audit

The intended acquisition shortcut was successfully created and then broken
outside the training split.

| split | specimens | groups | positive fraction | baseline-label alignment |
| --- | ---: | ---: | ---: | ---: |
| train | 306 | 153 | 0.5163 | 0.9477 |
| calibration | 192 | 96 | 0.5260 | 0.5208 |
| development | 114 | 57 | 0.5175 | 0.4912 |
| test | 156 | 78 | 0.6218 | 0.5321 |

Starting conditions were z08 and z12.

Candidate counterfactual reacquisitions were z16 and z24.

The count threshold was 108.0 and was selected from the training split only.

## Cost-aware EIG

With the prespecified proxy cost weights:

- dose weight = 0.03
- time weight = 0.05

EIG selected no acquisitions.

Budget 1:

- coverage = 0.2500
- conditional false support = 0.0588235294
- selective verdict risk = 0.0512820513
- mean dose = 0.0

Budget 2:

- coverage = 0.2500
- conditional false support = 0.0588235294
- selective verdict risk = 0.0512820513
- mean dose = 0.0

These values were identical to the no-reacquisition policy.

Therefore the proxy experimental-cost scale remained too large relative to
estimated information gain for this retrospective BBBC006 replay.

## Information-only EIG sensitivity

With both proxy cost weights set to zero, EIG did acquire images.

Budget 1:

- coverage = 0.3590
- conditional false support = 0.0588235294
- selective verdict risk = 0.0535714286
- mean dose = 1.000

Budget 2:

- coverage = 0.4167
- conditional false support = 0.0588235294
- selective verdict risk = 0.0307692308
- mean dose = 1.974

Thus information-only EIG increased coverage relative to no reacquisition.

However, it did not reduce conditional false support.

At acquisition budget 2, the principal acquisition policies converged to
essentially the same coverage and selective-risk result.

## Paired comparisons

For conditional false support, EIG-minus-baseline differences were 0.0
against every tested baseline at both budgets.

The stored paired group-bootstrap confidence intervals were [0.0, 0.0].

Therefore BBBC006-v2 provides no evidence for an EIG-specific false-support
advantage.

## Scientific conclusion

BBBC006-v2 is a negative result for the strong claim that counterfactual EIG
selects diagnostically superior reacquisitions on this benchmark.

The experiment nevertheless establishes several useful facts:

1. the controlled focus shortcut was successfully induced in training and
   broken outside training;
2. the same-field retrospective replay machinery functions on real
   fluorescence images;
3. removing proxy acquisition costs causes EIG to reacquire;
4. reacquisition can increase coverage;
5. the current BBBC006 action catalog and evidence structure do not produce a
   unique EIG action-selection advantage.

The result should not be converted into a positive claim by changing
thresholds, policies, evidence models, or endpoints after observing this test
set.

Any such redesign must be evaluated on new untouched data.

## Claim boundary

The result does not establish:

- biological validity;
- physical intervention validity;
- calibrated phototoxicity or photon cost;
- prospective microscope performance;
- superiority over uncertainty-guided adaptive acquisition.

BBBC006 should now be considered development/exploratory data for this
project rather than a source of a new confirmatory result.
