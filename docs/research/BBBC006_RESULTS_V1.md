# BBBC006 v1 real-image replay — frozen result

## Status

Completed retrospective real-image replay.

This result is NOT:
- biological validation
- physical microscope validation
- prospective closed-loop validation

## Frozen protocol

- Dataset: BBBC006 v1
- Channel: w1 / Hoechst
- Initial view: z08
- Candidate reacquisitions: z16, z24
- Task: nucleus count >= 50
- Predictor: logistic regression
- Split grouping: well
- Test specimens: 156
- Test groups: 78
- Policies: EIG, random, fixed, uncertainty, OOD, quality, no reacquisition
- Budgets: 1, 2

## Integrity

- computational run: COMPLETE
- total policy/test episodes: 2,184
- frozen artifacts checked: 35
- artifact mismatches: 0

## Main result

BBBC006-v1 does not provide positive evidence for the active EIG policy.

The initial predictor was correct on 151/156 test specimens and incorrect on only 5.

EIG selected no reacquisition for any test specimen at either budget.

Therefore EIG was behaviorally identical to the no-reacquisition policy.

### EIG

Budget 1 and budget 2:

- coverage: 1.000
- false-support conditional: 1.000
- false-support joint: 0.03205
- selective verdict risk: 0.03205
- Brier score: 0.03083
- mean dose: 0
- mean acquisition time: 0

### Active comparison policies

At budget 1, fixed / OOD / quality approximately achieved:

- coverage: 0.9679
- false-support conditional: 0.600
- selective verdict risk: 0.01987
- mean dose: 1

Uncertainty:

- coverage: 0.9615
- false-support conditional: 0.600
- selective verdict risk: 0.0200
- mean dose: 1

The false-support denominator contains only five incorrect original
predictions, so these estimates are statistically weak.

## Interpretation

The v1 experiment exposes two limitations.

1. The count-threshold task produced an extremely accurate initial predictor,
   leaving only five negative verification cases in the held-out test set.

2. The cost-aware EIG policy never selected a new acquisition. With unit
   acquisition proxies and the frozen cost weights, the predicted information
   value did not clear the action threshold.

The result does not justify modifying the method and reusing the same held-out
test as confirmatory evidence.

Any revised policy or real-image benchmark must be treated as a new study and
validated on new untouched evidence.

## Scientific conclusion

BBBC006-v1 is a completed negative / diagnostic real-image benchmark.

It demonstrates that the frozen EIG implementation is not automatically useful
on an easy same-field focus task and that action-cost calibration and the
frequency of negative verification cases are critical design variables.

This result will be retained and reported.
