# Synthetic study — frozen result

## Status

Computational validation only. This is not biological or hardware validation.

The frozen synthetic study contains:

- 53 complete experiment runs
- 267,120 policy/test episodes
- 18 replication configurations
- 12 misspecification configurations
- 6 acquisition-cost configurations
- 12 critical-ablation configurations
- 4 explicit structural/negative-control runs
- one base run
- zero artifact-integrity problems
- one frozen research-source hash

Master analysis commit:

`d9646898ba408c4628ccc71d59c4853a60467710`

## Primary conclusion

Cost-aware expected-information-gain (EIG) acquisition is not uniformly superior to every
alternative under every synthetic condition.

The evidence instead supports a narrower result:

> EIG can provide a favorable verification risk–coverage tradeoff, particularly under
> acquisition-shift / misspecification and cost-sensitive conditions, but its advantage is
> architecture-, nuisance-, and acquisition-budget-dependent.

This conclusion is frozen before real-image evaluation.

## Important results

### Standard replication

Across model type, nuisance strength, and seed, EIG generally increased coverage.

Against uncertainty:

- budget 1:
  - coverage difference: +0.1262
  - false-support difference: -0.1318
  - selective-risk difference: -0.00665

- budget 2:
  - coverage difference: +0.0764
  - false-support difference: -0.0322
  - selective-risk difference: approximately neutral

- budget 3:
  - coverage difference: +0.0681
  - false-support difference: +0.0366
  - selective-risk difference: -0.00193

Thus additional acquisitions do not monotonically improve every metric.

### Misspecification study

At budget 3, versus uncertainty:

- coverage difference: +0.0233
- false-support difference: -0.0368
- selective-risk difference: -0.0516
- EIG had lower false support in 12/12 configurations
- EIG had lower selective risk in 12/12 configurations

Versus fixed acquisition:

- false-support difference: -0.0536
- selective-risk difference: -0.0624
- lower false support in 12/12 configurations

This is the strongest synthetic evidence for the method.

### Cost sensitivity

At budget 3, versus uncertainty:

- coverage difference: +0.0153
- false-support difference: -0.0222
- selective-risk difference: -0.0297

The false-support and selective-risk directions favored EIG in all six cost configurations.

Versus fixed acquisition:

- false-support difference: -0.0444
- paired 95% CI favored EIG in all 6/6 cost configurations
- selective-risk difference: -0.0456

### Negative calibration control

Shuffling calibration targets changed EIG from:

- coverage: 0.325 -> 0.0583
- selective verdict risk: 0.0385 -> 0.2143
- Brier score: 0.1557 -> 0.2809

The apparent false-support rate became zero because the system resolved very few cases.
This demonstrates why false support must be interpreted jointly with coverage and selective risk.

### Cost ablations

Removing the dose penalty left the primary decision metrics essentially unchanged but increased
mean dose from 2.073 to 2.398.

Removing the time penalty produced little change in this simulator.

Removing the registration gate also produced little change in this controlled synthetic setting;
its main value remains real-image and hardware safety.

## Interpretation limits

- Grid cells are experimental configurations, not independent biological specimens.
- Synthetic correctness is not proof that predictions are biology-driven.
- Cross-configuration means are descriptive, not pooled clinical confidence intervals.
- No result here constitutes physical microscope validation.
- No result here constitutes biological validation.
- The method will not be modified in response to these results without using new untouched
  validation conditions.

## Next gate

Real-image replay on BBBC006, followed by controlled physical microscope validation if the
real-image result survives.
