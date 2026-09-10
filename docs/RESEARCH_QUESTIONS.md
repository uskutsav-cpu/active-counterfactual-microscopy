# Research questions, claims, and falsifiers

## RQ1 — Can active counterfactual verification detect acquisition shortcuts?

**Claim:** targeted reacquisition reduces false-support errors relative to no reacquisition and cost-matched non-causal baselines.

**Falsifier:** equal-cost random/fixed reacquisition performs equivalently, or observed gains are fully explained by improved SNR/image quality.

## RQ2 — Does expected information gain select better interventions than uncertainty?

**Claim:** interventions that best discriminate `H_B` vs `H_N` are not always those with maximum predictive uncertainty.

**Falsifier:** after matching action cost and candidate set, uncertainty-guided acquisition performs equally across nuisance mechanisms.

## RQ3 — Does cost-aware selection matter?

**Claim:** explicit dose/time penalties improve information gained per unit experimental cost.

**Falsifier:** cost weighting only shifts actions without improving any Pareto frontier.

## RQ4 — When should the system abstain?

**Claim:** abstention protects against model misspecification, weak interventions, failed registration, and temporal biological change.

**Falsifier:** abstention is poorly calibrated or simply hides errors without improving selective risk.

## Critical ablations

- Remove dose penalty.
- Remove time penalty.
- Replace EIG with predictive entropy.
- Replace H_B/H_N model with generic OOD score.
- Remove posterior update and use a single fixed robustness threshold.
- Randomize intervention labels while preserving costs.
- Remove registration-quality gating.
- Add unmodeled nuisance mechanisms.
