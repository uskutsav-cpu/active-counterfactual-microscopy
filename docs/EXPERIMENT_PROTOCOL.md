# Experiment protocol

## Goal
Test whether cost-aware counterfactual reacquisition identifies nuisance-driven AI predictions more efficiently than passive or non-causal acquisition baselines.

## Phase A — synthetic ground truth

Construct simulated tasks where the prediction can depend on:

1. biology only;
2. nuisance only;
3. a controlled mixture of biology and nuisance;
4. an unmodeled nuisance mechanism;
5. temporal biological change during verification.

Sweep signal-to-noise ratio, shortcut strength, calibration error, action cost, and intervention efficacy.

### Required result
The proposed policy must beat random/fixed acquisition when its causal model is approximately correct, while exposing degradation or abstention when the model is misspecified.

## Phase B — retrospective acquisition replay

Use samples imaged under multiple optical conditions. Treat one condition as time `t0`; hide the others as candidate reacquisitions. The policy chooses which hidden condition to reveal next.

This phase tests acquisition logic before live microscope integration, but it does **not** reproduce all closed-loop effects such as bleaching and latency.

## Phase C — hardware-in-the-loop

For each field of view:

1. acquire baseline image;
2. compute prediction and candidate-action utilities;
3. execute selected optical perturbation;
4. reacquire and register;
5. update posterior and emit verdict;
6. log dose/time/latency and restore baseline state if required.

Randomize acquisition-policy order across specimens where possible. Include repeated no-change acquisitions to estimate intrinsic measurement variability.

## Baselines

- No reacquisition.
- Random valid reacquisition.
- Fixed robustness panel.
- Predictive-uncertainty-guided reacquisition.
- OOD/embedding-shift-guided reacquisition.
- Image-quality-optimized reacquisition.

## Primary metrics

- false-support rate;
- correct verification rate;
- falsification recall;
- selective risk vs coverage;
- abstention rate;
- resolved cases per photon-dose unit;
- resolved cases per acquisition second.

## Secondary metrics

- calibration / Brier score of `P(H_B)`;
- posterior entropy reduction;
- regret relative to oracle action choice;
- registration failure rate;
- action execution repeatability;
- total closed-loop latency.

## Statistical plan

Report paired comparisons because policies can be replayed on matched specimens when the acquisition design permits it. Use specimen-level resampling rather than frame-level resampling when multiple frames come from the same biological sample. Predeclare the primary endpoint and confidence interval procedure before final biological testing.

## Failure criteria

The main claim is not supported if gains disappear at matched dose/time, if the method succeeds only because selected actions improve image quality, if performance depends on test-set-tuned thresholds, or if “falsification” is better explained by drift/bleaching/biological dynamics than by nuisance-shortcut evidence.
