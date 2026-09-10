# Implemented methods and claim boundary

## Targets are not interchangeable

For the default `correctness` target, H=1 means the initial prediction equals a reference task label; H=0 means it does not. This is an operational reliability-verification experiment. It is not an operational definition of whether a model uses biology.

The `annotated_hypothesis` mode uses independently provided `reference_label` values and requires `reference_source`. A causal analysis should establish those annotations through controlled mechanisms or independent experimental evidence. Predictions can be correct through shortcuts, wrong despite biological evidence, or stable through an untested nuisance. Mixed mechanisms need an explicit study design; a binary model is not automatically an exhaustive causal description.

## Data partitioning and conditioning

Training, calibration, development and test groups are disjoint. The biological predictor and feature OOD reference are fitted on training images. The evidence quantizer and action-outcome models are fitted on calibration panels. Optional decision threshold selection uses development groups. Test reference labels are attached only after action selection and inference finish.

Baseline context is a fixed coarse partition of initial predicted class and confidence. Sparse contexts back off to the whole calibration population. This is not a learned universal causal latent model.

## Image evidence

A candidate image produces seven statistics: signed prediction-probability change relative to the initial predicted class, absolute probability change, training-scale-normalized feature displacement, focus-quality change, one minus registered correlation, estimated translation, and candidate predictive entropy. A calibration-fitted quantizer converts this vector into one of K evidence states.

No per-image min–max normalization is used in prediction preprocessing because it would remove acquisition brightness information. Registration uses intensity-centered images only for alignment, followed by non-wrapping overlap checks. Integer translation is a deliberately limited registration model; large deformation, 3D structure changes and live-cell motion are not automatically solved.

## Joint law for repeated evidence

Let the calibration panel's discrete vector be S=(S_1,...,S_A), with one outcome per candidate action. For hypothesis h and fixed baseline context c, the model is the empirical full-panel distribution plus a total uniform pseudomass alpha over the K^A possible vectors.

For an observed partial history D containing m distinct actions, its hypothesis-conditional probability is:

    P(D | h,c) = (N_h,c(D) + alpha / K^m) / (N_h,c + alpha).

For a new action a:

    P(S_a=s | h,c,D) = (N_h,c(D,S_a=s) + alpha/K^(m+1))
                      / (N_h,c(D) + alpha/K^m).

The posterior is updated using the full history. This is not a product of independent per-view likelihoods. The included tests check Bayes consistency, conditioning-order invariance and redundant correlated evidence. The model becomes sparse as histories lengthen; a minimum calibration-match gate forces abstention rather than interpreting an unsupported leaf confidently.

This static-panel model does not automatically model bleaching, order-dependent physiology or fresh repeated exposures. Replay forbids revealing the same stored action twice. A physical experiment must test whether its acquisition sequence is adequately represented by the calibration process.

## Action objective and comparison

EIG is the posterior-weighted Jensen–Shannon divergence of the two conditional outcome distributions, bounded by current binary-hypothesis entropy. Utility subtracts dose and time penalties. Hard step/dose/time bounds are enforced independently of this soft objective.

Random, fixed, uncertainty, OOD and quality selectors share the same available action menu and inference layer. The last three estimate future action scores from baseline-only features using calibration-fitted regressors. They do not examine hidden candidate test images. They are practical baseline implementations, not reproductions of a specific paper.

Identical budget caps do not imply identical realized cost. Report realized dose/time and risk/coverage together. A no-reacquisition policy incurs zero extra-acquisition cost. Failed attempted acquisitions retain their reserved cost. Gain changes do not make reacquisition dose-free.

## Verdicts and calibration

Support/falsify thresholds are either prespecified or selected on development data. Failed registration, initial OOD or sparse evidence triggers abstention. Optional threshold tuning groups repeated fields and counts a group as erroneous if any selected field has an erroneous verdict. A Bonferroni correction covers a fixed 24-candidate threshold search within one policy/budget evaluation. It does not cover selecting favorable policies, budgets, datasets or final claims after inspecting results; it also assumes independent exchangeable development groups.

## Endpoint definitions

- Conditional false support: P(SUPPORTED | H=0).
- Joint false support: P(SUPPORTED and H=0).
- Supported-prediction risk: P(H=0 | SUPPORTED).
- Coverage: fraction returning either support or falsification.
- Selective verdict risk: erroneous binary verifier decisions among non-abstentions.

These are different denominators. An all-abstain method can have zero conditional false support while verifying nothing. Undefined denominators are null, not zero. Brier and calibration diagnostics refer to the stated H target, not an independently established causal probability.

Paired percentile bootstrap resamples groups, preserving matched specimens across policies. The included primary interval is not adjusted across comparisons. Use the pilot to design final sample size and preregister the final endpoint and independent sampling unit.
