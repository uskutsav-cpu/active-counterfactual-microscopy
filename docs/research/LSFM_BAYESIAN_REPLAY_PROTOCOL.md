# LSFM Bayesian Multi-Action Replay Protocol

## Status

Frozen before development-policy evaluation.

This is optical-state engineering only.

It is not biological validation, hardware validation, or a test of the final
biology-vs-nuisance counterfactual claim.

## Source evidence

The response matrix is the frozen LSFM relative-acquisition engineering result:

- 42 real specimens
- 10 hidden initial focus states/specimen
- 6 real relative optical actions/state
- 2,520 real same-specimen response pairs
- every action previously passed the incremental-information gate

## Hidden optical states

Ten states:

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

## Candidate actions

Six acquisitions relative to the original baseline stage position:

- -18 um
- -12 um
- -6 um
- +6 um
- +12 um
- +18 um

An action is never repeated within an episode.

## Specimen split

Seed: 20260911

Exactly 42 specimens are deterministically divided into:

- calibration: 24 specimens
- development: 9 specimens
- final test: 9 specimens

All 10 hidden states and all six actions from one specimen remain in the same
split.

The final test split must not be inspected while modifying the model.

## Optical evidence model

The model is a discrete empirical full-panel joint likelihood.

Calibration-only fitting:

1. baseline optical features are discretized with KMeans;
2. each action's incremental evidence is discretized with its own KMeans model;
3. each calibration episode becomes one vector containing the baseline outcome
   plus all six action outcomes;
4. the hidden target is one of the ten signed focus states;
5. posterior conditioning uses the same smoothed empirical joint law.

This deliberately avoids multiplying action images as if reacquisitions were
independent.

Default discretization:

- bins per observation: 3
- KMeans initializations: 20
- empirical-joint pseudomass alpha: 2.0
- state prior pseudocount: 1.0

## Baseline observation

The baseline observation is available before any active acquisition and is
included once in the posterior.

Action evidence contains only incremental baseline-to-candidate features, so the
baseline feature vector is not double-counted.

## Bayesian posterior

For hidden state s and observed history h:

P(s | h) is computed by Bayes' rule using the smoothed calibration-only
full-joint likelihood.

## Expected information gain

For each unobserved action a:

EIG(a) =
H[P(s | h)]
-
E_{z ~ P(z | h,a)} H[P(s | h,a,z)]

where z is the discretized evidence outcome.

EIG never receives the true hidden state or an unrevealed candidate image.

## Policies

- eig
- random
- fixed_positive
- fixed_negative
- fixed_alternating
- expected_quality
- posterior_greedy
- oracle_focus

`oracle_focus` is an upper-bound reference and is excluded from inferential
EIG-vs-baseline comparisons because it receives the hidden true state.

## Budgets

Every non-oracle policy is evaluated at:

- 1 acquisition
- 2 acquisitions
- 3 acquisitions

## Primary engineering outcomes

- posterior-mean absolute hidden-state error
- posterior entropy

## Secondary outcomes

- MAP-state absolute error
- fraction within 6 um
- fraction within 12 um
- probability assigned to the true state
- entropy reduction
- closest acquired image's residual defocus
- oracle best-view regret
- action-selection diversity
- empirical joint-history support

## Statistics

Paired comparisons are grouped by specimen.

For EIG vs each non-oracle baseline:

- EIG-minus-baseline mean difference
- 95% specimen bootstrap confidence interval
- 2,000 bootstrap repetitions

Hidden states from one specimen are not treated as independent bootstrap units.

## Development gate

Do not open the final test until the development run is complete and artifact
integrity is verified.

A development run is considered mechanically healthy only if:

1. EIG selects at least three distinct actions at budget 1;
2. EIG lowers mean posterior entropy relative to random acquisition at budget 1;
3. EIG does not have worse posterior-mean MAE than random by more than 2 um at
   budget 1;
4. zero-support histories are not dominant at budget 3.

If the gate fails, modifications are allowed using calibration/development data,
but the final test remains untouched.

## Final-test rule

Once final-test results are opened:

- do not retune bin count
- do not change alpha
- do not change policies
- do not change state/action sets
- do not change endpoints
- do not alter the split

Any redesign after final-test inspection requires new untouched evidence.

## Claim boundary

Passing this study means that EIG can be evaluated as a real multi-action
optical-state acquisition policy.

It still does not establish biological counterfactual verification.
