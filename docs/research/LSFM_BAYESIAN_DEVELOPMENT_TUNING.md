# LSFM Bayesian Development Tuning

## Reason

The frozen default development configuration used:

- n_bins = 3
- alpha = 2.0

It passed the budget-1 entropy and MAE gates but selected only two distinct
actions at budget 1, failing the frozen action-diversity gate.

The final-test specimens remain untouched.

## Frozen development-only grid

n_bins:

- 3
- 4
- 5

empirical-joint alpha:

- 1.0
- 2.0
- 4.0

Exactly nine configurations will be evaluated.

No state/action sets, features, split, policies, or endpoints are changed.

## Mechanical eligibility

A configuration is eligible only if:

1. EIG selects >=3 distinct actions at budget 1;
2. EIG budget-1 entropy is lower than random budget-1 entropy;
3. EIG budget-1 posterior-mean MAE is no more than 2 um worse than random;
4. EIG budget-3 zero-support fraction is <0.50.

"Dominant zero support" is operationally defined as >=50% of episodes.

## Selection rule

Among eligible configurations:

1. choose the lowest average EIG posterior entropy over budgets 1, 2, and 3;
2. tie-break with lower average EIG posterior-mean MAE;
3. then prefer fewer bins;
4. then prefer alpha closest to 2.0.

The untouched final test will be evaluated exactly once using the selected
configuration.
