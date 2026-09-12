# LSFM Bayesian Final Configuration

## Status

Frozen before opening the final-test policy results.

## Selected configuration

Development-only tuning selected:

- n_bins: 5
- alpha: 1.0
- seed: 20260911
- bootstrap repetitions: 2000

## Development gate

The selected configuration had:

- budget-1 EIG action diversity: 4
- budget-1 EIG entropy: 2.412492
- budget-1 random entropy: 2.546480
- budget-1 EIG posterior-mean MAE: 9.508000 um
- budget-1 random posterior-mean MAE: 9.676501 um
- budget-3 zero-support fraction: 0.200000
- average EIG entropy across budgets: 2.146277
- average EIG MAE across budgets: 9.537330 um

This configuration was selected using the frozen development selection rule.

## Final-test rule

The nine final-test specimens will now be evaluated exactly once.

After final-test inspection:

- n_bins will not change
- alpha will not change
- policies will not change
- states/actions will not change
- endpoints will not change
- specimen split will not change
- the final test will not be rerun as a tuning set

Any later redesign requires new untouched evidence.

## Claim boundary

This is optical-state engineering validation.

It is not yet biological counterfactual validation or prospective hardware validation.
