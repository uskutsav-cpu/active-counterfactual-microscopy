# Methods

## 1. Variables

Let:

- `x0` be the initial fluorescence observation;
- `z` be the biological quantity/prediction of interest;
- `n` denote acquisition nuisance;
- `a` be an allowed optical intervention;
- `y_a` be the reacquired observation or a verification statistic extracted from it;
- `H_B` denote the biology-driven explanation;
- `H_N` denote the nuisance-driven explanation.

The initial model produces a biological prediction and evidence state `D0`. Verification asks whether a carefully chosen intervention generates evidence more likely under `H_B` or `H_N`.

## 2. Predictive evidence model

For every candidate intervention `a`, model

`p(y_a | H_B, D)` and `p(y_a | H_N, D)`.

The current prototype represents `y_a` as a finite evidence state and stores these predictive distributions directly. Later versions can use conditional density models over logits, feature deltas, registered intensity statistics, segmentation changes, or task-specific outputs.

## 3. Acquisition utility

The default policy maximizes

`U(a) = I(H; Y_a | D) - lambda_dose * C_dose(a) - lambda_time * C_time(a)`.

For binary `H`, the mutual information can be computed exactly from the weighted Jensen-Shannon divergence between the two hypothesis-conditional predictive distributions.

A useful normalized secondary score is

`I(H;Y_a|D) / (eps + alpha*C_dose(a) + beta*C_time(a))`.

It should be reported as an analysis metric, not silently substituted for the preregistered selection objective.

## 4. Posterior update

After observing outcome `y` under action `a`:

`P(H_B | y,a,D) ∝ P(y | H_B,a,D) P(H_B|D)`.

The nuisance posterior follows analogously.

## 5. Verdicts

With preregistered thresholds `tau_support` and `tau_falsify`:

- `SUPPORTED` if `P(H_B | D) >= tau_support`;
- `FALSIFIED` if `P(H_B | D) <= tau_falsify`;
- `ABSTAIN` otherwise.

The thresholds must be selected without test-set leakage. Calibration should be reported.

## 6. What must change in a real implementation

The scientific challenge is constructing hypothesis-conditional predictive models that correspond to plausible biology and nuisance mechanisms. Candidate approaches include nuisance simulators, controlled calibration data, hierarchical generative models, invariance models, causal latent-variable models, and repeated acquisitions with known parameter changes.

The framework should not assume that all prediction change is nuisance-driven: bleaching, motion, morphology changes, signaling dynamics, and phototoxicity can change real biology during the verification loop.
