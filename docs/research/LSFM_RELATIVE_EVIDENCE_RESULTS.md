# LSFM Relative-Action Evidence Result

## Status

Frozen optical-engineering result.

This is not biological validation, hardware validation, or evidence of
EIG superiority.

## Dataset

- 42 real LSFM specimens
- 51 focus planes/specimen
- 2 um axial spacing
- plane 26 = reference focus
- 420 hidden baseline states
- 2,520 real same-specimen intervention pairs
- three specimen-disjoint CV folds of 14 specimens each

## Baseline-image-only result

The baseline image alone was weakly informative about the sign and magnitude
of defocus:

- mean signed-defocus ROC AUC: 0.5805
- mean sign accuracy: 0.5500
- mean signed-offset MAE: 18.201 um

## Relative-acquisition result

| Move | Pair AUC | AUC gain | Offset MAE | MAE reduction | Quality-direction accuracy | Registration valid |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| -18 um | 0.9872 | +0.4067 | 12.124 | +33.4% | 0.9238 | 0.9952 |
| -12 um | 0.9841 | +0.4036 | 11.787 | +35.2% | 0.9577 | 0.9952 |
| -6 um | 0.9722 | +0.3917 | 12.223 | +32.8% | 0.9357 | 1.0000 |
| +6 um | 0.9573 | +0.3768 | 12.300 | +32.4% | 0.9000 | 1.0000 |
| +12 um | 0.9761 | +0.3956 | 11.952 | +34.3% | 0.9365 | 0.9952 |
| +18 um | 0.9661 | +0.3855 | 11.698 | +35.7% | 0.9024 | 0.9929 |

All six actions passed the prespecified incremental-information gate.

Mean pair ROC AUC across the six relative acquisitions:

0.9739

Therefore:

PROCEED TO BAYESIAN/EIG ENGINEERING = TRUE

## Interpretation

A real relative optical acquisition contains substantial held-out information
about hidden signed focus state beyond the information available in the
baseline image alone.

The result supports building a multi-action Bayesian acquisition policy.

It does not yet show:

- EIG beats fixed acquisition
- EIG beats uncertainty-guided acquisition
- EIG beats expected-image-quality acquisition
- biological verification works
- real microscope closed-loop operation works

## Next gate

Train the optical-state likelihood model using only calibration specimens,
then compare adaptive EIG against fixed/random/greedy/quality policies on
held-out specimens at acquisition budgets 1, 2, and 3.
