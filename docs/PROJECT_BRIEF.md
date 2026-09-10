# One-page project brief

## Active counterfactual verification for fluorescence microscopy

### Problem
A fluorescence microscopy model can be confident for the wrong reason. Standard uncertainty or OOD scores can reveal that a sample looks unfamiliar, but they do not directly test whether a specific biological prediction depends on acquisition nuisance such as illumination, detector gain, focus, exposure, or optical transfer differences.

### Proposed idea
After the first prediction, choose a low-cost reacquisition of the **same specimen** that is maximally informative about two competing explanations: **biology-driven evidence** versus an **acquisition-driven shortcut**. Candidate interventions are scored by expected information gain, minus photon-dose and acquisition-time penalties. The new measurement updates evidence for the two explanations, producing **SUPPORTED / FALSIFIED / ABSTAIN**.

### Central hypothesis
For matched acquisition budget, intervention selection optimized to discriminate biology-vs-nuisance explanations will reduce false-support errors more efficiently than uncertainty-guided reacquisition, OOD-guided reacquisition, fixed robustness panels, or random perturbations.

### Minimal experimental path
1. Simulated nuisance mechanisms with known ground truth.
2. Retrospective replay on multi-condition fluorescence data.
3. Hardware-in-the-loop validation on the same field of view.
4. Biological task benchmark with controlled nuisance shortcuts.

### Hardware questions
- Which optical parameters can be changed quickly and repeatably?
- Which interventions primarily alter acquisition nuisance rather than biology?
- How should dose be measured or proxied across exposure/illumination changes?
- What registration accuracy is needed for same-specimen evidence comparison?
- What latency is achievable for closed-loop acquisition?

### Success criterion
The project succeeds only if active counterfactual verification improves selective correctness or false-support detection **at matched dose/time**, not merely if it creates visually different images.
