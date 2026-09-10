# Draft preregistration template — NOT preregistered

Complete and independently review this before final experiments. A committed template is not itself a registered study.

**Primary claim:** [Define an operational verification improvement under a specified action family.]

**Verification target:** [Reference-label correctness OR independently established causal hypothesis. Do not conflate them.]

**Biological prediction task and reference provenance:** [Task, inclusion criteria, independent reference method, uncertainty.]

**Interventions:** [Which parameter changes, why biology is expected stable, what controls test this assumption.]

**Sampling unit:** [Specimen/well/session/instrument. Explain repeated measurements and clustering.]

**Training/calibration/development/final-test splits:** [Commit the split manifest and hashes.]

**Costs and candidate set:** [Measured units or explicitly labeled proxies, baseline cost, hard caps, treatment of failed exposures.]

**Primary metric and coverage constraint:** [For example conditional false support at prespecified support coverage, under prespecified budgets. Zero false support through total abstention is not sufficient.]

**Baselines:** [All declared policies and independently selected fixed panel. Published algorithms require actual reproduction, not name matching.]

**Thresholds and gates:** [How development data select thresholds; allowed registration failures; minimum evidence support; uncertainty failure behavior.]

**Power and interval method:** [Use pilot group-level variance, effect size, confidence level, multiplicity handling and primary comparison.]

**Negative controls and ablations:** [Known non-informative interventions; calibration-label shuffle; no-change repeats; cost/gate removal; nuisance and temporal shifts.]

**Failure criteria:** [No advantage against strong cost/coverage-matched baselines; gains explained solely by image quality; label leakage; confounds from drift/biology; uncalibrated support.]

**Final-test freeze:** [Date, dataset hash, source commit, model/config hashes, analysis scripts, access rule.]

**Release:** [Raw-data permissions, checkpoints, scripts, negative findings, hardware calibration, manuscript claims.]
