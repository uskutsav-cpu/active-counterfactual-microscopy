# Roadmap with exit criteria

## M0 — Formal model

**Deliverables**
- Operational definitions of `H_B` and `H_N`.
- Candidate optical intervention set.
- Dose/time cost definition.
- Preregistered verdict semantics and metrics.

**Exit criterion:** every variable used in the central claim is observable, estimable, or explicitly latent with a validation strategy.

## M1 — Simulation benchmark

**Deliverables**
- Synthetic nuisance generator.
- EIG, random, fixed, uncertainty-style policies.
- Cost-matched evaluation over shortcut strength and model misspecification.

**Exit criterion:** tests and simulation demonstrate expected behavior without relying on hand-selected successful examples.

## M2 — Retrospective replay

**Deliverables**
- Multi-condition dataset loader.
- Same-specimen indexing and registration metadata.
- Replay benchmark with hidden candidate acquisitions.

**Exit criterion:** active selection improves at least one preregistered verification metric at matched budget on held-out specimens.

## M3 — Hardware-in-the-loop

**Deliverables**
- Microscope adapter.
- Dry-run + bounds checking.
- Latency and repeatability characterization.
- Dose and registration logging.

**Exit criterion:** selected interventions can be executed reproducibly with measured cost and acceptable registration success.

## M4 — Biological validation

**Deliverables**
- Controlled shortcut benchmark.
- Blind/held-out evaluation.
- Baselines, ablations, confidence intervals.

**Exit criterion:** lower false-support or selective risk at matched dose/time, with confounds ruled out.

## M5 — Manuscript-quality release

**Deliverables**
- Frozen configs and seeds.
- Reproducible figure scripts.
- Data statement and limitations.
- External or cross-platform replication if feasible.
