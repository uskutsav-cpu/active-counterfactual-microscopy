# Active Counterfactual Verification for Fluorescence Microscopy

> **Research question:** Can a microscopy system actively reacquire the *same specimen* under a low-cost optical perturbation to test whether an AI prediction is biology-driven or depends on acquisition-specific nuisance information?

[![CI](https://github.com/uskutsav-cpu/active-counterfactual-microscopy/actions/workflows/ci.yml/badge.svg)](https://github.com/uskutsav-cpu/active-counterfactual-microscopy/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-research%20prototype-orange)](#project-status)

## Why this project exists

Most robustness checks in computational microscopy are passive: uncertainty estimation, out-of-distribution detection, fixed corruption tests, calibration, or retrospective stress tests. This project studies an **active verification** alternative.

After an AI system produces a biological prediction, the microscope chooses a targeted reacquisition—e.g. illumination intensity, exposure, focus, detector gain, or another optical condition—not merely to improve image quality, but to discriminate between two explanations:

- **H_B (biology-driven):** the prediction is supported by specimen biology and should remain appropriately stable under nuisance-changing reacquisition.
- **H_N (nuisance-driven):** the prediction relies materially on acquisition-specific shortcuts and should change in a way better explained by imaging nuisance.

The next acquisition is selected by expected information gain, with explicit penalties for photon dose and acquisition time:

\[
 a^* = \arg\max_a I(H;Y_a\mid D) - \lambda_d C_{dose}(a) - \lambda_t C_{time}(a)
\]

The verification layer then returns **SUPPORTED**, **FALSIFIED**, or **ABSTAIN**.

## Core contribution being tested

The intended scientific contribution is **not** “take another image when uncertain.” The stronger claim is:

> Choose the *counterfactual optical intervention* that most efficiently separates a biology-driven explanation from an acquisition-shortcut explanation, under experimental cost constraints, and use the resulting evidence to verify or reject the original AI prediction.

This repository is designed to make that claim falsifiable.

## System overview

```mermaid
flowchart LR
    X[Initial fluorescence image] --> M[Biological prediction model]
    M --> P[Prediction + latent evidence]
    P --> H[Competing explanations\nH_B vs H_N]
    H --> A[Candidate optical interventions]
    A --> U[Expected information gain\n- dose penalty - time penalty]
    U --> R[Reacquire same specimen]
    R --> B[Bayesian evidence update]
    B --> V{Verification verdict}
    V -->|high P(H_B)| S[SUPPORTED]
    V -->|low P(H_B)| F[FALSIFIED]
    V -->|insufficient evidence| Z[ABSTAIN]
```

## Repository layout

```text
.
├── configs/                         # Reproducible experiment configurations
├── docs/
│   ├── EXPERIMENT_PROTOCOL.md       # Benchmark protocol and statistical plan
│   ├── HARDWARE_VALIDATION.md       # Microscope integration + calibration checklist
│   ├── METHODS.md                   # Mathematical formulation
│   ├── PROJECT_BRIEF.md             # One-page mentor/collaborator brief
│   ├── RESEARCH_QUESTIONS.md        # Claims, falsifiers, and ablations
│   └── ROADMAP.md                   # Milestone-based execution plan
├── scripts/
│   └── demo_simulation.py           # End-to-end simulated verification run
├── src/counterfactual_microscopy/
│   ├── actions.py                   # Optical intervention representation
│   ├── benchmark.py                 # Policy comparison harness
│   ├── hypotheses.py                # Competing predictive explanations
│   ├── metrics.py                   # Verification metrics
│   ├── policy.py                    # Information-gain acquisition policy
│   ├── simulator.py                 # Closed-loop simulated microscope
│   ├── utility.py                   # EIG + experimental-cost objective
│   ├── verdicts.py                  # Supported / falsified / abstain logic
│   └── hardware/                    # Adapter boundary for real microscopes
└── tests/                           # Unit tests for scientific core
```

## Quick start

```bash
git clone https://github.com/uskutsav-cpu/active-counterfactual-microscopy.git
cd active-counterfactual-microscopy
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pytest -q
python scripts/demo_simulation.py
```

Expected demo behavior: the information-gain policy should prefer a low-cost acquisition that produces the largest separation between the biology and nuisance hypotheses, then update the posterior and emit a verification verdict.

## Minimal API example

```python
from counterfactual_microscopy.actions import AcquisitionAction
from counterfactual_microscopy.hypotheses import DiscretePredictiveModel
from counterfactual_microscopy.policy import InformationGainPolicy

model = DiscretePredictiveModel(
    {
        "lower_gain": {
            "biology": [0.80, 0.15, 0.05],
            "nuisance": [0.20, 0.30, 0.50],
        },
        "refocus": {
            "biology": [0.55, 0.30, 0.15],
            "nuisance": [0.45, 0.35, 0.20],
        },
    }
)

actions = [
    AcquisitionAction("lower_gain", dose_cost=0.05, time_cost=0.10),
    AcquisitionAction("refocus", dose_cost=0.02, time_cost=0.12),
]

policy = InformationGainPolicy(dose_weight=0.2, time_weight=0.1)
choice = policy.select(actions, model, p_biology=0.5)
print(choice.action.name, choice.information_gain, choice.utility)
```

## Experimental benchmark

The main benchmark should compare the proposed policy against:

| Policy | Acquisition choice | Uses competing causal explanations? | Cost-aware? |
|---|---|---:|---:|
| **Counterfactual EIG (ours)** | Max information separating H_B vs H_N | Yes | Yes |
| Uncertainty-guided | Max predictive/model uncertainty | No | Optional |
| OOD-guided | Max embedding/distribution shift signal | No | Optional |
| Fixed robustness panel | Predefined perturbation sequence | No | Usually no |
| Random reacquisition | Random valid perturbation | No | No |
| No reacquisition | Single-pass prediction | No | N/A |

Primary endpoints should include verification AUROC/AUPRC where labels are available, selective risk at fixed coverage, falsification recall, false-support rate, abstention rate, photons per resolved case, seconds per resolved case, and information gained per unit dose.

See [`docs/EXPERIMENT_PROTOCOL.md`](docs/EXPERIMENT_PROTOCOL.md) for the full plan.

## Hardware boundary

The scientific policy is intentionally decoupled from microscope control. Hardware-specific code implements a small adapter interface:

```python
class MicroscopeAdapter(Protocol):
    def acquire(self, action: AcquisitionAction) -> Observation: ...
    def estimate_cost(self, action: AcquisitionAction) -> AcquisitionCost: ...
    def validate_action(self, action: AcquisitionAction) -> None: ...
```

This makes it possible to begin with a simulator or prerecorded multi-condition image stacks, then replace the mock adapter with Micro-Manager, vendor SDK, or custom DAQ control without rewriting the inference layer.

## Key scientific safeguards

1. **Same-specimen verification.** Reacquisition must preserve specimen identity and register measurements sufficiently well to interpret changes.
2. **Intervention semantics.** Optical perturbations must alter nuisance variables without intentionally changing biological state over the verification interval.
3. **Dose accounting.** Photon exposure is a first-class cost, not an afterthought.
4. **Temporal confounding controls.** Drift, bleaching, motion, physiology, and phototoxicity can mimic “falsification” and must be modeled or bounded.
5. **Predeclared abstention.** The system should be allowed to return *abstain* rather than force a biological conclusion.
6. **No circular validation.** The same features used to select an intervention should not be treated as independent ground truth for verification.

## Project status

**Current stage:** research prototype / methods development.

The repository contains a functional discrete-evidence simulator and cost-aware information-gain policy. It does **not** claim biological validation yet. Real microscopy results should only be added after acquisition calibration, preregistered decision thresholds, and controlled same-specimen experiments.

## Roadmap

- **M0 — Formalization:** define H_B/H_N, intervention family, cost model, and falsifiable claims.
- **M1 — Simulation:** prove the acquisition policy behaves correctly under synthetic nuisance mechanisms.
- **M2 — Retrospective replay:** use multi-condition fluorescence datasets as if conditions were acquired sequentially.
- **M3 — Hardware-in-the-loop:** integrate microscope control and validate latency, registration, dose, and repeatability.
- **M4 — Biological validation:** test whether active verification reduces false-support errors under real acquisition shortcuts.
- **M5 — Manuscript:** locked baselines, ablations, preregistered analysis, reproducible figures, external replication where possible.

Detailed exit criteria are in [`docs/ROADMAP.md`](docs/ROADMAP.md).

## Reproducibility philosophy

Every experiment should record:

- specimen/sample ID and acquisition timestamp;
- initial prediction and confidence;
- candidate interventions considered;
- predicted EIG and cost terms for every candidate;
- selected intervention and exact hardware parameters;
- raw reacquisition plus registration transform;
- posterior update and final verdict;
- total photon-dose proxy and elapsed acquisition time;
- software commit SHA and config file.

## Citation

If this work becomes citable, update [`CITATION.cff`](CITATION.cff) with the manuscript DOI/version. Until then, cite the repository commit SHA used for a result.

## License

Code is released under the MIT License. Imaging datasets and microscope SDKs may have separate licenses and are not automatically covered by this repository license.
