# Completion gates: code is not a completed experiment

| Gate | Deliverable | Current package |
| --- | --- | --- |
| Software integration | Existing tests plus new tests, image smoke, preserved old modules | Implemented and locally exercised |
| Simulation | Controlled images, distinct splits, baselines, costs, traces, figures | Implemented; demonstration runs included |
| Operational target | Precise independent interpretation of H and support/falsify | Correctness and independent-annotation modes implemented; causal labeling design remains research |
| Real-image replay | Verified paired fields, reference-label provenance, held-out groups | Loaders and pipeline implemented; official full-dataset experiment not executed here |
| Strong empirical comparison | Preregistered primary metric with coverage/cost constraints, sufficient independent groups | Analysis code implemented; powered study remains to be run |
| Hardware calibration | Actual instrument response, safe envelope, latency, registration, dose | Mock-tested adapter and gate; no real hardware tested |
| Biological validation | Controlled acquisition shortcut and independent blinded reference | Not performed |
| Manuscript and release | Honest supported claims, full results, reproducible figures, independent checks | Report generation implemented; manuscript evidence not fabricated |

Progress should be measured by these deliverables, not source lines, test coverage alone, or elapsed compute. A completed computational job means its declared workflow ran; it does not force the scientific hypothesis to succeed.

The next empirical priority is an audited real-image replay with a defensible reference task and paired-field semantics, while securing instrument access/calibration in parallel. Keep any failed comparisons and all-abstention cases in the analysis. Do not move to an expensive hardware study solely because the synthetic demonstration is favorable.
