# Source verification

Checked 2026-09-10. These sources ground dataset metadata and APIs, not a novelty claim or endorsement.

- Broad BBBC005: https://bbbc.broadinstitute.org/BBBC005 — synthetic image origin, count/blur filename metadata, size and download link.
- Broad BBBC006: https://bbbc.broadinstitute.org/BBBC006 — paired field acquisition, focal planes, TIFF encoding and algorithm-generated count reference. The displayed count ZIP label links to an actual CSV.
- Pycro-Manager Core documentation: https://pycro-manager.readthedocs.io/en/latest/core.html — Java/Python backend differences, method naming, snap/tagged-image interface.
- Scikit-learn common pitfalls: https://scikit-learn.org/stable/common_pitfalls.html — separating fitting from held-out evaluation and avoiding preprocessing leakage.
- Scikit-learn model persistence: https://scikit-learn.org/stable/model_persistence.html — joblib/pickle execution risk and environment compatibility.

Locally generated SHA-256 files record integrity/provenance; they are not publisher signatures. Nothing in these sources establishes that a correct or invariant prediction must be biology-driven.
