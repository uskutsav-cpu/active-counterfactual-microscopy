# Execution runbook

All fenced `bash` blocks below are executable Terminal commands. Run them from the repository root. Python source belongs in files; mathematical notation is not a shell command.

## 1. Activate and check

```bash
cd "$HOME/Documents/active-counterfactual-microscopy"
source .venv/bin/activate
python -m pip install -r requirements-research.txt
python -m pip install -e .
python -m ruff check src/counterfactual_microscopy/research tests/research
python -m pytest -q
```

The optional CNN test is skipped when PyTorch is absent. To enable the CNN, install `requirements-research-ml.txt`. No GPU is required by the included code, though larger experiments need independent profiling.

## 2. End-to-end smoke experiment

```bash
python -m counterfactual_microscopy.research smoke --out results/research/smoke
python -m counterfactual_microscopy.research verify-run --run results/research/smoke
open results/research/smoke/report.html
```

On non-macOS systems, open the HTML in a browser manually. The default smoke runs 48 distinct test specimens through seven policies and two acquisition-count caps: **672 episodes, not 672 independent specimens**. It is an integration demonstration, not a powered scientific study.

## 3. Larger synthetic benchmark

```bash
python -m counterfactual_microscopy.research run --config configs/research/synthetic.yaml --out results/research/synthetic
python -m counterfactual_microscopy.research verify-run --run results/research/synthetic
```

The configuration specifies specimen counts, seeds, simulator conditions, the predictor, evidence quantization, hard cost caps, gates, and thresholds. Changing them requires a fresh output directory. Do not tune a final scientific claim using a held-out test already inspected during development.

## 4. Prespecified replications and stress tests

```bash
python -m counterfactual_microscopy.research sweep --config configs/research/synthetic.yaml --grid configs/research/ablation-grid.yaml --out results/research/replications
python -m counterfactual_microscopy.research sweep --config configs/research/synthetic.yaml --grid configs/research/misspecification-grid.yaml --out results/research/misspecification
python -m counterfactual_microscopy.research sweep --config configs/research/synthetic.yaml --grid configs/research/cost-grid.yaml --out results/research/costs
```

Each grid cell has its own source/config/data locks. `sweep.json` logs failures as well as completed runs; failed cells are never silently discarded. These commands run every configured cell. Start with the smoke rather than launching several large grids simultaneously.

Additional available ablations: set `shuffle_calibration_targets: true`, `dose_weight: 0`, `time_weight: 0`, or `gates.registration_enabled: false` in a separately named config. A calibration-target shuffle is a negative control, not a viable model.

## 5. Development-calibrated decisions

```bash
python -m counterfactual_microscopy.research run --config configs/research/calibrated.yaml --out results/research/calibrated
```

Support and falsification thresholds are selected on development groups only, with a fixed corrected search grid. Insufficient evidence can disable one or both verdicts. This is an intentional abstention outcome, not a broken run. See `thresholds.json`. The procedure does not provide error guarantees for arbitrary distribution shift.

## 6. Real-image replay

Follow DATA_GUIDE to produce four NPZ panels. Then:

```bash
python -m counterfactual_microscopy.research run --config configs/research/bbbc006.yaml --out results/research/bbbc006
```

If calibration contains only one verification-target class, the engine stops rather than fabricating a conditional distribution. Redesign using development data and independent review; do not keep reseeding the final test until a favorable result appears.

## 7. Hardware boundary without hardware

```bash
python -m counterfactual_microscopy.research hardware-dry-run --out results/research/hardware-dry-run.json
```

This uses `FakeCore` only. It does not connect to a microscope, and its output explicitly says so. Physical operation requires the hardware gate and an instrument-specific calibration.

## 8. Inspect and reproduce

```bash
python -m counterfactual_microscopy.research report --run results/research/smoke
python -m counterfactual_microscopy.research verify-run --run results/research/smoke
```

Report regeneration only reads saved test rows. If a changed plotting environment produces different bytes, verification will report the difference; it does not silently reseal changed artifacts. Keep the original frozen run and make a copy for a revised rendering.

## Recovery

For an interrupted run with unchanged code, data and configuration, rerun the identical command with `--resume`. Completed checkpoints are checksum checked. For altered code or data, use a new output directory. No command resets your repository, force-pushes, deletes experimental data, or bypasses failed tests.
