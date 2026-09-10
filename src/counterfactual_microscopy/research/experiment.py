"""One-command, resumable computational experiment with frozen config and source hash."""

from __future__ import annotations

import hashlib
import itertools
import json
import traceback
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .calibration import JointEvidence, fit_calibration
from .config import Config
from .decisions import Thresholds, tune_thresholds
from .io import atomic_json, digest, environment, load_panel, source_digest, write_csv
from .models import FeatureReference, make_predictor
from .replay import Gates, evaluate
from .reporting import render_report
from .splits import NAMES, assert_disjoint, record
from .statistics import paired_bootstrap, summary
from .synthetic import SyntheticConfig, generate


def make_panels(cfg: Config) -> dict:
    if cfg.source == "panels":
        return {n: load_panel(cfg.panels[n]) for n in NAMES}
    panels = {}
    for i, n in enumerate(NAMES):
        panels[n] = generate(
            SyntheticConfig(
                n=getattr(cfg, n + "_n"),
                size=cfg.image_size,
                seed=cfg.seed + 1000 * i,
                correlation=getattr(cfg, n + "_correlation"),
                signal=cfg.signal,
                photon_scale=cfg.photon_scale,
                read_noise=cfg.read_noise,
                nuisance_strength=cfg.nuisance_strength,
                temporal_change=cfg.test_temporal_change if n == "test" else 0.0,
                unknown_nuisance=cfg.test_unknown_nuisance if n == "test" else 0.0,
            ),
            prefix=n,
        )
    return panels


def _data_hash(panel) -> str:
    h = hashlib.sha256()
    for a in [panel.images, panel.labels]:
        h.update(str(a.shape).encode())
        h.update(str(a.dtype).encode())
        h.update(np.ascontiguousarray(a).data)
    if panel.reference is not None:
        h.update(np.ascontiguousarray(panel.reference).data)
    h.update(json.dumps(panel.provenance, sort_keys=True).encode())
    return h.hexdigest()


def run(cfg: Config, out: str | Path, resume: bool = False) -> dict:
    cfg.validate()
    out = Path(out)
    source = source_digest()
    lock = {"config": cfg.to_dict(), "source_sha256": source}
    if out.exists():
        if not resume:
            raise FileExistsError("run directory exists; use --resume or a fresh output directory")
        if (
            not (out / "config.lock.json").exists()
            or json.loads((out / "config.lock.json").read_text()) != lock
        ):
            raise ValueError("resume refused: config or research source changed")
        if (out / "status.json").exists() and json.loads((out / "status.json").read_text()).get(
            "state"
        ) == "COMPLETE":
            receipts = json.loads((out / "artifact_hashes.json").read_text())
            damaged = [
                p for p, h in receipts.items() if not (out / p).is_file() or digest(out / p) != h
            ]
            if damaged:
                raise ValueError("completed-run artifact checksum mismatch: " + ", ".join(damaged))
            return {"state": "COMPLETE", "reused_complete_run": True, "out": str(out)}
    out.mkdir(parents=True, exist_ok=True)
    atomic_json(out / "config.lock.json", lock)
    atomic_json(out / "environment.json", environment())

    def status(stage, **extra):
        atomic_json(out / "status.json", {"state": "RUNNING", "stage": stage, **extra})

    try:
        status("prepare_data")
        print("Preparing data and checking group separation...", flush=True)
        panels = make_panels(cfg)
        assert_disjoint(panels)
        for n, p in panels.items():
            if not p.provenance.get("pairing_verified"):
                raise ValueError(f"{n}: pairing not verified")
        data = {
            "splits": record(panels),
            "sha256": {n: _data_hash(p) for n, p in panels.items()},
            "provenance": {n: p.provenance for n, p in panels.items()},
        }
        if (
            resume
            and (out / "data.lock.json").exists()
            and (json.loads((out / "data.lock.json").read_text()) != data)
        ):
            raise ValueError("resume refused: dataset contents changed")
        atomic_json(out / "data.lock.json", data)
        status("train")
        print("Training predictor on training images only...", flush=True)
        predictor = make_predictor(cfg.model, cfg.seed, cfg.cnn_epochs).fit(
            panels["train"].images[:, 0], panels["train"].labels
        )
        reference = FeatureReference().fit(panels["train"].images[:, 0], cfg.ood_gate_multiplier)
        status("calibrate")
        print("Fitting calibration-only evidence model...", flush=True)
        bundle = fit_calibration(
            panels["calibration"],
            predictor,
            reference,
            target_kind=cfg.target,
            n_states=cfg.n_states,
            seed=cfg.seed,
            alpha=cfg.likelihood_alpha,
            min_context=cfg.min_context,
        )
        if cfg.shuffle_calibration_targets:
            y = bundle.joint.y.copy()
            np.random.default_rng(cfg.seed + 900).shuffle(y)
            bundle.joint = JointEvidence(
                bundle.joint.x,
                y,
                bundle.joint.contexts,
                cfg.n_states,
                alpha=cfg.likelihood_alpha,
                min_context=cfg.min_context,
            )
            bundle.report["ablation"] = "shuffled calibration targets"
        atomic_json(out / "calibration.json", bundle.report)
        if cfg.save_models:
            import joblib

            joblib.dump(
                {"predictor": predictor, "bundle": bundle, "actions": panels["train"].actions},
                out / "trusted_model.joblib",
                compress=3,
            )
            atomic_json(
                out / "model_receipt.json",
                {
                    "sha256": digest(out / "trusted_model.joblib"),
                    "warning": "joblib can execute code; load only your own trusted artifacts in the same environment",
                },
            )
        gates = Gates(**cfg.gates)
        all_rows = []
        threshold_reports = {}
        receipts = {}
        if resume and (out / "checkpoint_receipts.json").exists():
            receipts = json.loads((out / "checkpoint_receipts.json").read_text())
        for budget in cfg.budgets:
            for policy in cfg.policies:
                key = f"{policy}_b{budget}"
                checkpoint = out / "checkpoints" / f"{key}.json"
                if resume and checkpoint.exists():
                    if receipts.get(key) != digest(checkpoint):
                        raise ValueError(f"checkpoint hash mismatch: {key}")
                    saved = json.loads(checkpoint.read_text())
                    all_rows.extend(saved["rows"])
                    threshold_reports[key] = saved["threshold_report"]
                    continue
                status("evaluate", policy=policy, budget=budget)
                print(f"Evaluating {policy}, acquisition cap {budget}...", flush=True)
                shared = {
                    "policy": policy,
                    "max_steps": budget,
                    "max_dose": cfg.max_dose,
                    "max_seconds": cfg.max_seconds,
                    "gates": gates,
                    "seed": cfg.seed,
                    "dose_weight": cfg.dose_weight,
                    "time_weight": cfg.time_weight,
                    "early_stop": cfg.early_stop,
                    "stop_if_negative": cfg.stop_if_negative,
                    "fixed_order": tuple(cfg.fixed_order),
                }
                threshold = Thresholds(cfg.support_threshold, cfg.falsify_threshold)
                threshold_report = {"mode": "prespecified", "thresholds": asdict(threshold)}
                if cfg.tune_thresholds:
                    dev = evaluate(
                        panels["development"],
                        predictor,
                        bundle,
                        thresholds=Thresholds(None, None),
                        **shared,
                    )
                    threshold, threshold_report = tune_thresholds(
                        dev, cfg.max_development_group_error, cfg.min_development_groups
                    )
                rows = evaluate(panels["test"], predictor, bundle, thresholds=threshold, **shared)
                atomic_json(checkpoint, {"rows": rows, "threshold_report": threshold_report})
                receipts[key] = digest(checkpoint)
                atomic_json(out / "checkpoint_receipts.json", receipts)
                all_rows.extend(rows)
                threshold_reports[key] = threshold_report
        atomic_json(out / "thresholds.json", threshold_reports)
        atomic_json(out / "test_episodes.json", all_rows)
        write_csv(
            out / "episodes.csv", [{k: v for k, v in r.items() if k != "trace"} for r in all_rows]
        )
        atomic_json(out / "summary.json", summary(all_rows))
        status("bootstrap")
        comparisons = []
        for budget in cfg.budgets:
            eig = [r for r in all_rows if r["policy"] == "eig" and r["budget_steps"] == budget]
            if not eig:
                continue
            for policy in cfg.policies:
                if policy == "eig":
                    continue
                other = [
                    r for r in all_rows if r["policy"] == policy and r["budget_steps"] == budget
                ]
                comparison = paired_bootstrap(
                    eig, other, "false_support_conditional", cfg.bootstrap_repeats, cfg.seed
                )
                comparisons.append(
                    {"first": "eig", "second": policy, "budget_steps": budget, **comparison}
                )
        atomic_json(out / "paired_comparisons.json", comparisons)
        status("report")
        report = render_report(out)
        artifacts = {
            str(p.relative_to(out)): digest(p)
            for p in out.rglob("*")
            if p.is_file() and p.name not in ["artifact_hashes.json", "status.json"]
        }
        atomic_json(out / "artifact_hashes.json", artifacts)
        final = {
            "state": "COMPLETE",
            "stage": "computational_experiment",
            "biological_validation": False,
            "hardware_validation": False,
            "episodes": len(all_rows),
            "out": str(out),
            **report,
        }
        atomic_json(out / "status.json", final)
        print(f"Completed computational experiment: {out}", flush=True)
        return final
    except BaseException as exc:
        atomic_json(
            out / "status.json",
            {
                "state": "FAILED",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            },
        )
        raise


def sweep(cfg: Config, grid: dict[str, list], out: str | Path, resume: bool = False) -> dict:
    """Prespecified grid, separate run per cell. Never selects a winner for you."""
    if not grid or any(not isinstance(v, list) or not v for v in grid.values()):
        raise ValueError("grid must contain nonempty lists")
    fields = set(cfg.to_dict())
    if not set(grid).issubset(fields):
        raise ValueError("unknown sweep parameter")
    root = Path(out)
    root.mkdir(parents=True, exist_ok=True)
    outcomes = []
    keys = sorted(grid)
    for values in itertools.product(*(grid[k] for k in keys)):
        updates = dict(zip(keys, values))
        data = cfg.to_dict()
        data.update(updates)
        run_id = hashlib.sha256(json.dumps(updates, sort_keys=True).encode()).hexdigest()[:12]
        path = root / run_id
        try:
            result = run(Config(**data), path, resume=resume)
            outcomes.append({"parameters": updates, "run": str(path), "state": result["state"]})
        except Exception as exc:
            outcomes.append(
                {"parameters": updates, "run": str(path), "state": "FAILED", "error": str(exc)}
            )
        atomic_json(
            root / "sweep.json", {"outcomes": outcomes, "grid": grid, "selection_performed": False}
        )
    return {
        "runs": len(outcomes),
        "failed": sum(x["state"] == "FAILED" for x in outcomes),
        "out": str(root),
    }
