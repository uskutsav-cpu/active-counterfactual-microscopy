"""Strict validated experiment configs; unknown keys fail rather than being ignored."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

from .decisions import Thresholds
from .policies import POLICIES
from .replay import Gates
from .validation import finite, integer


@dataclass
class Config:
    seed: int = 42
    source: str = "synthetic"
    panels: dict[str, str] = field(default_factory=dict)
    model: str = "logistic"
    cnn_epochs: int = 12
    target: str = "correctness"
    train_n: int = 300
    calibration_n: int = 160
    development_n: int = 100
    test_n: int = 120
    image_size: int = 48
    train_correlation: float = 0.95
    calibration_correlation: float = 0.5
    development_correlation: float = 0.5
    test_correlation: float = 0.5
    nuisance_strength: float = 1.0
    signal: float = 0.28
    photon_scale: float = 500.0
    read_noise: float = 1.5
    test_temporal_change: float = 0.0
    test_unknown_nuisance: float = 0.0
    n_states: int = 5
    likelihood_alpha: float = 2.0
    min_context: int = 12
    ood_gate_multiplier: float = 3.0
    policies: list[str] = field(default_factory=lambda: list(POLICIES))
    budgets: list[int] = field(default_factory=lambda: [1, 2])
    max_dose: float = 3.0
    max_seconds: float = 3.0
    dose_weight: float = 0.03
    time_weight: float = 0.05
    stop_if_negative: bool = True
    early_stop: bool = False
    tune_thresholds: bool = False
    support_threshold: float | None = 0.9
    falsify_threshold: float | None = 0.1
    max_development_group_error: float = 0.15
    min_development_groups: int = 10
    bootstrap_repeats: int = 100
    gates: dict = field(default_factory=dict)
    fixed_order: list[int] = field(default_factory=list)
    shuffle_calibration_targets: bool = False
    save_models: bool = True

    def validate(self) -> Config:
        integer("seed", self.seed)
        for name in [
            "cnn_epochs",
            "train_n",
            "calibration_n",
            "development_n",
            "test_n",
            "min_context",
            "min_development_groups",
            "bootstrap_repeats",
        ]:
            integer(name, getattr(self, name), 1)
        integer("image_size", self.image_size, 16)
        integer("n_states", self.n_states, 2)
        if self.source not in ["synthetic", "panels"]:
            raise ValueError("source must be synthetic or panels")
        if self.source == "panels" and set(self.panels) != {
            "train",
            "calibration",
            "development",
            "test",
        }:
            raise ValueError("panels source requires train/calibration/development/test paths")
        if self.model not in ["logistic", "forest", "cnn"]:
            raise ValueError("unknown predictor")
        if self.target not in ["correctness", "annotated_hypothesis"]:
            raise ValueError("unknown target semantics")
        for name in [
            "train_correlation",
            "calibration_correlation",
            "development_correlation",
            "test_correlation",
            "test_temporal_change",
            "test_unknown_nuisance",
            "max_development_group_error",
        ]:
            finite(name, getattr(self, name), 0, 1)
        for name in [
            "max_dose",
            "max_seconds",
            "dose_weight",
            "time_weight",
            "read_noise",
            "nuisance_strength",
        ]:
            finite(name, getattr(self, name), 0)
        for name in ["signal", "photon_scale", "likelihood_alpha"]:
            finite(name, getattr(self, name), 1e-12)
        finite("ood_gate_multiplier", self.ood_gate_multiplier, 1)
        if (
            not self.policies
            or not set(self.policies).issubset(POLICIES)
            or len(set(self.policies)) != len(self.policies)
        ):
            raise ValueError("policies must be a nonempty unique supported set")
        if not self.budgets or len(set(self.budgets)) != len(self.budgets):
            raise ValueError("budgets must be nonempty/unique")
        for b in self.budgets:
            integer("budget", b)
        for name in [
            "stop_if_negative",
            "early_stop",
            "tune_thresholds",
            "shuffle_calibration_targets",
            "save_models",
        ]:
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be boolean")
        if self.tune_thresholds and self.early_stop:
            raise ValueError(
                "development-tuned thresholds currently require early_stop=false to match trajectory distributions"
            )
        Gates(**self.gates)
        Thresholds(self.support_threshold, self.falsify_threshold)
        return self

    def to_dict(self) -> dict:
        return asdict(self)


def load(path: str | Path) -> Config:
    path = Path(path)
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise ValueError("configuration must be a mapping")
    try:
        cfg = Config(**data)
    except TypeError as exc:
        raise ValueError(f"unknown/invalid configuration keys: {exc}") from exc
    cfg.validate()
    if cfg.source == "panels":
        cfg.panels = {
            n: str((path.resolve().parent / p).resolve()) if not Path(p).is_absolute() else p
            for n, p in cfg.panels.items()
        }
    return cfg


def smoke() -> Config:
    return Config(
        train_n=140,
        calibration_n=100,
        development_n=40,
        test_n=48,
        image_size=32,
        budgets=[1, 2],
        bootstrap_repeats=20,
        n_states=4,
        gates={"min_history_support": 2},
        save_models=True,
    ).validate()
