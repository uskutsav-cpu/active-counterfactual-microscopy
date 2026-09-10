"""Known-latent fluorescence-like image generator; not a calibrated optical simulator."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from .io import seed_for
from .types import Intervention, Panel
from .validation import finite, integer

ACTIONS = (
    Intervention("repeat", 1.0, 0.1),
    Intervention("refocus", 1.0, 0.25),
    Intervention("normalize_gain", 1.0, 0.12),
    Intervention("dim_exposure", 0.5, 0.07),
    Intervention("flatfield", 1.0, 0.18),
)


@dataclass(frozen=True)
class SyntheticConfig:
    n: int = 400
    size: int = 48
    seed: int = 42
    correlation: float = 0.95
    signal: float = 0.28
    photon_scale: float = 500.0
    read_noise: float = 1.5
    nuisance_strength: float = 1.0
    temporal_change: float = 0.0
    unknown_nuisance: float = 0.0

    def __post_init__(self):
        integer("n", self.n, 2)
        integer("size", self.size, 16)
        integer("seed", self.seed)
        for k in ["correlation", "temporal_change", "unknown_nuisance"]:
            finite(k, getattr(self, k), 0, 1)
        for k in ["signal", "photon_scale"]:
            finite(k, getattr(self, k), 1e-06)
        for k in ["read_noise", "nuisance_strength"]:
            finite(k, getattr(self, k), 0)


def cells(count: int, size: int, rng: np.random.Generator) -> np.ndarray:
    yy, xx = np.mgrid[:size, :size]
    image = np.zeros((size, size), dtype=float)
    for _ in range(count):
        cy, cx = rng.uniform(4, size - 4, 2)
        sy, sx = rng.uniform(0.9, 1.8, 2)
        image += rng.uniform(0.7, 1.3) * np.exp(
            -0.5 * ((yy - cy) / sy) ** 2 - 0.5 * ((xx - cx) / sx) ** 2
        )
    return image


def render(
    clean: np.ndarray,
    *,
    blur: float,
    gain: float,
    background: float,
    exposure: float,
    photon_scale: float,
    read_noise: float,
    shading: float,
    rng: np.random.Generator,
) -> np.ndarray:
    for k, v in [
        ("blur", blur),
        ("gain", gain),
        ("background", background),
        ("read_noise", read_noise),
    ]:
        finite(k, v, 0)
    finite("exposure", exposure, 1e-12)
    finite("photon_scale", photon_scale, 1e-12)
    finite("shading", shading)
    light = np.maximum(
        0,
        (ndimage.gaussian_filter(clean, blur) + background)
        * (1 + shading * np.linspace(-1, 1, clean.shape[1])[None, :]),
    )
    electrons = rng.poisson(light * exposure * photon_scale).astype(float)
    electrons += rng.normal(0, read_noise, clean.shape)
    return np.clip(gain * electrons / photon_scale, 0, 1).astype(np.float32)


def generate(config: SyntheticConfig, prefix: str = "sim") -> Panel:
    rng = np.random.default_rng(config.seed)
    images = np.zeros((config.n, len(ACTIONS) + 1, config.size, config.size), np.float32)
    labels = []
    counts = []
    changed = []
    for i in range(config.n):
        label = i % 2
        count = int(rng.integers(9, 15) if label else rng.integers(3, 8))
        clean = config.signal * cells(count, config.size, rng)
        shortcut = label if rng.random() < config.correlation else 1 - label
        sign = 2 * shortcut - 1
        strength = config.nuisance_strength
        settings = {
            "blur": max(0.15, 1 - 0.65 * sign * strength),
            "gain": max(0.2, 1 + 0.35 * sign * strength),
            "background": max(0.005, 0.06 + 0.04 * sign * strength),
            "exposure": 1.0,
            "photon_scale": config.photon_scale,
            "read_noise": config.read_noise,
            "shading": 0.25 * sign * strength,
        }
        images[i, 0] = render(
            clean, **settings, rng=np.random.default_rng(seed_for(config.seed, i, "base"))
        )
        dynamic = rng.random() < config.temporal_change
        unknown = rng.random() < config.unknown_nuisance
        for j, a in enumerate(ACTIONS):
            s = dict(settings)
            if a.name == "refocus":
                s["blur"] = 0.2
            if a.name == "normalize_gain":
                s["gain"] = 1.0
            if a.name == "dim_exposure":
                s["exposure"] = 0.5
            if a.name == "flatfield":
                s.update(shading=0.0, background=0.06)
            target = np.roll(clean, 6, axis=0) * 0.6 if dynamic else clean.copy()
            if unknown:
                target[config.size // 2 - 1 : config.size // 2 + 2, :] += 0.3
            images[i, j + 1] = render(
                target, **s, rng=np.random.default_rng(seed_for(config.seed, i, j))
            )
        labels.append(label)
        counts.append(count)
        changed.append(dynamic)
    ids = [f"{prefix}_{config.seed}_{i:05d}" for i in range(config.n)]
    return Panel(
        images,
        np.array(labels),
        ids,
        ids.copy(),
        ACTIONS,
        {
            "source": "synthetic fluorescence-like generator",
            "pairing_verified": True,
            "label_source": "known simulated cell-count bins",
            "counts": counts,
            "temporal_changed": changed,
            "correlation": config.correlation,
            "dose_unit": "relative exposure proxy",
            "time_unit": "simulated seconds",
            "physical_validation": False,
        },
    )
