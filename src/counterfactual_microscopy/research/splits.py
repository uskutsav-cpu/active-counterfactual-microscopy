"""Persisted group-safe splits; never select seeds on test performance."""

from __future__ import annotations

import hashlib
import json

import numpy as np

from .types import Panel

NAMES = ("train", "calibration", "development", "test")


def assert_disjoint(panels: dict[str, Panel]) -> None:
    groups = set()
    specimens = set()
    catalog = None
    for name, p in panels.items():
        if groups.intersection(p.group_ids) or specimens.intersection(p.specimen_ids):
            raise ValueError(f"group/specimen leakage at {name}")
        groups.update(p.group_ids)
        specimens.update(p.specimen_ids)
        current = [(a.name, a.dose, a.seconds, a.parameters) for a in p.actions]
        if catalog is not None and catalog != current:
            raise ValueError("all splits must share an acquisition catalog")
        catalog = current


def indices(
    group_ids: list[str], seed: int = 42, fractions: tuple[float, ...] = (0.4, 0.25, 0.15, 0.2)
) -> dict[str, np.ndarray]:
    f = np.asarray(fractions, dtype=float)
    if (
        f.shape != (4,)
        or not np.isfinite(f).all()
        or np.any(f <= 0)
        or (not np.isclose(f.sum(), 1))
    ):
        raise ValueError("four positive fractions must sum to one")
    groups = np.array(sorted(set(group_ids)))
    if len(groups) < 8:
        raise ValueError("at least eight independent groups required")
    groups = groups[np.random.default_rng(seed).permutation(len(groups))]
    counts = np.floor(len(groups) * f).astype(int)
    counts[-1] = len(groups) - counts[:-1].sum()
    if np.any(counts < 1):
        raise ValueError("empty split")
    return {
        n: np.flatnonzero(np.isin(group_ids, g))
        for n, g in zip(NAMES, np.split(groups, np.cumsum(counts)[:-1]))
    }


def record(panels: dict[str, Panel]) -> dict:
    data = {
        n: {"specimens": p.specimen_ids, "groups": sorted(set(p.group_ids))}
        for n, p in panels.items()
    }
    return {
        "splits": data,
        "sha256": hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest(),
    }
