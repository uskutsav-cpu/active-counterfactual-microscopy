"""Typed acquisition records. Costs are proxies until independently calibrated."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .validation import binary, finite, integer, numeric_array


@dataclass(frozen=True)
class Intervention:
    name: str
    dose: float
    seconds: float
    parameters: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name or not self.name.replace("_", "").replace("-", "").isalnum():
            raise ValueError("invalid action name")
        finite("dose", self.dose, 0)
        finite("seconds", self.seconds, 0)
        if self.dose == 0 and self.seconds == 0:
            raise ValueError("a reacquisition cannot have both costs zero")
        for k, v in self.parameters.items():
            finite(k, v)


@dataclass
class Budget:
    max_steps: int
    max_dose: float
    max_seconds: float
    steps: int = 0
    dose: float = 0.0
    seconds: float = 0.0

    def __post_init__(self) -> None:
        integer("max_steps", self.max_steps)
        integer("steps", self.steps)
        for k in ["max_dose", "max_seconds", "dose", "seconds"]:
            finite(k, getattr(self, k), 0)
        if (
            self.steps > self.max_steps
            or self.dose > self.max_dose
            or self.seconds > self.max_seconds
        ):
            raise ValueError("initial consumption exceeds budget")

    def permits(self, action: Intervention) -> bool:
        return (
            self.steps < self.max_steps
            and self.dose + action.dose <= self.max_dose + 1e-12
            and (self.seconds + action.seconds <= self.max_seconds + 1e-12)
        )

    def charge(self, action: Intervention) -> None:
        """Reserve BEFORE execution. Failed exposure attempts are never refunded."""
        if not self.permits(action):
            raise ValueError("hard acquisition budget exceeded")
        self.steps += 1
        self.dose += action.dose
        self.seconds += action.seconds


@dataclass
class Panel:
    """Static same-field response panel, baseline first. Labels stay outside inference.

    `labels` are binary biological-task labels, NOT shortcut labels. `reference`
    optionally supplies separately annotated hypothesis labels with provenance.
    """

    images: np.ndarray
    labels: np.ndarray
    specimen_ids: list[str]
    group_ids: list[str]
    actions: tuple[Intervention, ...]
    provenance: dict[str, Any]
    reference: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.images = numeric_array("images", self.images, 4)
        self.labels = binary("labels", self.labels)
        n, v, h, w = self.images.shape
        if min(h, w) < 8 or v != len(self.actions) + 1 or (not self.actions):
            raise ValueError("panel needs baseline, actions, and images at least 8x8")
        if any(len(x) != n for x in [self.labels, self.specimen_ids, self.group_ids]):
            raise ValueError("panel metadata length mismatch")
        if len(set(self.specimen_ids)) != n or any(
            not x for x in self.specimen_ids + self.group_ids
        ):
            raise ValueError("specimen IDs must be unique; all IDs must be nonempty")
        if len({a.name for a in self.actions}) != len(self.actions):
            raise ValueError("duplicate action names")
        if self.reference is not None:
            self.reference = binary("reference", self.reference)
            if len(self.reference) != n or not self.provenance.get("reference_source"):
                raise ValueError("reference annotations need matching length and reference_source")

    def subset(self, indices: list[int] | np.ndarray) -> Panel:
        ix = np.asarray(indices)
        if ix.ndim != 1 or not len(ix) or ix.dtype.kind not in "iu":
            raise ValueError("subset indices must be a nonempty integer vector")
        return Panel(
            self.images[ix],
            self.labels[ix],
            [self.specimen_ids[i] for i in ix],
            [self.group_ids[i] for i in ix],
            self.actions,
            dict(self.provenance),
            None if self.reference is None else self.reference[ix],
        )

    def target(self, predictions: np.ndarray, kind: str) -> np.ndarray:
        if kind == "correctness":
            predictions = binary("predictions", predictions)
            if predictions.shape != self.labels.shape:
                raise ValueError("predictions shape mismatch")
            return (predictions == self.labels).astype(int)
        if kind == "annotated_hypothesis" and self.reference is not None:
            return self.reference.copy()
        raise ValueError("annotated_hypothesis requires independent reference labels")
