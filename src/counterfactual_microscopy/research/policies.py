"""Shared-budget action policies with label-free selection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .calibration import Bundle
from .types import Intervention
from .validation import finite

POLICIES = ("eig", "random", "fixed", "uncertainty", "ood", "quality", "no_reacquisition")


@dataclass(frozen=True)
class ActionScore:
    index: int
    value: float
    information_gain: float
    penalty: float


class Selector:
    def __init__(
        self,
        name: str,
        seed: int = 42,
        dose_weight: float = 0.03,
        time_weight: float = 0.05,
        fixed_order: tuple[int, ...] = (),
        stop_if_negative: bool = True,
    ):
        if name not in POLICIES:
            raise ValueError("unknown policy")
        finite("dose_weight", dose_weight, 0)
        finite("time_weight", time_weight, 0)
        self.name = name
        self.rng = np.random.default_rng(seed)
        self.dose_weight, self.time_weight = (dose_weight, time_weight)
        self.fixed_order = fixed_order
        self.stop_if_negative = stop_if_negative

    def rank(
        self,
        indices: list[int],
        actions: tuple[Intervention, ...],
        bundle: Bundle,
        context: int,
        history: dict[int, int],
        expected: np.ndarray,
    ) -> list[ActionScore]:
        scores = []
        for i in indices:
            a = actions[i]
            eig = bundle.joint.information_gain(i, context, history)
            penalty = self.dose_weight * a.dose + self.time_weight * a.seconds
            if self.name == "eig":
                value = eig - penalty
            elif self.name == "random":
                value = float(self.rng.random())
            elif self.name == "fixed":
                order = self.fixed_order or tuple(range(len(actions)))
                if set(order) != set(range(len(actions))) or len(order) != len(actions):
                    raise ValueError("fixed_order must be an action-index permutation")
                value = -float(order.index(i))
            elif self.name in {"uncertainty", "ood", "quality"}:
                value = float(expected[i, {"uncertainty": 0, "ood": 1, "quality": 2}[self.name]])
            else:
                continue
            scores.append(ActionScore(i, value, eig, penalty))
        return sorted(scores, key=lambda s: (-s.value, s.index))

    def choose(self, ranked: list[ActionScore]) -> ActionScore | None:
        if not ranked or self.name == "no_reacquisition":
            return None
        if self.name == "eig" and self.stop_if_negative and (ranked[0].value <= 0):
            return None
        return ranked[0]
