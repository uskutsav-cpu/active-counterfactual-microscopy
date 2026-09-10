from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from .actions import AcquisitionAction
from .hypotheses import DiscretePredictiveModel
from .policy import InformationGainPolicy


@dataclass(slots=True)
class RandomPolicy:
    seed: int | None = None
    _rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)

    def select(
        self,
        actions: Sequence[AcquisitionAction],
        model: DiscretePredictiveModel,
        p_biology: float,
    ) -> AcquisitionAction:
        del model, p_biology
        if not actions:
            raise ValueError("at least one candidate action is required")
        return actions[int(self._rng.integers(0, len(actions)))]


@dataclass(slots=True)
class FixedPolicy:
    action_name: str

    def select(
        self,
        actions: Sequence[AcquisitionAction],
        model: DiscretePredictiveModel,
        p_biology: float,
    ) -> AcquisitionAction:
        del model, p_biology
        for action in actions:
            if action.name == self.action_name:
                return action
        raise KeyError(f"fixed action {self.action_name!r} not found")


__all__ = ["FixedPolicy", "InformationGainPolicy", "RandomPolicy"]
