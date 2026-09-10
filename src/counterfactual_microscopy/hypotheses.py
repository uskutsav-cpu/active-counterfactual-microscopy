from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


def _normalize(values: Sequence[float]) -> np.ndarray:
    p = np.asarray(values, dtype=float)
    if p.ndim != 1 or p.size < 2:
        raise ValueError("predictive distribution must be a 1D vector with >=2 outcomes")
    if np.any(p < 0) or not np.all(np.isfinite(p)):
        raise ValueError("probabilities must be finite and non-negative")
    total = float(p.sum())
    if total <= 0:
        raise ValueError("probabilities must sum to a positive value")
    return p / total


@dataclass(slots=True)
class DiscretePredictiveModel:
    """Predictive evidence distributions under biology- vs nuisance-driven hypotheses.

    `spec[action_name]` must contain two vectors: `biology` and `nuisance`.
    The outcome index can represent any discretized verification statistic.
    """

    spec: Mapping[str, Mapping[str, Sequence[float]]]

    def predictive(self, action_name: str, hypothesis: str) -> np.ndarray:
        if action_name not in self.spec:
            raise KeyError(f"unknown action: {action_name}")
        if hypothesis not in {"biology", "nuisance"}:
            raise ValueError("hypothesis must be 'biology' or 'nuisance'")
        return _normalize(self.spec[action_name][hypothesis])

    def validate_action(self, action_name: str) -> None:
        p_b = self.predictive(action_name, "biology")
        p_n = self.predictive(action_name, "nuisance")
        if p_b.shape != p_n.shape:
            raise ValueError(f"outcome spaces differ for action {action_name!r}")
