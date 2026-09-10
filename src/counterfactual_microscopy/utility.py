from __future__ import annotations

import numpy as np

from .actions import AcquisitionAction
from .hypotheses import DiscretePredictiveModel


def entropy_bits(p: np.ndarray) -> float:
    p = np.asarray(p, dtype=float)
    mask = p > 0
    return float(-np.sum(p[mask] * np.log2(p[mask])))


def expected_information_gain(
    action: AcquisitionAction,
    model: DiscretePredictiveModel,
    p_biology: float,
) -> float:
    """Compute I(H;Y|D,a) in bits for a binary hypothesis variable."""

    if not 0.0 <= p_biology <= 1.0:
        raise ValueError("p_biology must be within [0, 1]")
    model.validate_action(action.name)

    p_b = model.predictive(action.name, "biology")
    p_n = model.predictive(action.name, "nuisance")
    prior = np.array([p_biology, 1.0 - p_biology], dtype=float)

    mixture = p_biology * p_b + (1.0 - p_biology) * p_n
    conditional_entropy = p_biology * entropy_bits(p_b) + (1.0 - p_biology) * entropy_bits(p_n)
    return max(0.0, entropy_bits(mixture) - conditional_entropy)


def acquisition_utility(
    action: AcquisitionAction,
    model: DiscretePredictiveModel,
    p_biology: float,
    *,
    dose_weight: float,
    time_weight: float,
) -> tuple[float, float]:
    if dose_weight < 0 or time_weight < 0:
        raise ValueError("cost weights must be non-negative")
    eig = expected_information_gain(action, model, p_biology)
    utility = eig - dose_weight * action.dose_cost - time_weight * action.time_cost
    return eig, utility
