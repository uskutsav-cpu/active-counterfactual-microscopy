from dataclasses import dataclass

import numpy as np

from .actions import AcquisitionAction
from .hypotheses import DiscretePredictiveModel


@dataclass(frozen=True, slots=True)
class Observation:
    action_name: str
    outcome: int


def update_biology_posterior(
    prior_biology: float,
    observation: Observation,
    model: DiscretePredictiveModel,
) -> float:
    if not 0.0 <= prior_biology <= 1.0:
        raise ValueError("prior_biology must be within [0, 1]")

    p_b = model.predictive(observation.action_name, "biology")
    p_n = model.predictive(observation.action_name, "nuisance")
    if not 0 <= observation.outcome < len(p_b):
        raise IndexError("observation outcome outside predictive outcome space")

    likelihood_b = float(p_b[observation.outcome])
    likelihood_n = float(p_n[observation.outcome])
    numerator = prior_biology * likelihood_b
    denominator = numerator + (1.0 - prior_biology) * likelihood_n
    if denominator == 0:
        return prior_biology
    return numerator / denominator


class SimulatedMicroscope:
    """Sample evidence outcomes from a known latent explanation."""

    def __init__(
        self,
        model: DiscretePredictiveModel,
        *,
        ground_truth: str,
        seed: int | None = None,
    ) -> None:
        if ground_truth not in {"biology", "nuisance"}:
            raise ValueError("ground_truth must be 'biology' or 'nuisance'")
        self.model = model
        self.ground_truth = ground_truth
        self.rng = np.random.default_rng(seed)

    def acquire(self, action: AcquisitionAction) -> Observation:
        p = self.model.predictive(action.name, self.ground_truth)
        outcome = int(self.rng.choice(len(p), p=p))
        return Observation(action_name=action.name, outcome=outcome)
