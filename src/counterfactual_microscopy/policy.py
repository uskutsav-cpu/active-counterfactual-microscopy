from dataclasses import dataclass
from typing import Sequence

from .actions import AcquisitionAction
from .hypotheses import DiscretePredictiveModel
from .utility import acquisition_utility


@dataclass(frozen=True, slots=True)
class ActionScore:
    action: AcquisitionAction
    information_gain: float
    utility: float


@dataclass(slots=True)
class InformationGainPolicy:
    dose_weight: float = 0.0
    time_weight: float = 0.0

    def rank(
        self,
        actions: Sequence[AcquisitionAction],
        model: DiscretePredictiveModel,
        p_biology: float,
    ) -> list[ActionScore]:
        if not actions:
            raise ValueError("at least one candidate action is required")

        scores = []
        for action in actions:
            eig, utility = acquisition_utility(
                action,
                model,
                p_biology,
                dose_weight=self.dose_weight,
                time_weight=self.time_weight,
            )
            scores.append(ActionScore(action=action, information_gain=eig, utility=utility))

        return sorted(scores, key=lambda s: (s.utility, s.information_gain), reverse=True)

    def select(
        self,
        actions: Sequence[AcquisitionAction],
        model: DiscretePredictiveModel,
        p_biology: float,
    ) -> ActionScore:
        return self.rank(actions, model, p_biology)[0]
