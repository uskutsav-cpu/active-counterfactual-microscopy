"""Active counterfactual verification for fluorescence microscopy."""

from .actions import AcquisitionAction
from .hypotheses import DiscretePredictiveModel
from .policy import InformationGainPolicy
from .verdicts import Verdict, verdict_from_posterior

__all__ = [
    "AcquisitionAction",
    "DiscretePredictiveModel",
    "InformationGainPolicy",
    "Verdict",
    "verdict_from_posterior",
]

__version__ = "0.1.0"
