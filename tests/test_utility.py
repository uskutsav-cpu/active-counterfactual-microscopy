import pytest

from counterfactual_microscopy.actions import AcquisitionAction
from counterfactual_microscopy.hypotheses import DiscretePredictiveModel
from counterfactual_microscopy.policy import InformationGainPolicy
from counterfactual_microscopy.utility import expected_information_gain


def test_informative_action_has_positive_eig() -> None:
    model = DiscretePredictiveModel(
        {"a": {"biology": [0.9, 0.1], "nuisance": [0.1, 0.9]}}
    )
    eig = expected_information_gain(AcquisitionAction("a"), model, 0.5)
    assert eig > 0.5


def test_identical_predictives_have_zero_eig() -> None:
    model = DiscretePredictiveModel(
        {"a": {"biology": [0.7, 0.3], "nuisance": [0.7, 0.3]}}
    )
    eig = expected_information_gain(AcquisitionAction("a"), model, 0.5)
    assert eig == pytest.approx(0.0, abs=1e-12)


def test_policy_trades_information_against_cost() -> None:
    model = DiscretePredictiveModel(
        {
            "expensive": {"biology": [0.99, 0.01], "nuisance": [0.01, 0.99]},
            "cheap": {"biology": [0.85, 0.15], "nuisance": [0.15, 0.85]},
        }
    )
    actions = [
        AcquisitionAction("expensive", dose_cost=10.0),
        AcquisitionAction("cheap", dose_cost=0.0),
    ]
    selected = InformationGainPolicy(dose_weight=0.2).select(actions, model, 0.5)
    assert selected.action.name == "cheap"
