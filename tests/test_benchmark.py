from counterfactual_microscopy.actions import AcquisitionAction
from counterfactual_microscopy.benchmark import FixedPolicy, RandomPolicy
from counterfactual_microscopy.hypotheses import DiscretePredictiveModel


def make_model():
    return DiscretePredictiveModel(
        {
            "a": {
                "biology": [0.8, 0.2],
                "nuisance": [0.2, 0.8],
            },
            "b": {
                "biology": [0.6, 0.4],
                "nuisance": [0.4, 0.6],
            },
        }
    )


def test_random_policy_selects_valid_action():
    actions = [AcquisitionAction("a"), AcquisitionAction("b")]
    chosen = RandomPolicy(seed=0).select(actions, make_model(), 0.5)
    assert chosen in actions


def test_fixed_policy_selects_requested_action():
    actions = [AcquisitionAction("a"), AcquisitionAction("b")]
    chosen = FixedPolicy("b").select(actions, make_model(), 0.5)
    assert chosen.name == "b"
