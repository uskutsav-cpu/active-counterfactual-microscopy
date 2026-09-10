from counterfactual_microscopy.hypotheses import DiscretePredictiveModel
from counterfactual_microscopy.simulator import Observation, update_biology_posterior


def test_biology_like_observation_increases_biology_posterior() -> None:
    model = DiscretePredictiveModel({"a": {"biology": [0.9, 0.1], "nuisance": [0.2, 0.8]}})
    posterior = update_biology_posterior(0.5, Observation("a", 0), model)
    assert posterior > 0.5


def test_nuisance_like_observation_decreases_biology_posterior() -> None:
    model = DiscretePredictiveModel({"a": {"biology": [0.9, 0.1], "nuisance": [0.2, 0.8]}})
    posterior = update_biology_posterior(0.5, Observation("a", 1), model)
    assert posterior < 0.5
