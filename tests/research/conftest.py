import pytest

from counterfactual_microscopy.research.calibration import fit_calibration
from counterfactual_microscopy.research.models import FeaturePredictor, FeatureReference
from counterfactual_microscopy.research.synthetic import SyntheticConfig, generate


@pytest.fixture(scope="session")
def research_system():
    train = generate(SyntheticConfig(n=100, size=24, seed=10, correlation=0.98), "train")
    calibration = generate(SyntheticConfig(n=100, size=24, seed=11, correlation=0.5), "cal")
    test = generate(SyntheticConfig(n=20, size=24, seed=12, correlation=0.5), "test")
    predictor = FeaturePredictor("logistic", 10).fit(train.images[:, 0], train.labels)
    reference = FeatureReference().fit(train.images[:, 0])
    bundle = fit_calibration(calibration, predictor, reference, n_states=3, seed=10)
    return (train, calibration, test, predictor, bundle)
