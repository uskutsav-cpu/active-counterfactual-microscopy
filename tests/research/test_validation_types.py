from dataclasses import replace

import numpy as np
import pytest

from counterfactual_microscopy.research.config import Config, load
from counterfactual_microscopy.research.decisions import Thresholds, binomial_upper
from counterfactual_microscopy.research.synthetic import SyntheticConfig
from counterfactual_microscopy.research.types import Budget, Intervention
from counterfactual_microscopy.research.validation import binary, finite, integer


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), True, None, "bad"])
def test_nonfinite_or_invalid_rejected(value):
    with pytest.raises(ValueError):
        finite("x", value)


@pytest.mark.parametrize("value", [1.2, True, -1, "2", None])
def test_integer_rejected(value):
    with pytest.raises(ValueError):
        integer("x", value)


@pytest.mark.parametrize("value", [[0, 2], [float("nan"), 1], [], [[0, 1]]])
def test_binary_rejected(value):
    with pytest.raises(ValueError):
        binary("y", value)


def test_valid_numbers():
    assert finite("x", "1.25") == 1.25
    assert integer("x", np.int64(2)) == 2
    np.testing.assert_array_equal(binary("y", [True, False]), [1, 0])


@pytest.mark.parametrize("name", ["", "bad/action", "../../x", "a b"])
def test_bad_action_name(name):
    with pytest.raises(ValueError):
        Intervention(name, 1, 1)


@pytest.mark.parametrize("dose,time", [(float("nan"), 1), (-1, 1), (1, float("inf")), (0, 0)])
def test_bad_action_cost(dose, time):
    with pytest.raises(ValueError):
        Intervention("a", dose, time)


def test_budget_charges_and_refuses():
    a = Intervention("a", 0.5, 0.2)
    b = Budget(2, 1, 0.4)
    assert b.permits(a)
    b.charge(a)
    b.charge(a)
    assert b.steps == 2 and (not b.permits(a))
    with pytest.raises(ValueError):
        b.charge(a)


@pytest.mark.parametrize(
    "kwargs", [{"max_steps": -1}, {"max_dose": -1}, {"max_seconds": float("nan")}, {"steps": 3}]
)
def test_bad_budget(kwargs):
    values = {"max_steps": 2, "max_dose": 1, "max_seconds": 1}
    values.update(kwargs)
    with pytest.raises(ValueError):
        Budget(**values)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n": 1},
        {"size": 8},
        {"correlation": 1.1},
        {"read_noise": -1},
        {"photon_scale": 0},
        {"temporal_change": -1},
        {"unknown_nuisance": float("nan")},
    ],
)
def test_bad_simulation_config(kwargs):
    with pytest.raises(ValueError):
        SyntheticConfig(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"model": "bad"},
        {"source": "bad"},
        {"target": "biology"},
        {"budgets": [1, 1]},
        {"policies": ["oracle"]},
        {"tune_thresholds": True, "early_stop": True},
        {"gates": {"max_shift": -1}},
        {"seed": True},
        {"source": "panels"},
    ],
)
def test_bad_experiment_config(kwargs):
    with pytest.raises((ValueError, TypeError)):
        Config(**kwargs).validate()


def test_unknown_config_key(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text("this_is_a_typo: 1\n")
    with pytest.raises(ValueError):
        load(p)


@pytest.mark.parametrize(
    "support,falsify", [(0.5, 0.1), (0.9, 0.5), (1.1, 0.1), (0.9, -0.1), (float("nan"), 0.1)]
)
def test_bad_thresholds(support, falsify):
    with pytest.raises(ValueError):
        Thresholds(support, falsify)


def test_decisions_and_eligibility():
    t = Thresholds(0.9, 0.1)
    assert t.decide(0.95) == "supported"
    assert t.decide(0.05) == "falsified"
    assert t.decide(0.5) == "abstain"
    assert t.decide(0.95, False) == "abstain"
    assert Thresholds(None, None).decide(0.99) == "abstain"


def test_exact_binomial_zero_errors():
    assert binomial_upper(0, 10) == pytest.approx(1 - 0.05**0.1)
    assert binomial_upper(0, 0) == 1
    with pytest.raises(ValueError):
        binomial_upper(2, 1)


def test_panel_validation(research_system):
    train, *_ = research_system
    with pytest.raises(ValueError):
        replace(train, labels=np.zeros(1))
    with pytest.raises(ValueError):
        replace(train, specimen_ids=["same"] * len(train.labels))
    with pytest.raises(ValueError):
        replace(train, reference=np.ones(len(train.labels)))
    p = train.subset([0, 1])
    assert p.images.shape[0] == 2
    with pytest.raises(ValueError):
        train.subset([])
    with pytest.raises(ValueError):
        train.target(np.zeros(len(train.labels)), "biology")
