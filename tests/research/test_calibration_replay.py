from dataclasses import replace

import numpy as np
import pytest

from counterfactual_microscopy.research.calibration import JointEvidence, Quantizer
from counterfactual_microscopy.research.decisions import Thresholds, tune_thresholds
from counterfactual_microscopy.research.policies import POLICIES, Selector
from counterfactual_microscopy.research.replay import Gates, ReplayProvider, evaluate, run_episode
from counterfactual_microscopy.research.types import Budget


@pytest.fixture
def joint():
    x = np.array([[0, 0]] * 30 + [[1, 1]] * 10 + [[0, 0]] * 10 + [[1, 1]] * 30)
    y = np.array([0] * 40 + [1] * 40)
    return JointEvidence(x, y, np.zeros(80, dtype=int), 2, alpha=0.01)


def test_joint_probabilities_and_information(joint):
    assert joint.prior(0) == 0.5
    assert joint.information_gain(0, 0, {}) > 0
    assert joint.information_gain(1, 0, {0: 1}) < 0.001
    for h in [0, 1]:
        assert joint.conditional(0, 0, {}, h).sum() == pytest.approx(1)


def test_duplicate_evidence_not_counted_twice(joint):
    p1 = joint.posterior(0, {0: 1})
    p2 = joint.posterior(0, {0: 1, 1: 1})
    assert p1 == pytest.approx(0.75, abs=0.001)
    assert p2 == pytest.approx(p1, abs=0.001)


@pytest.mark.parametrize("state", [0, 1])
def test_bayes_consistency(joint, state):
    prior = joint.prior(0)
    a = joint.conditional(0, 0, {}, 1)[state]
    b = joint.conditional(0, 0, {}, 0)[state]
    expected = prior * a / (prior * a + (1 - prior) * b)
    assert joint.posterior(0, {0: state}) == pytest.approx(expected)


def test_order_invariant(joint):
    assert joint.posterior(0, {0: 1, 1: 0}) == pytest.approx(joint.posterior(0, {1: 0, 0: 1}))
    assert joint.support(0, {0: 1, 1: 0}) == 0


@pytest.mark.parametrize("history", [{3: 0}, {0: 3}, {-1: 0}, {0: -1}])
def test_invalid_history(joint, history):
    with pytest.raises(ValueError):
        joint.posterior(0, history)


def test_repeated_action_and_context(joint):
    with pytest.raises(ValueError):
        joint.information_gain(0, 0, {0: 1})
    assert joint.choose_context(9) == -1
    with pytest.raises(ValueError):
        joint.prior(9)


def test_single_class_calibration_rejected():
    with pytest.raises(ValueError):
        JointEvidence(np.zeros((10, 2)), np.ones(10), np.zeros(10), 2)


def test_quantizer_calibration_only_determinism():
    x = np.random.default_rng(0).normal(size=(50, 7))
    a = Quantizer(3, 4).fit(x)
    b = Quantizer(3, 4).fit(x)
    np.testing.assert_array_equal(a.transform(x), b.transform(x))


def test_provider_reveal_audit(research_system):
    _, _, panel, *_ = research_system
    provider = ReplayProvider(panel.images[0])
    provider.baseline()
    assert provider.accesses == []
    provider.acquire(2)
    assert provider.accesses == [2]
    with pytest.raises(ValueError):
        provider.acquire(2)
    with pytest.raises(IndexError):
        provider.acquire(999)


@pytest.mark.parametrize("policy", POLICIES)
def test_each_policy_hard_budget(policy, research_system):
    _, _, panel, predictor, bundle = research_system
    provider = ReplayProvider(panel.images[0])
    selector = Selector(policy, seed=3, stop_if_negative=False)
    result = run_episode(
        provider,
        panel.actions,
        predictor,
        bundle,
        selector,
        Budget(1, 1.0, 1.0),
        Thresholds(),
        Gates(initial_ood=False, min_history_support=0, registration=False),
    )
    assert result["steps"] <= 1 and result["dose"] <= 1
    assert len(provider.accesses) == result["steps"]
    assert result["verdict"] in ["supported", "falsified", "abstain"]
    if policy == "no_reacquisition":
        assert provider.accesses == []


def test_test_labels_do_not_change_selection(research_system):
    _, _, panel, predictor, bundle = research_system
    panel = panel.subset([0, 1, 2])
    flipped = replace(panel, labels=1 - panel.labels)
    kwargs = {
        "policy": "eig",
        "max_steps": 2,
        "max_dose": 2,
        "max_seconds": 1,
        "thresholds": Thresholds(),
        "gates": Gates(initial_ood=False, min_history_support=0, registration=False),
        "stop_if_negative": False,
    }
    a = evaluate(panel, predictor, bundle, **kwargs)
    b = evaluate(flipped, predictor, bundle, **kwargs)
    for x, y in zip(a, b):
        assert x["revealed_actions"] == y["revealed_actions"]
        assert x["posterior"] == y["posterior"]
        assert x["target"] == 1 - y["target"]


def test_unchosen_images_not_inspected(research_system):
    _, _, panel, predictor, bundle = research_system

    class OnlyOne:
        def baseline(self):
            return panel.images[0, 0]

        def acquire(self, index):
            assert index == 0, "only the selected action may be requested"
            return panel.images[0, 1]

    result = run_episode(
        OnlyOne(),
        panel.actions,
        predictor,
        bundle,
        Selector("fixed"),
        Budget(1, 1, 1),
        Thresholds(),
        Gates(initial_ood=False, min_history_support=0, registration=False),
    )
    assert result["steps"] == 1


def test_failed_acquisition_charged_and_abstains(research_system):
    _, _, panel, predictor, bundle = research_system

    class Broken:
        def baseline(self):
            return panel.images[0, 0]

        def acquire(self, index):
            raise RuntimeError("camera failure")

    result = run_episode(
        Broken(),
        panel.actions,
        predictor,
        bundle,
        Selector("fixed"),
        Budget(2, 2, 1),
        Thresholds(),
        Gates(initial_ood=False),
    )
    assert result["steps"] == 1 and result["dose"] == 1 and (result["verdict"] == "abstain")
    assert not result["eligible"]


def test_no_budget_has_no_acquisition(research_system):
    _, _, panel, predictor, bundle = research_system
    provider = ReplayProvider(panel.images[0])
    result = run_episode(
        provider,
        panel.actions,
        predictor,
        bundle,
        Selector("eig"),
        Budget(0, 0, 0),
        Thresholds(),
        Gates(),
    )
    assert result["steps"] == 0 and provider.accesses == []


def test_threshold_calibration_disables_unsupported_verdicts():
    rows = [
        {"group_id": str(i), "target": i % 2, "posterior": 0.6, "eligible": True} for i in range(30)
    ]
    threshold, report = tune_thresholds(rows)
    assert threshold.support is None and threshold.falsify is None and report["both_disabled"]
