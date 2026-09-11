import numpy as np

from counterfactual_microscopy.research.optical_bayes import (
    MultiStateJoint,
    OpticalBayesModel,
    VectorQuantizer,
    choose_action,
    entropy_bits,
    split_specimens,
)


def make_joint():
    outcomes = np.array(
        [
            [0, 0, 0],
            [0, 0, 0],
            [0, 1, 0],
            [0, 1, 0],
            [0, 2, 0],
            [0, 2, 0],
        ],
        dtype=int,
    )
    targets = np.array([0, 0, 1, 1, 2, 2], dtype=int)
    return MultiStateJoint(
        outcomes,
        targets,
        cardinalities=(2, 3, 2),
        alpha=1.0,
    )


def test_entropy_bits_uniform():
    assert np.isclose(entropy_bits(np.ones(4) / 4), 2.0)


def test_multistate_joint_prefers_informative_action():
    joint = make_joint()
    history = {0: 0}

    informative = joint.information_gain(1, history)
    useless = joint.information_gain(2, history)

    assert informative > 0.5
    assert useless < 1e-9

    posterior = joint.posterior({0: 0, 1: 2})
    assert np.isclose(posterior.sum(), 1.0)
    assert posterior[2] == posterior.max()


def test_quantizer_roundtrip_shape():
    x = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.2],
            [1.0, 1.0],
            [1.1, 1.2],
            [2.0, 2.0],
            [2.1, 2.2],
        ]
    )
    q = VectorQuantizer(3, seed=1).fit(x)
    y = q.transform(x)

    assert y.shape == (6,)
    assert set(y) <= {0, 1, 2}


def test_split_specimens_is_disjoint_and_sized():
    specimens = [f"s{i:02d}" for i in range(42)]
    split = split_specimens(specimens, 123)

    assert len(split["calibration"]) == 24
    assert len(split["development"]) == 9
    assert len(split["test"]) == 9

    assert not (set(split["calibration"]) & set(split["development"]))
    assert not (set(split["calibration"]) & set(split["test"]))
    assert not (set(split["development"]) & set(split["test"]))


def test_choose_action_eig_uses_joint_information():
    joint = make_joint()

    # Minimal fitted quantizers for OpticalBayesModel construction.
    baseline_q = VectorQuantizer(2, seed=1).fit(np.array([[0.0], [0.1], [1.0], [1.1]]))
    action_q1 = VectorQuantizer(3, seed=2).fit(np.array([[0.0], [0.1], [1.0], [1.1], [2.0], [2.1]]))
    action_q2 = VectorQuantizer(2, seed=3).fit(np.array([[0.0], [0.1], [1.0], [1.1]]))

    model = OpticalBayesModel(
        states_um=(-6, 0, 6),
        actions_um=(-6, 6),
        baseline_quantizer=baseline_q,
        action_quantizers=(action_q1, action_q2),
        joint=joint,
        expected_quality=np.ones((3, 2)),
        calibration_specimens=("a", "b"),
        n_bins=3,
    )

    chosen, scores = choose_action(
        "eig",
        model,
        history={0: 0},
        used_actions=set(),
        rng=np.random.default_rng(0),
    )

    assert chosen == 0
    assert len(scores) == 2
