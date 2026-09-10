import numpy as np
import pytest
from scipy import ndimage

from counterfactual_microscopy.research.images import (
    as_float,
    entropy,
    evidence,
    features,
    quality,
    register,
)
from counterfactual_microscopy.research.models import FeaturePredictor, TinyCNN
from counterfactual_microscopy.research.synthetic import SyntheticConfig, generate, render


def test_detector_scale_preserved():
    x = np.array([[0, 128], [255, 64]], dtype=np.uint8)
    assert as_float(x)[0, 1] == pytest.approx(128 / 255)
    with pytest.raises(ValueError):
        as_float(np.ones((8, 8)) * 2)


def test_features_and_blur():
    rng = np.random.default_rng(1)
    x = rng.uniform(0, 0.9, (32, 32)).astype(np.float32)
    blur = ndimage.gaussian_filter(x, 2)
    assert features(np.stack([x, blur])).shape == (2, 16)
    assert quality(x) > quality(blur)


@pytest.mark.parametrize("p", [0.0, 0.1, 0.5, 0.9, 1.0])
def test_entropy_bounds(p):
    assert 0 <= entropy(p) <= 1 + 1e-12
    assert entropy(p) == pytest.approx(entropy(1 - p))


@pytest.mark.parametrize("dy,dx", [(0, 0), (2, -3), (-4, 1), (1, 4)])
def test_registration_translation(dy, dx):
    x = np.random.default_rng(2).uniform(0.1, 0.8, (32, 32)).astype(np.float32)
    y = np.roll(x, (dy, dx), (0, 1))
    reg = register(x, y, max_shift=6, min_correlation=0.5)
    assert (reg.shift_y, reg.shift_x) == (-dy, -dx)
    assert reg.valid and reg.correlation == pytest.approx(1)


def test_registration_flat_and_shapes():
    assert not register(np.ones((16, 16)), np.ones((16, 16))).valid
    assert register(np.zeros((16, 16)), np.zeros((17, 16))).reason == "shape_mismatch"


def test_evidence_finite(research_system):
    *_, test, predictor, bundle = research_system
    a, b = test.images[0, :2]
    p = predictor.predict_proba(test.images[0, :2])
    e = evidence(a, b, *p, bundle.reference.scale, register(a, b))
    assert e.shape == (7,) and np.isfinite(e).all()
    with pytest.raises(ValueError):
        evidence(a, b, *p, np.zeros(16), register(a, b))


def test_simulation_deterministic_and_pairing():
    cfg = SyntheticConfig(n=8, size=24, seed=17)
    a, b = (generate(cfg), generate(cfg))
    np.testing.assert_array_equal(a.images, b.images)
    assert not np.array_equal(a.images[:, 0], a.images[:, 1])
    assert a.provenance["pairing_verified"]
    assert set(a.labels) == {0, 1}


@pytest.mark.parametrize("kind", ["logistic", "forest"])
def test_feature_model_training(kind, research_system):
    train, _, test, *_ = research_system
    model = FeaturePredictor(kind, 1)
    with pytest.raises(RuntimeError):
        model.predict_proba(test.images[:, 0])
    model.fit(train.images[:, 0], train.labels)
    p = model.predict_proba(test.images[:, 0])
    assert p.shape == (20,) and np.all((p >= 0) & (p <= 1))


def test_model_rejects_single_class(research_system):
    train, *_ = research_system
    with pytest.raises(ValueError):
        FeaturePredictor().fit(train.images[:, 0], np.ones(len(train.labels)))


def test_tiny_cnn_optional(research_system):
    torch = pytest.importorskip("torch")
    torch.set_num_threads(1)
    train, _, test, *_ = research_system
    model = TinyCNN(1, epochs=1, batch_size=8).fit(train.images[:16, 0], train.labels[:16])
    p = model.predict_proba(test.images[:3, 0])
    assert p.shape == (3,) and np.isfinite(p).all()


def test_render_noise_and_saturation():
    params = {
        "blur": 0.0,
        "gain": 10.0,
        "background": 0.0,
        "exposure": 1.0,
        "photon_scale": 1000.0,
        "read_noise": 0.0,
        "shading": 0.0,
    }
    x = render(np.ones((16, 16)), **params, rng=np.random.default_rng(0))
    assert x.min() == 1
    with pytest.raises(ValueError):
        render(np.ones((16, 16)), **{**params, "exposure": 0}, rng=np.random.default_rng(0))
