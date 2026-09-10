"""CPU-first predictors and train-fitted OOD features."""

from __future__ import annotations

from typing import Protocol

import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .images import features
from .validation import binary, finite, integer, numeric_array


class Predictor(Protocol):
    def fit(self, images: np.ndarray, labels: np.ndarray) -> Predictor: ...

    def predict_proba(self, images: np.ndarray) -> np.ndarray: ...


class FeaturePredictor:
    def __init__(self, kind: str = "logistic", seed: int = 42):
        if kind not in {"logistic", "forest"}:
            raise ValueError("unknown model")
        self.kind = kind
        learner = (
            LogisticRegression(C=0.5, max_iter=1000, random_state=seed)
            if kind == "logistic"
            else RandomForestClassifier(
                n_estimators=80, max_depth=6, min_samples_leaf=3, random_state=seed, n_jobs=1
            )
        )
        self.model = make_pipeline(StandardScaler(), learner)
        self.fitted = False

    def fit(self, images: np.ndarray, labels: np.ndarray) -> FeaturePredictor:
        y = binary("labels", labels)
        x = features(images)
        if len(x) != len(y) or len(np.unique(y)) != 2:
            raise ValueError("aligned data with both classes required")
        self.model.fit(x, y)
        self.fitted = True
        return self

    def predict_proba(self, images: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("model not fitted")
        return self.model.predict_proba(features(images))[:, 1]


class TinyCNN:
    """Optional torch baseline; fixed epochs, no test-driven early stopping."""

    def __init__(self, seed: int = 42, epochs: int = 12, batch_size: int = 32):
        integer("epochs", epochs, 1)
        integer("batch_size", batch_size, 1)
        self.seed, self.epochs, self.batch_size = (seed, epochs, batch_size)
        self.net = None

    def fit(self, images: np.ndarray, labels: np.ndarray) -> TinyCNN:
        import torch
        from torch import nn

        x = numeric_array("images", images, 3).astype(np.float32)
        y = binary("labels", labels)
        if len(x) != len(y) or len(np.unique(y)) != 2:
            raise ValueError("CNN requires aligned data with both classes")
        torch.manual_seed(self.seed)
        self.net = nn.Sequential(
            nn.Conv2d(1, 12, 3, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(2),
            nn.Conv2d(12, 24, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(24, 1),
        )
        opt = torch.optim.Adam(self.net.parameters(), lr=0.003)
        tx = torch.from_numpy(x[:, None])
        ty = torch.from_numpy(y.astype(np.float32))
        rng = np.random.default_rng(self.seed)
        self.losses = []
        for _ in range(self.epochs):
            self.net.train()
            losses = []
            for ix in np.array_split(
                rng.permutation(len(x)), max(1, int(np.ceil(len(x) / self.batch_size)))
            ):
                opt.zero_grad()
                loss = nn.functional.binary_cross_entropy_with_logits(
                    self.net(tx[ix]).squeeze(1), ty[ix]
                )
                loss.backward()
                opt.step()
                losses.append(float(loss.detach()))
            self.losses.append(float(np.mean(losses)))
        return self

    def predict_proba(self, images: np.ndarray) -> np.ndarray:
        import torch

        if self.net is None:
            raise RuntimeError("CNN not fitted")
        x = numeric_array("images", images, 3).astype(np.float32)
        self.net.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(x), self.batch_size):
                out.append(
                    torch.sigmoid(
                        self.net(torch.from_numpy(x[i : i + self.batch_size, None])).squeeze(1)
                    ).numpy()
                )
        return np.concatenate(out)


class FeatureReference:
    def fit(self, images: np.ndarray, gate_multiplier: float = 3.0) -> FeatureReference:
        finite("gate_multiplier", gate_multiplier, 1)
        x = features(images)
        self.scaler = StandardScaler().fit(x)
        z = self.scaler.transform(x)
        self.cov = LedoitWolf().fit(z)
        self.gate = max(1.0, float(np.quantile(self.cov.mahalanobis(z), 0.99)) * gate_multiplier)
        return self

    @property
    def scale(self) -> np.ndarray:
        if not hasattr(self, "scaler"):
            raise RuntimeError("reference not fitted")
        return np.maximum(self.scaler.scale_, 1e-06)

    def score(self, images: np.ndarray) -> np.ndarray:
        if not hasattr(self, "cov"):
            raise RuntimeError("reference not fitted")
        return self.cov.mahalanobis(self.scaler.transform(features(images)))


def make_predictor(kind: str, seed: int, epochs: int = 12) -> Predictor:
    return TinyCNN(seed, epochs) if kind == "cnn" else FeaturePredictor(kind, seed)
