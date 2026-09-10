"""Calibration-only discretization and coherent full-history evidence likelihoods.

A panel is a vector of discrete action outcomes. Its joint distribution is an
empirical mixture plus uniform pseudomass. Conditioning that SAME joint law avoids
multiplying correlated images as though every reacquisition were independent.
This is a static-panel model, not a bleaching/dynamics model or causal proof.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import StandardScaler

from .images import entropy, evidence, features, quality, register
from .models import FeatureReference, Predictor
from .types import Panel
from .validation import binary, finite, integer, numeric_array


def contexts(p: np.ndarray) -> np.ndarray:
    p = numeric_array("probabilities", p, 1)
    if np.any((p < 0) | (p > 1)):
        raise ValueError("invalid baseline probabilities")
    return (2 * (p >= 0.5) + (np.maximum(p, 1 - p) >= 0.8)).astype(int)


class Quantizer:
    def __init__(self, n_states: int = 5, seed: int = 42):
        integer("n_states", n_states, 2)
        self.k = n_states
        self.seed = seed

    def fit(self, x: np.ndarray) -> Quantizer:
        x = numeric_array("evidence", x, 2)
        if len(np.unique(x, axis=0)) < self.k:
            raise ValueError("too few distinct evidence vectors")
        self.scaler = StandardScaler().fit(x)
        self.cluster = KMeans(n_clusters=self.k, n_init=10, random_state=self.seed).fit(
            self.scaler.transform(x)
        )
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if not hasattr(self, "cluster"):
            raise RuntimeError("quantizer not fitted")
        return self.cluster.predict(self.scaler.transform(numeric_array("evidence", x, 2)))


class JointEvidence:
    """Exact conditionals of a smoothed empirical JOINT static-panel distribution."""

    def __init__(
        self,
        outcomes: np.ndarray,
        target: np.ndarray,
        context: np.ndarray,
        n_states: int,
        *,
        alpha: float = 2.0,
        prior_alpha: float = 1.0,
        min_context: int = 12,
    ):
        x = numeric_array("outcomes", outcomes, 2)
        y = binary("target", target)
        c = numeric_array("contexts", context, 1)
        integer("n_states", n_states, 2)
        integer("min_context", min_context, 1)
        finite("alpha", alpha, 1e-12)
        finite("prior_alpha", prior_alpha, 1e-12)
        if len(x) != len(y) or c.shape != y.shape:
            raise ValueError("evidence metadata mismatch")
        if np.any(x != x.astype(int)) or np.any((x < 0) | (x >= n_states)):
            raise ValueError("outcomes outside support")
        if np.any(c != c.astype(int)) or np.any(c < 0):
            raise ValueError("invalid contexts")
        if np.min(np.bincount(y, minlength=2)) < 2:
            raise ValueError(
                "calibration requires at least two examples of EACH verification-target class"
            )
        self.x = x.astype(int)
        self.y = y
        self.contexts = c.astype(int)
        self.k = n_states
        self.alpha, self.prior_alpha, self.min_context = (alpha, prior_alpha, min_context)

    def choose_context(self, c: int) -> int:
        m = self.contexts == c
        if m.sum() < self.min_context or np.min(np.bincount(self.y[m], minlength=2)) < 2:
            return -1
        return int(c)

    def _base(self, c: int) -> np.ndarray:
        if c == -1:
            return np.ones(len(self.y), dtype=bool)
        m = self.contexts == c
        if not m.any():
            raise ValueError("unseen context: use choose_context fallback")
        return m

    def _matched(self, c: int, history: dict[int, int]) -> tuple[np.ndarray, np.ndarray]:
        base = self._base(c)
        m = base.copy()
        for a, s in history.items():
            integer("action", a)
            integer("state", s)
            if a >= self.x.shape[1] or s >= self.k:
                raise ValueError("invalid history")
            m &= self.x[:, a] == s
        return (base, m)

    def support(self, c: int, history: dict[int, int]) -> int:
        return int(self._matched(c, history)[1].sum())

    def prior(self, c: int) -> float:
        y = self.y[self._base(c)]
        return float((y.sum() + self.prior_alpha) / (len(y) + 2 * self.prior_alpha))

    def posterior(self, c: int, history: dict[int, int]) -> float:
        if not history:
            return self.prior(c)
        base, m = self._matched(c, history)
        prior = self.prior(c)
        scores = []
        for h, weight in [(0, 1 - prior), (1, prior)]:
            count = int(np.sum(m & (self.y == h)))
            total = int(np.sum(base & (self.y == h)))
            log_pseudo = np.log(self.alpha) - len(history) * np.log(self.k)
            log_num = np.logaddexp(np.log(count) if count else -np.inf, log_pseudo)
            scores.append(np.log(weight) + log_num - np.log(total + self.alpha))
        return float(np.exp(scores[1] - np.logaddexp(*scores)))

    def conditional(self, a: int, c: int, history: dict[int, int], hypothesis: int) -> np.ndarray:
        integer("action", a)
        if a >= self.x.shape[1] or a in history:
            raise ValueError("action invalid or already observed")
        if hypothesis not in [0, 1]:
            raise ValueError("hypothesis must be binary")
        _, m = self._matched(c, history)
        selected = m & (self.y == hypothesis)
        counts = np.bincount(self.x[selected, a], minlength=self.k).astype(float)
        mass = self.alpha * np.exp(-len(history) * np.log(self.k))
        p = (counts + mass / self.k) / (counts.sum() + mass)
        return p / p.sum()

    def information_gain(self, a: int, c: int, history: dict[int, int]) -> float:
        w = self.posterior(c, history)
        p0 = self.conditional(a, c, history, 0)
        p1 = self.conditional(a, c, history, 1)
        mix = (1 - w) * p0 + w * p1

        def ent(p):
            m = p > 0
            return float(-np.sum(p[m] * np.log2(p[m])))

        result = ent(mix) - (1 - w) * ent(p0) - w * ent(p1)
        return max(0.0, min(float(entropy(w)), result))


class ExpectedScores:
    """CALIBRATION-fitted predictors of candidate entropy, OOD distance, and quality.

    Test selection only sees baseline images and baseline probabilities. These are
    explicit baseline implementations, not reproductions of published algorithms.
    """

    def fit(
        self, baselines: np.ndarray, p0: np.ndarray, scores: np.ndarray, seed: int = 42
    ) -> ExpectedScores:
        x = np.column_stack([features(baselines), p0])
        self.shape = scores.shape[1:]
        self.model = ExtraTreesRegressor(
            n_estimators=48, max_depth=6, min_samples_leaf=5, random_state=seed, n_jobs=1
        )
        self.model.fit(x, scores.reshape(len(scores), -1))
        return self

    def predict(self, baseline: np.ndarray, p0: float) -> np.ndarray:
        return self.model.predict(np.column_stack([features(baseline[None]), [p0]])).reshape(
            self.shape
        )


@dataclass
class Bundle:
    quantizer: Quantizer
    joint: JointEvidence
    expected_scores: ExpectedScores
    reference: FeatureReference
    target_kind: str
    report: dict


def fit_calibration(
    panel: Panel,
    predictor: Predictor,
    reference: FeatureReference,
    *,
    target_kind: str = "correctness",
    n_states: int = 5,
    seed: int = 42,
    alpha: float = 2.0,
    min_context: int = 12,
) -> Bundle:
    n, v, h, w = panel.images.shape
    p = predictor.predict_proba(panel.images.reshape(n * v, h, w)).reshape(n, v)
    all_evidence = []
    scores = []
    for i in range(n):
        ie = []
        isc = []
        for j in range(v - 1):
            baseline = panel.images[i, 0]
            candidate = panel.images[i, j + 1]
            reg = register(baseline, candidate)
            ie.append(evidence(baseline, candidate, p[i, 0], p[i, j + 1], reference.scale, reg))
            isc.append(
                [
                    float(entropy(p[i, j + 1])),
                    float(reference.score(candidate[None])[0]),
                    quality(candidate),
                ]
            )
        all_evidence.append(ie)
        scores.append(isc)
    e = np.asarray(all_evidence)
    quantizer = Quantizer(n_states, seed).fit(e.reshape(-1, e.shape[-1]))
    outcomes = quantizer.transform(e.reshape(-1, e.shape[-1])).reshape(n, v - 1)
    target = panel.target((p[:, 0] >= 0.5).astype(int), target_kind)
    c = contexts(p[:, 0])
    joint = JointEvidence(outcomes, target, c, n_states, alpha=alpha, min_context=min_context)
    expected = ExpectedScores().fit(panel.images[:, 0], p[:, 0], np.asarray(scores), seed)
    return Bundle(
        quantizer,
        joint,
        expected,
        reference,
        target_kind,
        {
            "n": n,
            "positive": int(target.sum()),
            "negative": int((1 - target).sum()),
            "target": target_kind,
            "n_states": n_states,
            "fit_specimen_ids": panel.specimen_ids,
            "causal_identification_claim": False,
            "likelihood_model": "empirical full-panel joint plus uniform pseudomass",
            "reference_source": panel.provenance.get(
                "reference_source", panel.provenance.get("label_source")
            ),
        },
    )
