"""Discrete multi-state Bayesian optical-action model for LSFM engineering replay."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

STATE_OFFSETS_UM = (-30, -24, -18, -12, -6, 6, 12, 18, 24, 30)
ACTION_OFFSETS_UM = (-18, -12, -6, 6, 12, 18)

BASE_FEATURES = (
    "baseline_quality",
    "baseline_std",
    "baseline_gradient",
    "baseline_laplacian",
    "baseline_maxima",
    "baseline_bright_fraction",
    "baseline_dark_fraction",
)

ACTION_FEATURES = (
    "quality_delta",
    "std_delta",
    "gradient_delta",
    "laplacian_delta",
    "maxima_delta",
    "bright_fraction_delta",
    "dark_fraction_delta",
    "one_minus_correlation",
    "registration_shift",
)


def entropy_bits(probabilities: np.ndarray) -> float:
    p = np.asarray(probabilities, dtype=float)
    if p.ndim != 1 or len(p) == 0 or not np.isfinite(p).all():
        raise ValueError("probabilities must be a finite nonempty vector")
    if np.any(p < 0) or p.sum() <= 0:
        raise ValueError("invalid probabilities")
    p = p / p.sum()
    m = p > 0
    return float(-np.sum(p[m] * np.log2(p[m])))


class VectorQuantizer:
    """Calibration-only StandardScaler + KMeans discretizer."""

    def __init__(self, n_bins: int = 5, seed: int = 42):
        if n_bins < 2:
            raise ValueError("n_bins must be >= 2")
        self.n_bins = int(n_bins)

        seed_value = int(seed)

        if seed_value < 0:
            raise ValueError("seed must be nonnegative")

        # NumPy Generator/SeedSequence accepts arbitrarily large positive
        # integer seeds, but sklearn KMeans still routes integer random_state
        # through the legacy RandomState-compatible 32-bit seed boundary.
        # Preserve deterministic hashing while adapting only at that API edge.
        self.seed = seed_value % (1 << 32)

    def fit(self, x: np.ndarray) -> VectorQuantizer:
        x = np.asarray(x, dtype=float)
        if x.ndim != 2 or len(x) < self.n_bins or not np.isfinite(x).all():
            raise ValueError("invalid quantizer training matrix")
        if len(np.unique(x, axis=0)) < self.n_bins:
            raise ValueError("too few distinct observations for requested bins")
        self.scaler = StandardScaler().fit(x)
        z = self.scaler.transform(x)
        self.cluster = KMeans(
            n_clusters=self.n_bins,
            n_init=20,
            random_state=self.seed,
        ).fit(z)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if not hasattr(self, "cluster"):
            raise RuntimeError("quantizer not fitted")
        x = np.asarray(x, dtype=float)
        if x.ndim != 2 or not np.isfinite(x).all():
            raise ValueError("invalid quantizer input")
        return self.cluster.predict(self.scaler.transform(x)).astype(int)


class MultiStateJoint:
    """Smoothed empirical full-joint model over correlated discrete observations.

    outcomes has one column for the baseline observation plus one per candidate
    acquisition. Posterior conditioning always uses the same empirical joint law,
    avoiding multiplication of action outcomes as if they were conditionally
    independent.
    """

    def __init__(
        self,
        outcomes: np.ndarray,
        targets: np.ndarray,
        cardinalities: Iterable[int],
        *,
        alpha: float = 2.0,
        prior_alpha: float = 1.0,
    ):
        x = np.asarray(outcomes)
        y = np.asarray(targets)
        ks = tuple(int(k) for k in cardinalities)

        if x.ndim != 2 or y.ndim != 1 or len(x) != len(y):
            raise ValueError("joint evidence dimensions do not match")
        if len(ks) != x.shape[1] or any(k < 2 for k in ks):
            raise ValueError("invalid outcome cardinalities")
        if alpha <= 0 or prior_alpha <= 0:
            raise ValueError("smoothing parameters must be positive")
        if not np.issubdtype(x.dtype, np.integer):
            if np.any(x != x.astype(int)):
                raise ValueError("outcomes must be integers")
            x = x.astype(int)
        else:
            x = x.astype(int, copy=False)
        if not np.issubdtype(y.dtype, np.integer):
            if np.any(y != y.astype(int)):
                raise ValueError("targets must be integers")
            y = y.astype(int)
        else:
            y = y.astype(int, copy=False)

        if np.any(y < 0):
            raise ValueError("targets must be nonnegative")
        self.n_states = int(y.max()) + 1
        if set(np.unique(y)) != set(range(self.n_states)):
            raise ValueError("target states must be contiguous from zero")
        for j, k in enumerate(ks):
            if np.any((x[:, j] < 0) | (x[:, j] >= k)):
                raise ValueError(f"outcomes outside support in column {j}")

        counts = np.bincount(y, minlength=self.n_states)
        if np.any(counts < 2):
            raise ValueError("each hidden state needs at least two calibration rows")

        self.x = x
        self.y = y
        self.cardinalities = ks
        self.alpha = float(alpha)
        self.prior_alpha = float(prior_alpha)
        self.state_counts = counts.astype(float)

    def _history_mask(self, history: dict[int, int]) -> np.ndarray:
        mask = np.ones(len(self.y), dtype=bool)
        for column, outcome in history.items():
            if column < 0 or column >= self.x.shape[1]:
                raise ValueError("history column outside support")
            if outcome < 0 or outcome >= self.cardinalities[column]:
                raise ValueError("history outcome outside support")
            mask &= self.x[:, column] == outcome
        return mask

    def _pseudo_history_mass(self, history: dict[int, int]) -> float:
        denominator = 1.0
        for column in history:
            denominator *= self.cardinalities[column]
        return self.alpha / denominator

    def prior(self) -> np.ndarray:
        counts = self.state_counts + self.prior_alpha
        return counts / counts.sum()

    def support(self, history: dict[int, int]) -> int:
        return int(self._history_mask(history).sum())

    def posterior(self, history: dict[int, int]) -> np.ndarray:
        if not history:
            return self.prior()

        mask = self._history_mask(history)
        pseudo = self._pseudo_history_mass(history)
        prior = self.prior()
        log_scores = np.empty(self.n_states, dtype=float)

        for state in range(self.n_states):
            count = int(np.sum(mask & (self.y == state)))
            total = self.state_counts[state]
            likelihood = (count + pseudo) / (total + self.alpha)
            log_scores[state] = np.log(prior[state]) + np.log(likelihood)

        log_scores -= np.max(log_scores)
        scores = np.exp(log_scores)
        return scores / scores.sum()

    def conditional(
        self,
        column: int,
        history: dict[int, int],
        state: int,
    ) -> np.ndarray:
        if column in history:
            raise ValueError("cannot predict an already observed column")
        if column < 0 or column >= self.x.shape[1]:
            raise ValueError("column outside support")
        if state < 0 or state >= self.n_states:
            raise ValueError("state outside support")

        mask = self._history_mask(history) & (self.y == state)
        k = self.cardinalities[column]
        counts = np.bincount(self.x[mask, column], minlength=k).astype(float)
        history_mass = self._pseudo_history_mass(history)
        probabilities = (counts + history_mass / k) / (counts.sum() + history_mass)
        return probabilities / probabilities.sum()

    def predictive(self, column: int, history: dict[int, int]) -> np.ndarray:
        posterior = self.posterior(history)
        result = np.zeros(self.cardinalities[column], dtype=float)
        for state, weight in enumerate(posterior):
            result += weight * self.conditional(column, history, state)
        return result / result.sum()

    def information_gain(self, column: int, history: dict[int, int]) -> float:
        if column in history:
            raise ValueError("action already observed")
        current = self.posterior(history)
        before = entropy_bits(current)
        predictive = self.predictive(column, history)
        expected_after = 0.0
        for outcome, weight in enumerate(predictive):
            if weight <= 0:
                continue
            next_history = dict(history)
            next_history[column] = outcome
            expected_after += float(weight) * entropy_bits(self.posterior(next_history))
        value = before - expected_after
        return float(max(0.0, min(before, value)))


@dataclass
class OpticalBayesModel:
    states_um: tuple[int, ...]
    actions_um: tuple[int, ...]
    baseline_quantizer: VectorQuantizer
    action_quantizers: tuple[VectorQuantizer, ...]
    joint: MultiStateJoint
    expected_quality: np.ndarray
    calibration_specimens: tuple[str, ...]
    n_bins: int

    def __post_init__(self) -> None:
        if self.expected_quality.shape != (
            len(self.states_um),
            len(self.actions_um),
        ):
            raise ValueError("expected-quality table shape mismatch")

    def baseline_outcome(self, features: np.ndarray) -> int:
        return int(self.baseline_quantizer.transform(np.asarray(features, dtype=float)[None])[0])

    def action_outcome(self, action_index: int, features: np.ndarray) -> int:
        if action_index < 0 or action_index >= len(self.actions_um):
            raise ValueError("action index outside support")
        return int(
            self.action_quantizers[action_index].transform(np.asarray(features, dtype=float)[None])[
                0
            ]
        )

    def posterior(self, history: dict[int, int]) -> np.ndarray:
        return self.joint.posterior(history)

    def information_gain(self, action_index: int, history: dict[int, int]) -> float:
        return self.joint.information_gain(action_index + 1, history)

    def expected_candidate_quality(
        self,
        action_index: int,
        history: dict[int, int],
    ) -> float:
        posterior = self.posterior(history)
        return float(posterior @ self.expected_quality[:, action_index])

    def expected_focus_residual(
        self,
        action_index: int,
        history: dict[int, int],
    ) -> float:
        posterior = self.posterior(history)
        action = self.actions_um[action_index]
        residual = np.abs(np.asarray(self.states_um, dtype=float) + action)
        return float(posterior @ residual)


POLICIES = (
    "eig",
    "random",
    "fixed_positive",
    "fixed_negative",
    "fixed_alternating",
    "expected_quality",
    "posterior_greedy",
    "oracle_focus",
)

_FIXED_ORDERS = {
    "fixed_positive": (6, 12, 18, -6, -12, -18),
    "fixed_negative": (-6, -12, -18, 6, 12, 18),
    "fixed_alternating": (6, -6, 12, -12, 18, -18),
}


def split_specimens(
    specimens: Iterable[str],
    seed: int,
) -> dict[str, tuple[str, ...]]:
    values = sorted(set(specimens))
    if len(values) != 42:
        raise ValueError(f"expected 42 specimens; got {len(values)}")
    rng = np.random.default_rng(int(seed))
    order = rng.permutation(len(values))
    calibration = tuple(values[int(i)] for i in order[:24])
    development = tuple(values[int(i)] for i in order[24:33])
    test = tuple(values[int(i)] for i in order[33:42])
    result = {
        "calibration": calibration,
        "development": development,
        "test": test,
    }
    sets = [set(v) for v in result.values()]
    if any(sets[i] & sets[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("specimen split overlap")
    if set().union(*sets) != set(values):
        raise ValueError("specimen split does not cover all specimens")
    return result


def choose_action(
    policy: str,
    model: OpticalBayesModel,
    history: dict[int, int],
    used_actions: set[int],
    rng: np.random.Generator,
    *,
    true_state_um: int | None = None,
) -> tuple[int, list[dict]]:
    if policy not in POLICIES:
        raise ValueError(f"unknown policy: {policy}")
    available = [index for index in range(len(model.actions_um)) if index not in used_actions]
    if not available:
        raise ValueError("no actions remain")

    score_rows: list[dict] = []

    if policy == "random":
        chosen = int(rng.choice(available))
        for index in available:
            score_rows.append(
                {
                    "action_index": index,
                    "action_offset_um": model.actions_um[index],
                    "score": None,
                    "information_gain": model.information_gain(index, history),
                }
            )
        return chosen, score_rows

    if policy in _FIXED_ORDERS:
        order = _FIXED_ORDERS[policy]
        action_lookup = {offset: i for i, offset in enumerate(model.actions_um)}
        chosen = next(
            action_lookup[offset] for offset in order if action_lookup[offset] in available
        )
        for index in available:
            score_rows.append(
                {
                    "action_index": index,
                    "action_offset_um": model.actions_um[index],
                    "score": -float(order.index(model.actions_um[index])),
                    "information_gain": model.information_gain(index, history),
                }
            )
        return chosen, score_rows

    if policy == "oracle_focus":
        if true_state_um is None:
            raise ValueError("oracle_focus requires the hidden true state")
        scores = {index: -abs(true_state_um + model.actions_um[index]) for index in available}
    elif policy == "eig":
        scores = {index: model.information_gain(index, history) for index in available}
    elif policy == "expected_quality":
        scores = {index: model.expected_candidate_quality(index, history) for index in available}
    elif policy == "posterior_greedy":
        scores = {index: -model.expected_focus_residual(index, history) for index in available}
    else:
        raise RuntimeError("unreachable policy branch")

    chosen = min(
        available,
        key=lambda index: (-scores[index], index),
    )
    for index in available:
        score_rows.append(
            {
                "action_index": index,
                "action_offset_um": model.actions_um[index],
                "score": float(scores[index]),
                "information_gain": model.information_gain(index, history),
            }
        )
    return chosen, score_rows


def posterior_summary(
    posterior: np.ndarray,
    states_um: tuple[int, ...],
    true_state_um: int,
) -> dict:
    p = np.asarray(posterior, dtype=float)
    states = np.asarray(states_um, dtype=float)
    if p.shape != states.shape:
        raise ValueError("posterior/state shape mismatch")
    map_index = int(np.argmax(p))
    map_state = int(states_um[map_index])
    posterior_mean = float(p @ states)
    return {
        "map_state_um": map_state,
        "posterior_mean_um": posterior_mean,
        "map_abs_error_um": float(abs(map_state - true_state_um)),
        "posterior_mean_abs_error_um": float(abs(posterior_mean - true_state_um)),
        "entropy_bits": entropy_bits(p),
        "true_state_probability": float(p[states_um.index(true_state_um)]),
        "within_6um": int(abs(map_state - true_state_um) <= 6),
        "within_12um": int(abs(map_state - true_state_um) <= 12),
    }
