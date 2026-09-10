"""No-peeking replay: ground-truth labels are attached only AFTER inference."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Protocol

import numpy as np

from .calibration import Bundle, contexts
from .decisions import Thresholds
from .images import evidence, register
from .io import seed_for
from .models import Predictor
from .policies import Selector
from .types import Budget, Intervention, Panel
from .validation import finite, integer


class Provider(Protocol):
    def baseline(self) -> np.ndarray: ...

    def acquire(self, index: int) -> np.ndarray: ...


class ReplayProvider:
    def __init__(self, views: np.ndarray):
        self.__views = views
        self.accesses = []

    def baseline(self) -> np.ndarray:
        return self.__views[0].copy()

    def acquire(self, index: int) -> np.ndarray:
        integer("index", index)
        if index >= len(self.__views) - 1:
            raise IndexError("candidate unavailable")
        if index in self.accesses:
            raise ValueError("static replay cannot resample a previously revealed image")
        self.accesses.append(index)
        return self.__views[index + 1].copy()


@dataclass(frozen=True)
class Gates:
    min_history_support: int = 3
    max_shift: float = 6.0
    min_correlation: float = 0.05
    min_overlap: float = 0.7
    initial_ood: bool = True
    registration: bool = True

    def __post_init__(self):
        integer("min_history_support", self.min_history_support)
        finite("max_shift", self.max_shift, 0)
        finite("min_correlation", self.min_correlation, -1, 1)
        finite("min_overlap", self.min_overlap, 0, 1)


def run_episode(
    provider: Provider,
    actions: tuple[Intervention, ...],
    predictor: Predictor,
    bundle: Bundle,
    selector: Selector,
    budget: Budget,
    thresholds: Thresholds,
    gates: Gates,
    early_stop: bool = False,
) -> dict:
    """No task labels, independent annotations, or unchosen images enter this call."""
    start = time.perf_counter()
    baseline = provider.baseline()
    p0 = float(predictor.predict_proba(baseline[None])[0])
    finite("p0", p0, 0, 1)
    context = bundle.joint.choose_context(int(contexts(np.array([p0]))[0]))
    expected = bundle.expected_scores.predict(baseline, p0)
    ood = float(bundle.reference.score(baseline[None])[0])
    eligible = not gates.initial_ood or ood <= bundle.reference.gate
    reason = "ok" if eligible else "initial_ood"
    history = {}
    trace = []
    posterior = bundle.joint.posterior(context, history)
    while eligible:
        candidates = [i for i, a in enumerate(actions) if i not in history and budget.permits(a)]
        ranked = selector.rank(candidates, actions, bundle, context, history, expected)
        selected = selector.choose(ranked)
        if selected is None:
            if not trace and selector.name != "no_reacquisition":
                reason = "no_admissible_or_positive_utility_action"
            break
        i = selected.index
        a = actions[i]
        budget.charge(a)
        event = {
            "step": budget.steps,
            "action": a.name,
            "index": i,
            "candidate_scores": [asdict(s) for s in ranked],
            "dose": a.dose,
            "seconds": a.seconds,
            "posterior_before": posterior,
        }
        try:
            candidate = provider.acquire(i)
            pa = float(predictor.predict_proba(candidate[None])[0])
            reg = register(
                baseline,
                candidate,
                max_shift=gates.max_shift,
                min_correlation=gates.min_correlation,
                min_overlap=gates.min_overlap,
            )
            e = evidence(baseline, candidate, p0, pa, bundle.reference.scale, reg)
            state = int(bundle.quantizer.transform(e[None])[0])
            history[i] = state
            posterior = bundle.joint.posterior(context, history)
            matches = bundle.joint.support(context, history)
            event.update(
                candidate_probability=pa,
                registration=asdict(reg),
                evidence=e.tolist(),
                state=state,
                posterior_after=posterior,
                history_matches=matches,
            )
            if gates.registration and (not reg.valid):
                eligible = False
                reason = "registration:" + reg.reason
            elif matches < gates.min_history_support:
                eligible = False
                reason = "sparse_calibration_history"
        except (ValueError, RuntimeError, OSError, IndexError) as exc:
            eligible = False
            reason = "acquisition_or_inference_failure"
            event["error"] = f"{type(exc).__name__}: {exc}"
        trace.append(event)
        if early_stop and thresholds.decide(posterior, eligible) != "abstain":
            break
    return {
        "baseline_probability": p0,
        "baseline_prediction": int(p0 >= 0.5),
        "context": context,
        "posterior": posterior,
        "verdict": thresholds.decide(posterior, eligible),
        "eligible": eligible,
        "reason": reason,
        "steps": budget.steps,
        "dose": budget.dose,
        "acquisition_seconds": budget.seconds,
        "wall_seconds": time.perf_counter() - start,
        "initial_ood_score": ood,
        "trace": trace,
    }


def evaluate(
    panel: Panel,
    predictor: Predictor,
    bundle: Bundle,
    *,
    policy: str,
    max_steps: int,
    max_dose: float,
    max_seconds: float,
    thresholds: Thresholds,
    gates: Gates,
    seed: int = 42,
    dose_weight: float = 0.03,
    time_weight: float = 0.05,
    early_stop: bool = False,
    stop_if_negative: bool = True,
    fixed_order: tuple[int, ...] = (),
) -> list[dict]:
    rows = []
    for i, specimen in enumerate(panel.specimen_ids):
        provider = ReplayProvider(panel.images[i])
        selector = Selector(
            policy,
            seed_for(seed, specimen, policy),
            dose_weight,
            time_weight,
            fixed_order,
            stop_if_negative,
        )
        result = run_episode(
            provider,
            panel.actions,
            predictor,
            bundle,
            selector,
            Budget(max_steps, max_dose, max_seconds),
            thresholds,
            gates,
            early_stop,
        )
        label = int(panel.labels[i])
        target = (
            int(result["baseline_prediction"] == label)
            if bundle.target_kind == "correctness"
            else int(panel.reference[i])
        )
        rows.append(
            {
                "specimen_id": specimen,
                "group_id": panel.group_ids[i],
                "policy": policy,
                "budget_steps": max_steps,
                "budget_dose": max_dose,
                "budget_seconds": max_seconds,
                "task_label": label,
                "target": target,
                "target_kind": bundle.target_kind,
                "revealed_actions": provider.accesses,
                **result,
            }
        )
    return rows
