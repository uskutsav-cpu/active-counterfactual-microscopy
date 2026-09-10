from dataclasses import dataclass
from typing import Iterable

from .verdicts import Verdict


@dataclass(frozen=True, slots=True)
class TrialResult:
    truth: str
    verdict: Verdict
    dose_cost: float
    time_cost: float


def summarize_trials(trials: Iterable[TrialResult]) -> dict[str, float]:
    rows = list(trials)
    if not rows:
        raise ValueError("at least one trial is required")

    resolved = [r for r in rows if r.verdict is not Verdict.ABSTAIN]
    correct = [
        r
        for r in resolved
        if (r.truth == "biology" and r.verdict is Verdict.SUPPORTED)
        or (r.truth == "nuisance" and r.verdict is Verdict.FALSIFIED)
    ]
    false_support = [r for r in rows if r.truth == "nuisance" and r.verdict is Verdict.SUPPORTED]

    return {
        "n": float(len(rows)),
        "coverage": len(resolved) / len(rows),
        "correct_resolved_rate": len(correct) / max(1, len(resolved)),
        "false_support_rate": len(false_support) / len(rows),
        "mean_dose_cost": sum(r.dose_cost for r in rows) / len(rows),
        "mean_time_cost": sum(r.time_cost for r in rows) / len(rows),
    }
