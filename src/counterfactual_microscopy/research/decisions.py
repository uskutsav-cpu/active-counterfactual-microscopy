"""Abstention-aware verdicts and development-only threshold selection."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.stats import beta

from .validation import finite, integer


@dataclass(frozen=True)
class Thresholds:
    support: float | None = 0.9
    falsify: float | None = 0.1

    def __post_init__(self):
        if self.support is not None and finite("support", self.support, 0, 1) <= 0.5:
            raise ValueError("support threshold must exceed .5")
        if self.falsify is not None and finite("falsify", self.falsify, 0, 1) >= 0.5:
            raise ValueError("falsify threshold must be below .5")

    def decide(self, p: float, eligible: bool = True) -> str:
        finite("posterior", p, 0, 1)
        if not eligible:
            return "abstain"
        if self.support is not None and p >= self.support:
            return "supported"
        if self.falsify is not None and p <= self.falsify:
            return "falsified"
        return "abstain"


def binomial_upper(errors: int, n: int, alpha: float = 0.05) -> float:
    integer("errors", errors)
    integer("n", n)
    if errors > n or not 0 < alpha < 1:
        raise ValueError("invalid binomial parameters")
    return 1.0 if n == 0 or errors == n else float(beta.ppf(1 - alpha, errors + 1, n - errors))


def tune_thresholds(
    rows: list[dict], max_error: float = 0.15, min_groups: int = 10, confidence_alpha: float = 0.05
) -> tuple[Thresholds, dict]:
    """Select thresholds on DEVELOPMENT, with group-level errors and a fixed grid.

    A selected group is erroneous if ANY selected field's verdict is wrong.
    Bounds use a 24-candidate Bonferroni correction under exchangeable independent
    groups. There is NO claim of error control after arbitrary distribution shift.
    Separate policy/budget selections are not jointly multiplicity-controlled.
    """
    finite("max_error", max_error, 0, 1)
    integer("min_groups", min_groups, 1)
    if not rows:
        raise ValueError("empty development rows")
    choices = {}
    diagnostics = {}
    grid = np.linspace(0.55, 0.99, 12)
    for verdict in ["supported", "falsified"]:
        valid = []
        table = []
        for high in grid:
            threshold = float(high if verdict == "supported" else 1 - high)
            groups = {}
            for row in rows:
                selected = (
                    row["posterior"] >= threshold
                    if verdict == "supported"
                    else row["posterior"] <= threshold
                )
                if row["eligible"] and selected:
                    wrong = row["target"] == (0 if verdict == "supported" else 1)
                    groups[row["group_id"]] = groups.get(row["group_id"], False) or wrong
            n = len(groups)
            errors = int(sum(groups.values()))
            upper = binomial_upper(errors, n, confidence_alpha / 24)
            item = {"threshold": threshold, "groups": n, "error_groups": errors, "upper": upper}
            table.append(item)
            if n >= min_groups and upper <= max_error:
                valid.append(item)
        choices[verdict] = max(valid, key=lambda x: x["groups"])["threshold"] if valid else None
        diagnostics[verdict] = table
    threshold = Thresholds(choices["supported"], choices["falsified"])
    return (
        threshold,
        {
            "thresholds": asdict(threshold),
            "grid": diagnostics,
            "max_error": max_error,
            "unit": "selected group has any verdict error",
            "shift_guarantee": False,
            "both_disabled": all(v is None for v in choices.values()),
        },
    )
